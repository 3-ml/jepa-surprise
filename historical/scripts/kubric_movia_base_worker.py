import logging

import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
from kubric.safeimport.bpy import bpy
import numpy as np


SPAWN_REGION = [(-5, -5, 1), (5, 5, 5)]
VELOCITY_RANGE = [(-4.0, -4.0, 0.0), (4.0, 4.0, 0.0)]
CLEVR_OBJECTS = ("cube", "cylinder", "sphere")
KUBASIC_OBJECTS = (
    "cube",
    "cylinder",
    "sphere",
    "cone",
    "torus",
    "gear",
    "torus_knot",
    "sponge",
    "spot",
    "teapot",
    "suzanne",
)


parser = kb.ArgumentParser()

parser.add_argument("--objects_set", choices=["clevr", "kubasic"], default="clevr")
parser.add_argument(
    "--min_num_objects", type=int, default=3, help="minimum number of objects"
)
parser.add_argument(
    "--max_num_objects", type=int, default=10, help="maximum number of objects"
)

parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--background", choices=["clevr", "colored"], default="clevr")


parser.add_argument("--camera", choices=["clevr", "random"], default="clevr")


parser.add_argument(
    "--kubasic_assets",
    type=str,
    default="gs://kubric-public/assets/KuBasic/KuBasic.json",
)
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.add_argument("--samples_per_pixel", type=int, default=64)
parser.add_argument("--cycles_device", choices=["cpu", "cuda", "optix"], default="cpu")
parser.add_argument(
    "--denoiser", choices=["none", "openimage", "optix"], default="openimage"
)
parser.set_defaults(save_state=False, frame_end=24, frame_rate=12, resolution=256)
FLAGS = parser.parse_args()


scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
simulator = PyBullet(scene, scratch_dir)
renderer = Blender(
    scene,
    scratch_dir,
    samples_per_pixel=FLAGS.samples_per_pixel,
    use_denoising=FLAGS.denoiser != "none",
)
if FLAGS.cycles_device != "cpu":
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = FLAGS.cycles_device.upper()
    preferences.get_devices()
    devices = []
    for device in preferences.devices:
        device.use = device.type == FLAGS.cycles_device.upper()
        devices.append(
            {"name": device.name, "type": device.type, "use": bool(device.use)}
        )
    if not any(device["use"] for device in devices):
        raise RuntimeError(f"No {FLAGS.cycles_device.upper()} device: {devices}")
    renderer.blender_scene.cycles.device = "GPU"
else:
    devices = [{"name": "CPU", "type": "CPU", "use": True}]
if FLAGS.denoiser != "none":
    renderer.blender_scene.cycles.denoiser = {
        "openimage": "OPENIMAGEDENOISE",
        "optix": "OPTIX",
    }[FLAGS.denoiser]
scene.metadata["render_runtime"] = {
    "samples_per_pixel": FLAGS.samples_per_pixel,
    "cycles_device": FLAGS.cycles_device,
    "denoiser": FLAGS.denoiser,
    "devices": devices,
}
kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)


logging.info("Creating a large gray floor...")
floor_material = kb.PrincipledBSDFMaterial(roughness=1.0, specular=0.0)
scene += kubasic.create(
    "dome",
    name="floor",
    material=floor_material,
    scale=2.0,
    friction=FLAGS.floor_friction,
    restitution=FLAGS.floor_restitution,
    static=True,
    background=True,
)
if FLAGS.background == "clevr":
    floor_material.color = kb.Color.from_name("gray")
    scene.metadata["background"] = "clevr"
elif FLAGS.background == "colored":
    floor_material.color = kb.random_hue_color()
    scene.metadata["background"] = floor_material.color.hexstr


logging.info("Adding four (studio) lights to the scene similar to CLEVR...")
scene.add(kb.assets.utils.get_clevr_lights(rng=rng))
scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)


logging.info("Setting up the Camera...")
scene.camera = kb.PerspectiveCamera(focal_length=35.0, sensor_width=32)
if FLAGS.camera == "clevr":
    scene.camera.position = [7.48113, -6.50764, 5.34367] + rng.rand(3)
if FLAGS.camera == "random":
    scene.camera.position = kb.sample_point_in_half_sphere_shell(
        inner_radius=7.0, outer_radius=9.0, offset=0.1
    )
scene.camera.look_at((0, 0, 0))


num_objects = rng.randint(FLAGS.min_num_objects, FLAGS.max_num_objects + 1)
logging.info("Randomly placing %d objects:", num_objects)
for i in range(num_objects):
    if FLAGS.objects_set == "clevr":
        shape_name = rng.choice(CLEVR_OBJECTS)
        size_label, size = kb.randomness.sample_sizes("clevr", rng)
        color_label, random_color = kb.randomness.sample_color("clevr", rng)
    else:
        shape_name = rng.choice(KUBASIC_OBJECTS)
        size_label, size = kb.randomness.sample_sizes("uniform", rng)
        color_label, random_color = kb.randomness.sample_color("uniform_hue", rng)

    material_name = rng.choice(["metal", "rubber"])
    obj = kubasic.create(
        asset_id=shape_name,
        scale=size,
        name=f"{size_label} {color_label} {material_name} {shape_name}",
    )
    assert isinstance(obj, kb.FileBasedObject)

    if material_name == "metal":
        obj.material = kb.PrincipledBSDFMaterial(
            color=random_color, metallic=1.0, roughness=0.2, ior=2.5
        )
        obj.friction = 0.4
        obj.restitution = 0.3
        obj.mass *= 2.7 * size**3
    else:
        obj.material = kb.PrincipledBSDFMaterial(
            color=random_color, metallic=0.0, ior=1.25, roughness=0.7, specular=0.33
        )
        obj.friction = 0.8
        obj.restitution = 0.7
        obj.mass *= 1.1 * size**3

    obj.metadata = {
        "shape": shape_name.lower(),
        "size": size,
        "size_label": size_label,
        "material": material_name.lower(),
        "color": random_color.rgb,
        "color_label": color_label,
    }
    scene.add(obj)
    kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION, rng=rng)

    obj.velocity = rng.uniform(*VELOCITY_RANGE) - [obj.position[0], obj.position[1], 0]

    logging.info("    Added %s at %s", obj.asset_id, obj.position)


if FLAGS.save_state:
    logging.info(
        "Saving the simulator state to '%s' prior to the simulation.",
        output_dir / "scene.bullet",
    )
    simulator.save_state(output_dir / "scene.bullet")


logging.info("Running the simulation ...")
animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end + 1)


if FLAGS.save_state:
    logging.info("Saving the renderer state to '%s' ", output_dir / "scene.blend")
    renderer.save_state(output_dir / "scene.blend")


logging.info("Rendering the scene ...")
data_stack = renderer.render()


kb.compute_visibility(data_stack["segmentation"], scene.assets)
visible_foreground_assets = [
    asset
    for asset in scene.foreground_assets
    if np.max(asset.metadata["visibility"]) > 0
]
visible_foreground_assets = sorted(
    visible_foreground_assets,
    key=lambda asset: np.sum(asset.metadata["visibility"]),
    reverse=True,
)

data_stack["segmentation"] = kb.adjust_segmentation_idxs(
    data_stack["segmentation"], scene.assets, visible_foreground_assets
)
scene.metadata["num_instances"] = len(visible_foreground_assets)


kb.write_image_dict(data_stack, output_dir)
kb.post_processing.compute_bboxes(data_stack["segmentation"], visible_foreground_assets)


logging.info("Collecting and storing metadata for each object.")
kb.write_json(
    filename=output_dir / "metadata.json",
    data={
        "flags": vars(FLAGS),
        "metadata": kb.get_scene_metadata(scene),
        "camera": kb.get_camera_info(scene.camera),
        "instances": kb.get_instance_info(scene, visible_foreground_assets),
    },
)
kb.write_json(
    filename=output_dir / "events.json",
    data={
        "collisions": kb.process_collisions(
            collisions, scene, assets_subset=visible_foreground_assets
        ),
    },
)

kb.done()
