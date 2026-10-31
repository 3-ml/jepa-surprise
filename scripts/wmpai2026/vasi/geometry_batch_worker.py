"""One score-blind DEVELOPMENT fixture: visible support contact vs disabled contact.

Fixed scene, no random seed selection. PyBullet states drive Kubric/Blender keys;
no post-render/context pixel copying. Not confirmatory or a lawfulness oracle.
"""
from __future__ import annotations
import argparse
import colorsys
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

FRAMES = 48
FPS = 12
HZ = 240
EVENT = 12
APPEARANCES = ('clean', 'persistent_hue', 'abrupt_event')
BRANCHES = ('lawful', 'inconsistent')
CONFIG = {'scene_id': 'dev-contact-0001', 'split': 'development', 'frames': FRAMES,
          'fps': FPS, 'simulation_hz': HZ, 'collision_disable_frame': EVENT,
          'gravity': [0, 0, -9.81], 'sphere_radius': 0.5, 'sphere_mass': 1.0,
          'sphere_initial_position': [0, -0.3, 7.5], 'restitution': 0.25,
          'support_position': [0, 0, -0.12], 'support_half_extents': [1.1, 0.7, 0.12],
          'catch_floor_position': [0, 0, -2.6], 'catch_floor_half_extents': [3, 3, 0.1],
          'camera_position': [9, -16, 6], 'camera_look_at': [0, 0, 2.0],
          'camera_orthographic_scale': 12, 'resolution': [256, 256],
          'base_color_rgb': [0.12, 0.55, 0.85], 'hue_degrees': 30,
          'samples_per_pixel': 64, 'denoiser': 'none', 'cycles_device': 'CUDA',
          'render_seed': 1025, 'minimum_target_area_fraction': 0.0025,
          'purpose': 'score-blind single development fixture, not confirmatory'}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def simulate(branch):
    import pybullet as p
    client = p.connect(p.DIRECT)
    try:
        p.setGravity(*CONFIG['gravity'], physicsClientId=client)
        p.setTimeStep(1 / HZ, physicsClientId=client)
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, numSolverIterations=100,
                                   physicsClientId=client)
        bodies = {}
        for name in ('support', 'catch_floor', 'left_rail', 'right_rail'):
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=CONFIG[name+'_half_extents'],
                                           physicsClientId=client)
            bodies[name] = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                            basePosition=CONFIG[name+'_position'], physicsClientId=client)
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=CONFIG['sphere_radius'], physicsClientId=client)
        ball = bodies['sphere'] = p.createMultiBody(baseMass=CONFIG['sphere_mass'],
            baseCollisionShapeIndex=shape, basePosition=CONFIG['sphere_initial_position'], physicsClientId=client)
        for body in bodies.values():
            p.changeDynamics(body, -1, restitution=CONFIG['restitution'], lateralFriction=0.5,
                             linearDamping=0, angularDamping=0, physicsClientId=client)
        trace = []
        disable_step = EVENT * HZ // FPS
        for step in range((FRAMES-1) * HZ // FPS + 1):
            stops = (branch == 'lawful') == (CONFIG['world'] == 'solid')
            enabled = CONFIG['world'] == 'solid' if step < disable_step else stops
            if step in (0, disable_step):
                p.setCollisionFilterPair(ball, bodies['support'], -1, -1, int(enabled), physicsClientId=client)
            position, orientation = p.getBasePositionAndOrientation(ball, physicsClientId=client)
            linear, angular = p.getBaseVelocity(ball, physicsClientId=client)
            contacts = p.getContactPoints(bodyA=ball, physicsClientId=client)
            trace.append({'step': step, 'time_seconds': step/HZ,
                          'position': position, 'quaternion_xyzw': orientation,
                          'linear_velocity': linear, 'angular_velocity': angular,
                          'support_collision_enabled': enabled, 'applied_external_force': [0, 0, 0],
                          'applied_external_impulse': [0, 0, 0],
                          'contacts': [list(c) for c in contacts]})
            if step < (FRAMES-1)*HZ//FPS:
                p.stepSimulation(physicsClientId=client)
        return {'branch': branch, 'body_ids': bodies, 'config': CONFIG, 'trace': trace,
                'pybullet_version': importlib.metadata.version('pybullet')}
    finally:
        p.disconnect(client)


def validate_states(lawful, inconsistent):
    a, b = lawful['trace'], inconsistent['trace']
    assert len(a) == len(b) == (FRAMES-1)*HZ//FPS + 1
    assert all(x == y for x, y in zip(a[:EVENT*HZ//FPS], b[:EVENT*HZ//FPS])), 'pre-event trace differs'
    support = lawful['body_ids']['support']
    contacts = lambda rows: [r for r in rows if any(c[2] == support and c[9] > 0 for c in r['contacts'])]
    stop, fall = (a,b) if CONFIG['world']=='solid' else (b,a)
    cs, cf = contacts(stop), contacts(fall)
    assert cs, 'stopping trajectory lacks normal contact force'
    assert not cf, 'fall-through trajectory has central-support contact'
    assert min(r['position'][2] for r in stop) > CONFIG['sphere_radius']-0.03, 'stopping sphere penetrates substantially'
    assert min(r['position'][2] for r in fall) < -1.5, 'falling sphere did not pass through opening'
    assert all(r['support_collision_enabled'] for r in stop[EVENT*HZ//FPS:])
    assert not any(r['support_collision_enabled'] for r in fall[EVENT*HZ//FPS:])
    rail_ids={lawful['body_ids'][name] for name in ('left_rail','right_rail')}
    assert not any(c[2] in rail_ids for rows in (a,b) for r in rows for c in r['contacts']), 'sphere touched side rail' 
    for rows in (a, b):
        assert all(np.isfinite(r['position'] + r['linear_velocity']).all() for r in rows)
    divergence = next(r['step'] for r, s in zip(a, b) if r['position'] != s['position'])
    return {'status': 'pass', 'shared_pre_event_substeps': EVENT*HZ//FPS,
            'world': CONFIG['world'], 'first_stopping_support_contact_step': cs[0]['step'],
            'first_position_divergence_step': divergence,
            'first_position_divergence_frame_ceil': int(np.ceil(divergence*FPS/HZ)),
            'lawful_support_contact_steps': len(contacts(a)), 'inconsistent_support_contact_steps': len(contacts(b)),
            'minimum_center_z': {branch: min(r['position'][2] for r in rows)
                                 for branch, rows in zip(BRANCHES, (a,b))},
            'limitation': 'solidity violation in solid world or unsupported stop in gap world; explicit collision-mask intervention'}


def render_cell(root, branch, appearance):
    import bpy
    import kubric as kb
    from kubric.renderer import Blender
    from PIL import Image
    started = time.perf_counter()
    state = json.loads((root / f'{branch}-states.json').read_text())
    CONFIG.clear(); CONFIG.update(state['config'])
    out = root / branch / appearance
    out.mkdir(parents=True, exist_ok=False)
    scene = kb.Scene(frame_start=0, frame_end=47, frame_rate=FPS, step_rate=HZ,
                     resolution=(256,256), gravity=CONFIG['gravity'])
    scratch=Path(__import__("os").environ['JEPA_GEOMETRY_SCRATCH'])/root.parent.name/root.name/branch/appearance
    renderer = Blender(scene, scratch_dir=scratch, samples_per_pixel=64, use_denoising=False)
    support = kb.Cube(name='support', position=CONFIG['support_position'], scale=CONFIG['support_half_extents'],
                      static=True, segmentation_id=2,
                      material=kb.PrincipledBSDFMaterial(color=kb.Color(0.55,0.55,0.55), roughness=0.8))
    floor = kb.Cube(name='catch_floor', position=CONFIG['catch_floor_position'], scale=CONFIG['catch_floor_half_extents'],
                    static=True, segmentation_id=3,
                    material=kb.PrincipledBSDFMaterial(color=kb.Color(0.3,0.3,0.3), roughness=0.8))
    color = CONFIG['base_color_rgb']
    h,s,v = colorsys.rgb_to_hsv(*color)
    changed = colorsys.hsv_to_rgb((h+CONFIG['hue_degrees']/360)%1,s,v)
    material = kb.PrincipledBSDFMaterial(color=kb.Color(*(changed if appearance=='persistent_hue' else color)), roughness=0.7)
    ball = kb.Sphere(name='target_sphere', scale=CONFIG['sphere_radius'], segmentation_id=1, material=material)
    scene.add([floor, ball])
    if CONFIG['world']=='solid':scene.add(support)
    for idx,name in enumerate(('left_rail','right_rail'),start=4):
        scene += kb.Cube(name=name,position=CONFIG[name+'_position'],scale=CONFIG[name+'_half_extents'],
                         static=True,segmentation_id=idx,
                         material=kb.PrincipledBSDFMaterial(color=kb.Color(0.55,0.55,0.55),roughness=0.8))
    for frame in range(FRAMES):
        row = state['trace'][frame*HZ//FPS]
        ball.position = row['position']
        q = row['quaternion_xyzw']
        ball.quaternion = (q[3],q[0],q[1],q[2])
        ball.keyframe_insert('position', frame)
        ball.keyframe_insert('quaternion', frame)
    if appearance == 'abrupt_event':
        # Set material directly in Blender, with CONSTANT interpolation at source frame 12.
        obj = ball.linked_objects[renderer]
        mat = obj.data.materials[0]
        node = next(n for n in mat.node_tree.nodes if n.bl_idname=='ShaderNodeBsdfPrincipled')
        socket = node.inputs['Base Color']
        for frame, rgb in ((0,color),(11,color),(12,changed),(47,changed)):
            socket.default_value = (*rgb,1)
            socket.keyframe_insert('default_value',frame=frame)
        for curve in mat.node_tree.animation_data.action.fcurves:
            for key in curve.keyframe_points:key.interpolation='CONSTANT'
    scene += kb.DirectionalLight(name='sun', position=(-3,-5,10),look_at=(0,0,1),intensity=2)
    scene.ambient_illumination = kb.Color(0.1,0.1,0.1)
    scene.camera = kb.OrthographicCamera(position=CONFIG['camera_position'],look_at=CONFIG['camera_look_at'],
                                         orthographic_scale=CONFIG['camera_orthographic_scale'])
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='CUDA'; prefs.get_devices()
    for d in prefs.devices:d.use=d.type=='CUDA'
    assert any(d.use for d in prefs.devices), 'no CUDA renderer'
    renderer.blender_scene.cycles.device='GPU'
    renderer.blender_scene.cycles.seed=CONFIG['render_seed']
    renderer.blender_scene.render.threads_mode='FIXED'
    renderer.blender_scene.render.threads=4
    # Check every rendered pose against source trace, not just a reused state hash.
    obj=ball.linked_objects[renderer]
    sampled_poses=[]
    for frame in range(FRAMES):
        renderer.blender_scene.frame_set(frame)
        xyz=[float(v) for v in obj.location]
        if not np.allclose(xyz,state['trace'][frame*HZ//FPS]['position'],atol=1e-6,rtol=0):
            raise ValueError('renderer pose differs from simulator state')
        sampled_poses.append(xyz)
    rendered=renderer.render(frames=list(range(FRAMES)),return_layers=('rgba','segmentation','depth'))
    assert rendered['rgba'].shape==(48,256,256,4)
    for name,arr in rendered.items():assert np.isfinite(arr).all(),name
    (out/'frames').mkdir()
    for i,rgb in enumerate(rendered['rgba']):Image.fromarray(rgb.astype(np.uint8)).save(out/'frames'/f'rgba_{i:05d}.png')
    target=rendered['segmentation'][...,0]==1
    np.savez_compressed(out/'passes.npz',segmentation=rendered['segmentation'],depth=rendered['depth'])
    area=target.sum(axis=(1,2)).tolist()
    renderer.save_state(str(out/'scene.blend'))
    result={'scene_id':CONFIG['scene_id'],'split':'development','physical_condition':branch,
            'appearance_condition':appearance,'frames':48,'context_frames':12,
            'event_frame_0based':12,'appearance_dose_degrees':0 if appearance=='clean' else 30,
            'state_trace_path':f'{branch}-states.json','state_trace_sha256':sha(root/f'{branch}-states.json'),
            'source_frame_sha256':[sha(out/'frames'/f'rgba_{i:05d}.png') for i in range(FRAMES)],
            'source_frame_paths':[(Path(branch)/appearance/'frames'/f'rgba_{i:05d}.png').as_posix() for i in range(FRAMES)],
            'target_area_pixels_per_frame':area,'target_area_min_target_fraction':min(area[12:])/65536,
            'rendered_target_positions':sampled_poses,'bpy':bpy.app.version_string,
            'kubric_revision':subprocess.check_output(['git','-C',str(Path(kb.__file__).parents[1]),'rev-parse','HEAD'],text=True).strip(),
            'elapsed_seconds':time.perf_counter()-started,'passes_sha256':sha(out/'passes.npz')}
    save(out/'clip.json',result)
    print(json.dumps({'completed':f'{branch}/{appearance}','min_area':min(area[12:]),'seconds':result['elapsed_seconds']}),flush=True)


def audit_renders(root):
    from PIL import Image,ImageDraw
    clips=[];arrays={}
    for branch in BRANCHES:
        for appearance in APPEARANCES:
            clip=json.loads((root/branch/appearance/'clip.json').read_text())
            arr=np.stack([np.asarray(Image.open(root/p).convert('RGB')) for p in clip['source_frame_paths']])
            arrays[(branch,appearance)]=arr
            assert sha(root/clip['state_trace_path'])==clip['state_trace_sha256']
            for path,expected in zip(clip['source_frame_paths'],clip['source_frame_sha256']):assert sha(root/path)==expected
            clips.append(clip)
    from canonical_development_context import canonicalize
    clips, canonical_audit = canonicalize(root, clips)
    save(root/'context-canonicalization-audit.json',canonical_audit)
    for clip in clips:
        key=(clip['physical_condition'],clip['appearance_condition'])
        arrays[key]=np.stack([np.asarray(Image.open(root/p).convert('RGB')) for p in clip['source_frame_paths']])
    problems=[]
    for appearance in APPEARANCES:
        if not np.array_equal(arrays[('lawful',appearance)][:12],arrays[('inconsistent',appearance)][:12]):
            problems.append(f'paired context pixels differ for {appearance}')
    for branch in BRANCHES:
        if not np.array_equal(arrays[(branch,'clean')][:12],arrays[(branch,'abrupt_event')][:12]):
            problems.append(f'abrupt context differs for {branch}')
        if np.array_equal(arrays[(branch,'clean')][:12],arrays[(branch,'persistent_hue')][:12]):
            problems.append(f'persistent hue not visible in context for {branch}')
        members=[c for c in clips if c['physical_condition']==branch]
        if len({c['state_trace_sha256'] for c in members})!=1:problems.append('appearance state mismatch')
        if any(c['rendered_target_positions']!=members[0]['rendered_target_positions'] for c in members):
            problems.append('appearance rendered pose mismatch')
    for clip in clips:
        if clip['target_area_min_target_fraction']<CONFIG['minimum_target_area_fraction']:
            problems.append(f"insufficient visibility {clip['physical_condition']}/{clip['appearance_condition']}")
    # Contact sheet, both branches and all appearance cells at context/event/future samples.
    indices=[0,11,12,14,16,24,47]
    sheet=Image.new('RGB',(len(indices)*128,6*152),'white');draw=ImageDraw.Draw(sheet)
    for row,(key,arr) in enumerate(arrays.items()):
        draw.text((3,row*152+2),'/'.join(key),fill='black')
        for col,i in enumerate(indices):sheet.paste(Image.fromarray(arr[i]).resize((128,128)),(col*128,row*152+22))
    sheet.save(root/'contact-sheet.png')
    manifest={'status':'pass' if not problems else 'failed_score_blind_validation','issues':problems,
              'config':CONFIG,'script_sha256':sha(__file__),'clips':clips,'scores_generated':False,
              'clip_count':len(clips),'frame_count':48*len(clips),
              'context_policy':'v2 roundoff-bounded canonical model inputs; raw renders retained; future PNGs unchanged'}
    save(root/'manifest.json',manifest)
    if problems:raise RuntimeError('; '.join(problems))
    print('ALL SCORE-BLIND RENDER CHECKS PASSED',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--render-cell',nargs=2,metavar=('BRANCH','APPEARANCE'))
    parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    if args.config:
        supplied=json.loads(args.config.read_text())
        CONFIG.clear();CONFIG.update(supplied)
    if args.render_cell:
        render_cell(args.output,*args.render_cell);return
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'config.json',CONFIG)
    a,b=simulate('lawful'),simulate('inconsistent')
    save(args.output/'lawful-states.json',a);save(args.output/'inconsistent-states.json',b)
    audit=validate_states(a,b)
    repeat=simulate('lawful')
    assert repeat==a,'lawful simulator replay not deterministic'
    audit['lawful_repeat_exact']=True
    import copy
    invalid = copy.deepcopy(a if CONFIG['world']=='solid' else b)
    for row in invalid['trace']:row['contacts']=[]
    try:
        validate_states(invalid,b) if CONFIG['world']=='solid' else validate_states(a,invalid)
    except AssertionError:
        audit['missing_stopping_contact_rejected']=True
    else:raise RuntimeError('validator accepted missing lawful contact')
    invalid=copy.deepcopy(b)
    invalid['trace'][0]['position']=[1,2,3]
    try:
        validate_states(a,invalid)
    except AssertionError:
        audit['different_context_state_rejected']=True
    else:raise RuntimeError('validator accepted altered pre-event state')
    save(args.output/'state-audit.json',audit)
    print(json.dumps(audit),flush=True)
    for branch in BRANCHES:
        for appearance in APPEARANCES:
            subprocess.run([sys.executable,__file__,'--output',str(args.output),
                            '--render-cell',branch,appearance],check=True)
    audit_renders(args.output)

if __name__=='__main__':main()
