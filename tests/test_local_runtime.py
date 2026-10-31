"""CPU-only launcher tests. Optional NumPy/Pillow cases use synthetic images/models."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import local_runtime as runtime


def inventory(root, **overrides):
    data = dict(files={p.relative_to(root).as_posix(): runtime.sha(p) for p in root.rglob("*")
                       if p.is_file() and p != root / "local-inputs.json"},
                study_files=[], workspace={}, roles={})
    data.update(overrides)
    (root / "local-inputs.json").write_text(json.dumps(data))
    return data


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalRuntimeTests(unittest.TestCase):
    def test_content_seal_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "tensor"
            p.write_bytes(b"sealed")
            inventory(root)
            runtime.verify_inventory(root)
            p.write_bytes(b"tamper")
            with self.assertRaises(ValueError): runtime.verify_inventory(root)
            p.unlink()
            with self.assertRaises(ValueError): runtime.verify_inventory(root)
            p.mkdir()
            with self.assertRaises(ValueError): runtime.verify_inventory(root)
            p.rmdir()
            p.symlink_to(root / "local-inputs.json")
            with self.assertRaises(ValueError): runtime.verify_inventory(root)

    def test_inventory_schema_and_nested_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            nested = root / "sub/local-inputs.json"
            nested.write_text("{}")
            inventory(root)
            runtime.verify_inventory(root)
            nested.write_text('{"changed":true}')
            with self.assertRaises(ValueError): runtime.verify_inventory(root)
            (root / "local-inputs.json").write_text('{"files":{},"files":{}}')
            with self.assertRaises(ValueError): runtime.verify_inventory(root)
            for bad in [[], {}, {"files": [], "study_files": [], "workspace": {}, "roles": {}}]:
                (root / "local-inputs.json").write_text(json.dumps(bad))
                with self.assertRaises(ValueError): runtime.verify_inventory(root)

    def test_paths_layout_and_alias_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("model", "controls", "model/nested"):
                (root / name).mkdir(exist_ok=True)
            records = inventory(root, workspace={runtime.DESKTOP: "model", runtime.CONTROLS: "controls"})
            runtime.check_layout("physion", root, records)
            for name in ("", ".", "../outside", "a/../b", "a//b", "/absolute", "a\\b", "a/./b"):
                with self.assertRaises(ValueError): runtime.relative(root, name)
            for target in ("model/nested", "model", "../escape"):
                records["workspace"][runtime.CONTROLS] = target
                with self.assertRaises(ValueError): runtime.check_layout("physion", root, records)
            records["workspace"] = {runtime.DESKTOP: "model", runtime.CONTROLS: "controls"}
            records["study_files"] = ["model/config.json"]
            with self.assertRaises(ValueError): runtime.check_layout("physion", root, records)
            records["workspace"]["unknown/nested"] = "controls"
            with self.assertRaises(ValueError): runtime.check_layout("physion", root, records)

    def test_geometry_empty_assets_not_ready_cold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("freeze.json", "geometry-coefficients.json"):
                (root / name).write_text("{}")
            inventory(root, study_files=["freeze.json", "geometry-coefficients.json"])
            with patch.dict(sys.modules, {"torch": None, "transformers": None}):
                with self.assertRaisesRegex(ValueError, "workspace"):
                    runtime.check_study("geometry", root)

    def test_freeze_and_coefficient_anchored_to_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("freeze.json", "geometry-coefficients.json"):
                (root / name).write_text("{}")
            records = inventory(root)
            with patch.object(runtime, "check_layout"), patch.object(runtime, "verify_inventory", return_value=records):
                for study in ("physion", "geometry"):
                    with self.assertRaisesRegex(ValueError, "hash mismatch"):
                        runtime.check_study(study, root)

    def test_selected_scientific_freeze_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"; model.mkdir()
            controls = root / "controls"; controls.mkdir()
            config = model / "config.json"; config.write_text("qualified")
            control = controls / "control-manifest.jsonl"; control.write_text("qualified-control")
            external = {
                '${JEPA_VASI_WORKSPACE}\\' + runtime.DESKTOP + '\\model\\config.json': runtime.sha(config),
                '${JEPA_VASI_WORKSPACE}\\' + runtime.CONTROLS + '\\control-manifest.jsonl': runtime.sha(control),
                '${JEPA_VASI_WORKSPACE}\\' + runtime.DESKTOP + '\\controller.py': '0' * 64,
            }
            # Path mapper is tested against a synthetic published record, not user authority.
            (model / "model").mkdir(); config.rename(model / "model/config.json")
            records = dict(workspace={runtime.DESKTOP: "model", runtime.CONTROLS: "controls"})
            with patch.object(runtime, "frozen", return_value={"external_files": external}):
                runtime.scientific_external("physion", root, records)
                (model / "model/config.json").write_text("tampered")
                with self.assertRaises(ValueError): runtime.scientific_external("physion", root, records)

    def test_source_root_import_overrides_are_rejected_without_execution(self):
        import importlib.machinery
        import py_compile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "model-source"
            (source / "src").mkdir(parents=True)
            member = source / "src/models.py"; member.write_text("# pinned model source")
            marker = root / "executed"
            payload = root / "payload.py"
            payload.write_text(f"from pathlib import Path; Path({str(marker)!r}).touch()")
            freeze = {"external_files": {"${JEPA_VJEPA2_SOURCE}/src/models.py": runtime.sha(member)}}
            records = dict(workspace={}, roles={"JEPA_VJEPA2_SOURCE": "model-source"})
            git = types.SimpleNamespace(stdout=runtime.REVISION)
            with patch.object(runtime, "frozen", return_value=freeze), patch.object(runtime.subprocess, "run", return_value=git):
                runtime.scientific_external("geometry", root, records)
                for suffix in importlib.machinery.all_suffixes():
                    override = source / ("src" + suffix)
                    if suffix == ".pyc":
                        py_compile.compile(str(payload), cfile=str(override), doraise=True)
                    else:
                        override.write_bytes(payload.read_bytes() if suffix == ".py" else b"not an extension binary")
                    importlib.invalidate_caches()
                    # Inspect resolution only. No unrecorded module is imported, even
                    # when an invalid binary advertises an extension-module origin.
                    spec = importlib.machinery.PathFinder.find_spec("src", [str(source)])
                    self.assertEqual(spec.origin, str(override), suffix)
                    with self.assertRaisesRegex(ValueError, "src import override"):
                        runtime.scientific_external("geometry", root, records)
                    override.unlink()
                runtime.scientific_external("geometry", root, records)
            self.assertFalse(marker.exists())

    def test_physion_installed_source_hashes_with_unchanged_versions(self):
        paths = {
            "transformers": ["transformers/models/vjepa2/modeling_vjepa2.py",
                             "transformers/models/vjepa2/configuration_vjepa2.py", "transformers/modeling_utils.py"],
            "torch": ["torch/nn/functional.py", "torch/nn/modules/module.py", "torch/nn/modules/normalization.py"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); external = {}
            for package_paths in paths.values():
                for tail in package_paths:
                    file = root / tail; file.parent.mkdir(parents=True, exist_ok=True)
                    file.write_text("recorded " + tail)
                    external["external-record/pinned/" + file.name] = runtime.sha(file)
            freeze = dict(versions={"torch": "recorded", "transformers": "recorded"}, external_files=external)
            dist = types.SimpleNamespace(locate_file=lambda tail: root / tail)
            with patch.object(runtime, "frozen", return_value=freeze), \
                 patch.object(runtime.importlib.metadata, "version", return_value="recorded"), \
                 patch.object(runtime.importlib.metadata, "distribution", return_value=dist) as distribution:
                runtime.check_versions("physion")
                self.assertEqual([c.args[0] for c in distribution.call_args_list], ["transformers", "torch"])
                for tail in paths["torch"]:
                    file = root / tail; original = file.read_bytes()
                    file.write_bytes(b"modified, same package version")
                    with self.assertRaisesRegex(ValueError, file.name):
                        runtime.check_versions("physion")
                    file.write_bytes(original)
                runtime.check_versions("physion")

    def test_fresh_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"; bundle.mkdir()
            runtime.fresh_output(root / "new", bundle)
            for path in (root / "new", runtime.ROOT / "forbidden", bundle / "nested", root):
                with self.assertRaises(ValueError): runtime.output_path(path, bundle)
            with self.assertRaises(ValueError): runtime.output_path(runtime.ROOT.parent, bundle)

    def test_dispatch_validates_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bundle = root / "bundle"; bundle.mkdir()
            inventory(bundle)
            out = root / "never-created"
            with patch.object(runtime, "verify_published"):
                with self.assertRaises(ValueError): runtime.dispatch("geometry", bundle, out, 5)
            self.assertFalse(out.exists())

    def test_published_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source").write_text("code")
            (root / "release-manifest.json").write_text(json.dumps({"entries": [{"path": "source", "bytes": 4, "sha256": runtime.sha(root / "source")}]}))
            runtime.verify_published(root)
            (root / "source").write_text("evil")
            with self.assertRaises(ValueError): runtime.verify_published(root)

    def test_subprocess_success_errors_timeout(self):
        runtime.run_child([sys.executable, "-c", "pass"], 5)
        with self.assertRaises(subprocess.CalledProcessError):
            runtime.run_child([sys.executable, "-c", "raise SystemExit(7)"], 5)
        with self.assertRaises(subprocess.TimeoutExpired):
            runtime.run_child([sys.executable, "-c", "import time; time.sleep(10)"], .05)
        for value in (0, -1, float('nan'), float('inf'), 28801):
            with self.assertRaises(ValueError): runtime.timeout_value(value)

    def test_owned_group_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "unexpected-survivor"
            grandchild = f"import time; from pathlib import Path; time.sleep(.3); Path({str(marker)!r}).touch()"
            leader = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{grandchild!r}])"
            runtime.run_child([sys.executable, "-c", leader], 5)
            import time
            time.sleep(.4)
            self.assertFalse(marker.exists())

    def test_synthetic_child_dispatch_and_output_wiring(self):
        # Complements actual-entrypoint tests: only this plumbing fixture substitutes
        # scientific authority/source, in the test child, never through a CLI option.
        code = r"""
import sys, types, json
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import local_runtime as r
root = Path(sys.argv[2]); r.ROOT = root / 'publication'; r.ROOT.mkdir()
source = r.ROOT / 'release/geometry-transfer/frozen-source.zip'; source.mkdir(parents=True)
(source/'study_common.py').write_text('ROOT = None')
(source/'extract_geometry.py').write_text("from study_common import *\nimport sys,os\nsys.path.insert(0,os.environ['JEPA_VJEPA2_SOURCE'])\nimport src.models\ndef main():\n wait_permit('geometry'); deadline(); inference_guard(); assert src.models.PINNED; (ROOT/'child-result').write_text('pass')\n")
bundle = root / 'bundle'; bundle.mkdir()
model = bundle/'verified-model'; (model/'src').mkdir(parents=True)
(model/'src/models.py').write_text('PINNED = True')
# cwd and PYTHONPATH both offer an overriding regular module. Neither may run.
marker = root/'untrusted-executed'
(root/'src.py').write_text(f"from pathlib import Path; Path({str(marker)!r}).touch(); raise RuntimeError('cwd override')")
import os
# Also expose CWD explicitly, simulating a later import-path insertion. The
# namespace binding must protect imports independently of -I.
os.chdir(root)
sys.path.insert(0, str(root))
records = dict(workspace={}, roles={'JEPA_VJEPA2_SOURCE':'verified-model'}, study_files=[])
def verify(*args, **kw):
    assert 'src' not in sys.modules or sys.modules['src'].__path__ == [str(model/'src')]
    assert (model/'src/models.py').read_text() == 'PINNED = True'
    return records
r.scientific_freeze = verify
fake = types.ModuleType('torch'); fake.cuda = types.SimpleNamespace(is_available=lambda:True,mem_get_info=lambda i:(1,2),get_device_name=lambda i:'synthetic');sys.modules['torch']=fake
out = root / 'out'
r.dispatch('geometry', bundle, out, 5)
assert (out/'child-result').read_text() == 'pass'
assert (out/'workspace').is_dir()
assert not (bundle/'child-result').exists()
assert not marker.exists()
assert sys.modules['src'].__path__ == [str(model/'src')]
"""
        with tempfile.TemporaryDirectory() as tmp:
            runtime.run_child([sys.executable, "-I", "-B", "-c", code, str(runtime.ROOT/'tools'), tmp], 5,
                              {**os.environ, "PYTHONPATH": tmp})

    def test_loaded_src_modules_fail_before_binding(self):
        for name in ("src", "src.models"):
            with patch.dict(sys.modules, {name: types.ModuleType(name)}):
                with self.assertRaisesRegex(RuntimeError, "already loaded"):
                    runtime.bind_model_source(Path("verified-source"))

    def test_recorded_versions_are_enforced(self):
        with patch.object(runtime, "frozen", return_value={"versions": {"torch": "recorded"}}), \
             patch.object(runtime.importlib.metadata, "version", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "version mismatch"):
                runtime.check_versions("geometry")

    def test_actual_physion_and_geometry_import_root_contract(self):
        for study, entry in (("physion", "run_physion"), ("geometry", "extract_geometry")):
            source = runtime.ROOT / "release" / runtime.STUDIES[study] / "frozen-source.zip"
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, JEPA_VASI_WORKSPACE=tmp):
                common = module_from(source / "study_common.py", "study_common")
                fake = types.ModuleType("torch")
                fake.cuda = types.SimpleNamespace(is_available=lambda: True, mem_get_info=lambda i: (1024, 2048), get_device_name=lambda i: 'synthetic')
                scientific = unittest.mock.Mock(return_value={"verified": True})
                with patch.dict(sys.modules, {"study_common": common, "torch": fake}):
                    runtime.install_runtime(common, Path(tmp), 5, scientific)
                    actual = module_from(source / (entry + ".py"), entry)
                    self.assertEqual(actual.ROOT, Path(tmp))
                    actual.wait_permit(study); actual.deadline(); actual.inference_guard(); actual.check_freeze()
                    self.assertEqual(scientific.call_count, 2)
                    # The real entrypoint reaches its first external scientific read/import.
                    if study == "physion":
                        with self.assertRaises(FileNotFoundError): actual.main()
                    else:
                        with patch.dict(sys.modules, {"score_development24": None}):
                            with self.assertRaises(ModuleNotFoundError): actual.main()

    def test_frozen_dependency_roots_are_the_exact_variants(self):
        original = runtime.ROOT / "release/final256/original/frozen-source"
        tree = ast.parse((original / "scripts/wmpai2026/vasi/final256_score.py").read_text())
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertTrue({"main", "make_scorer", "reference_gate"} <= names)
        dependency = runtime.load(runtime.ROOT / "release/dependency-map.json")
        for item in dependency["preserved_variants"]:
            self.assertEqual(runtime.sha(runtime.ROOT / item["path"]), item["sha256"])


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL"), "requires offline NumPy/Pillow CPU fixtures")
class FrozenFinal256ContractTests(unittest.TestCase):
    def test_actual_gate_and_score_with_synthetic_backend(self):
        import numpy as np
        from PIL import Image
        original = runtime.ROOT / "release/final256/original/frozen-source/scripts/wmpai2026/vasi"
        class Tensor:
            def __init__(self, value): self.value = value
            def __getitem__(self, key): return Tensor(self.value[key])
            def contiguous(self): return self
            def numpy(self): return self.value
        def normalized(rgb, resolution): return Tensor(np.asarray(rgb, dtype=np.float32))
        def tsha(tensor): return __import__('hashlib').sha256(tensor.numpy().tobytes()).hexdigest()
        def load_clip(folder): return np.stack([np.asarray(Image.open(p).convert('RGB')) for p in sorted(folder.glob('*.png'))])
        class Scorer:
            def __init__(self): self.cache = {}; self.context_predictors = {}; self.models = types.SimpleNamespace(close=lambda: None)
            def score(self, rgb):
                norm = normalized(rgb, 256)
                return dict(score=.5, temporal_tubelet_l1=[.5]*18, context_sha=tsha(norm[:12]), target_input_sha256=tsha(norm[12:]), predictor_sha256=tsha(norm[:12]))
        fake_torch = types.ModuleType("torch")
        fake_torch.from_numpy = lambda x: x
        fake_torch.set_num_threads = lambda n: None
        fake_torch.__version__ = 'synthetic-only'
        fake_torch.cuda = types.SimpleNamespace(get_device_name=lambda: 'synthetic', empty_cache=lambda:None)
        fake_torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)), cudnn=types.SimpleNamespace(allow_tf32=False))
        core = types.ModuleType('run_intphys2_dense'); core.normalize_video = normalized
        native = types.ModuleType('score_development24')
        native.sha = runtime.sha; native.tsha = tsha; native.core = core
        native.CHECKPOINT_SHA = 'synthetic'; native.MODEL_REV = 'synthetic'
        native.pilot = types.SimpleNamespace(load_clip=load_clip, target_hold=lambda x:x, target_shuffle=lambda x,*a:(x,list(range(18))))
        native.baselines = types.SimpleNamespace(frame_features=lambda *a,**kw:dict(static=0.,motion=0.,shuffled=0.))
        common = module_from(original / 'final256_common.py', 'final256_common')
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'torch':fake_torch, 'run_intphys2_dense':core, 'score_development24':native, 'final256_common':common}):
            root = Path(tmp); sid = 'synthetic-0001'
            common.check_sources = lambda *a: None  # Synthetic publication only; production checks are separately tested.
            common.save(root / 'run-lock.json', {})
            common.save(root / 'freeze.json', {})
            common.save(root / 'inspected-registry.json', {'normalized_context_sha256':[]})
            worlds = {}
            for wi, world in enumerate(('solid','gap')):
                data = root / sid / world; clips=[]
                for bi, branch in enumerate(('lawful','inconsistent')):
                    state = data / (branch + '-states.json'); common.save(state, {'synthetic':branch})
                    for ai, app in enumerate(('clean','persistent_hue','abrupt_event')):
                        folder = data / branch / app; folder.mkdir(parents=True)
                        paths=[]
                        for i in range(48):
                            path = folder / f'{i:03d}.png'
                            value = 20 + wi*40 + (10 if app == 'persistent_hue' else 0) + (bi if i>=12 else 0)
                            Image.fromarray(np.full((2,2,3),value,dtype=np.uint8)).save(path)
                            paths.append(str(path.relative_to(data)))
                        hashes = [runtime.sha(data / p) for p in paths]
                        clips.append(dict(scene_id=sid, split='confirmatory', physical_condition=branch, appearance_condition=app,
                                          source_frame_paths=paths, source_frame_sha256=hashes, raw_source_frame_paths=paths,
                                          raw_source_frame_sha256=hashes, state_trace_path=state.name, state_trace_sha256=runtime.sha(state), target_area_min_target_fraction=.1))
                common.save(data / 'manifest.json', {'clips':clips})
                worlds[world] = {'manifest_sha256':runtime.sha(data/'manifest.json')}
            common.save(root / 'render-seal.json', {'eligible_count':1,'sealed_before_scoring':True,'candidates':[dict(scene_id=sid,status='eligible',worlds=worlds)]})
            gate = module_from(original / 'final256_input_gate.py', 'synthetic_gate')
            with patch.object(sys,'argv',['gate','--root',str(root)]): gate.main()
            self.assertEqual(common.load(root/'input-gate.json')['cells'],12)
            # Frozen scorer requires >=231 eligible scenes; use its actual preflight path
            # rather than weakening that production scientific gate for a tiny fixture.
            contact = root/'workspace/contact-development-1/lawful/clean/frames'
            contact.mkdir(parents=True)
            Image.fromarray(np.zeros((2,2,3),np.uint8)).save(contact/'000.png')
            scores = root/'workspace/contact-development-scores-1'; scores.mkdir()
            import csv
            with (scores/'scores.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=['physical_condition','appearance_condition','score','tubelet_scores_json'])
                writer.writeheader();writer.writerow(dict(physical_condition='lawful',appearance_condition='clean',score=.5,tubelet_scores_json=json.dumps([.5]*18)))
            score = module_from(original/'final256_score.py','synthetic_score')
            with patch.dict(os.environ,JEPA_VASI_WORKSPACE=str(root/'workspace')), patch.object(score,'make_scorer',Scorer), patch.object(sys,'argv',['score','--preflight-out',str(root/'parity.json')]):
                score.main()
            self.assertEqual(common.load(root/'parity.json')['status'],'pass')
            # An altered frame must fail the real original input gate.
            (root/sid/'solid/lawful/clean/000.png').write_bytes(b'tampered')
            with patch.object(sys,'argv',['gate','--root',str(root)]):
                with self.assertRaisesRegex(ValueError,'PNG/hash'): gate.main()


if __name__ == '__main__': unittest.main()
