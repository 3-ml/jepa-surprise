"""Model-free path, exact dependency and documentation contracts; stdlib only."""
import ast,collections,hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
def load(name):return json.loads((ROOT/'release'/name).read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class PathContracts(unittest.TestCase):
 def setUp(self):self.rows=load('path-contracts.json')['contracts']
 def evaluate(self,row):return Path(eval(row['expression'],{'__file__':str(ROOT/row['path'])})).resolve()
 def test_no_generic_or_literal_placeholders(self):
  for folder in ['scripts','results','release','protocol','docs']:
   for p in (ROOT/folder).rglob('*'):
    if p.suffix not in {'.py','.md','.json','.jsonl','.csv','.txt'}:continue
    text=p.read_text(encoding='utf-8-sig')
    self.assertNotIn('PUBLIC_'+'EXTERNAL_INPUT',text,str(p))
    self.assertNotIn('JEPA_'+'EXTERNAL_ROOT',text,str(p))
 def test_all_included_json_records_parse(self):
  entries=load('source-map.json')
  for e in entries:
   p=ROOT/e['path']
   if p.suffix=='.json':json.loads(p.read_text(encoding='utf-8-sig'))
   elif p.suffix=='.jsonl':
    for line in p.read_text(encoding='utf-8-sig').splitlines():
     if line.strip():json.loads(line)
 def test_every_expression_is_present_in_source_ast(self):
  trees={}
  for r in self.rows:
   tree=trees.setdefault(r['path'],ast.parse((ROOT/r['path']).read_text(encoding='utf-8-sig')))
   expected=ast.dump(ast.parse(r['expression'],mode='eval').body,include_attributes=False)
   self.assertIn(expected,{ast.dump(n,include_attributes=False) for n in ast.walk(tree)},r)
 def test_missing_external_inputs_fail_without_fallback(self):
  for r in self.rows:
   if r['environment']:
    with patch.dict(os.environ,{},clear=True):
     with self.assertRaises(KeyError,msg=str(r)):self.evaluate(r)
 def test_files_directories_and_relationships(self):
  envs={r['environment'] for r in self.rows if r['environment']}
  kinds={e:{r['kind'] for r in self.rows if r['environment']==e and not r['suffix']} for e in envs}
  for e,ks in kinds.items():self.assertLessEqual(len(ks),1,(e,ks))
  with tempfile.TemporaryDirectory() as tmp:
   env={e:str(Path(tmp)/e) for e in envs}
   for e in env:
    p=Path(env[e])
    if kinds[e]=={'file'}:p.write_bytes(b'fixture, not model/data')
    else:p.mkdir()
   with patch.dict(os.environ,env,clear=True):
    for r in self.rows:
     p=self.evaluate(r)
     if r['environment']:
      base=Path(env[r['environment']]);self.assertTrue(p==base or base in p.parents,r)
      if r['kind'].startswith('file'):p.parent.mkdir(parents=True,exist_ok=True);p.touch(exist_ok=True);self.assertTrue(p.is_file(),r)
      else:p.mkdir(parents=True,exist_ok=True);self.assertTrue(p.is_dir(),r)
     else:self.assertEqual(p,ROOT,r)
    # Distinct original identities may alias only the documented same-checkpoint/source/project copies.
    resolved=collections.defaultdict(set)
    for r in self.rows:resolved[self.evaluate(r)].add(r['original_identity'])
    alias_roles={'JEPA_VJEPA2G_CHECKPOINT','JEPA_VJEPA2_SOURCE',None}
    for p,ids in resolved.items():
     if len(ids)>1:self.assertTrue(all(r['environment'] in alias_roles for r in self.rows if self.evaluate(r)==p),(p,ids))
 def test_reviewed_nonaliasing_regressions(self):
  cases={
   'scripts/wmpai2026/vasi/score_development24.py':['CHECKPOINT','MODEL_ROOT','refroot','refcsv'],
   'release/final256/original/frozen-source/scripts/wmpai2026/vasi/score_development24.py':['CHECKPOINT','MODEL_ROOT','refroot','refcsv'],
   'scripts/wmpai2026/alex/gpu4gb_offload.py':['CHECKPOINT','VJEPA2_ROOT','SHARD_DIR','DATASET_ZIP'],
   'scripts/wmpai2026/alex/a4_multifamily16_frame_baselines.py':['FR','RESP'],
   'release/final256/audit-source/audit_final256.py':['FINAL','REPO','QUEUE','checkpoint'],
   'scripts/wmpai2026/vasi/export_lewm_actions.py':['ROOT','SOURCE'],
  }
  env={r['environment']:'/synthetic/'+r['environment'] for r in self.rows if r['environment']}
  with patch.dict(os.environ,env,clear=True):
   for path,names in cases.items():
    rows=[next(r for r in self.rows if r['path']==path and r['binding']==n) for n in names]
    self.assertEqual(len({self.evaluate(r) for r in rows}),len(names),path)
   grid=[r for r in self.rows if r['path']=='scripts/finalize_vjepa2g_grid_replication.py']
   self.assertEqual({r['suffix'] for r in grid},{'vjepa2g-grid','vjepa2g-grid-stage-e'})
   self.assertEqual(len({self.evaluate(r)/'benchmark_results.json' for r in grid}),2)
   # Contact clean frames remain beneath the same reference render tree, CSV is separate.
   rows=[r for r in self.rows if r['path']=='scripts/wmpai2026/vasi/geometry_development24.py']
   frames=next(r for r in rows if r['kind']=='directory');csv=next(r for r in rows if r['kind']=='file')
   self.assertEqual(frames['suffix'],'contact-development-1/lawful/clean/frames')
   self.assertEqual(csv['suffix'],'contact-development-scores-1/scores.csv')
 def test_exact_frozen_core_wiring_and_native_body(self):
  deps=load('dependency-map.json');core=ROOT/deps['physion_core']['path']
  self.assertEqual(digest(core),deps['physion_core']['sha256'])
  body=next(n for n in ast.parse(core.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='score_section')
  for name in ['physion-static','physion-appearance','geometry-transfer']:
   p=ROOT/'release'/name/'frozen-source.zip/canonical_score.py';tree=ast.parse(p.read_text())
   assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='CORE_SOURCE' for x in n.targets))
   root=p.parent;actual=eval(compile(ast.Expression(assignment.value),'core-contract','eval'),{'ROOT':root})
   self.assertEqual(actual,core)
   self.assertIn(deps['physion_core']['sha256'],p.read_text())
   new=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score_tensor')
   old_ast=ast.dump(ast.Module(body=body.body[3:],type_ignores=[]),include_attributes=False)
   new_ast=ast.dump(ast.Module(body=new.body[1:],type_ignores=[]),include_attributes=False)
   self.assertEqual(old_ast,new_ast,name)
 def test_variants_and_exact_dependency_edges(self):
  deps=load('dependency-map.json')
  for v in deps['preserved_variants']:
   self.assertEqual(digest(ROOT/v['path']),v['sha256'],v)
  for edge in deps['exact_edges']:
   self.assertEqual(digest(ROOT/edge['target']),edge['sha256'],edge)
 def test_scientific_protocols_and_scope(self):
  protocols=[p for p in (ROOT/'release').rglob('protocol.md')]
  self.assertEqual(len(protocols),7)
  physion=[p.read_bytes() for p in protocols if 'final256' not in p.parts]
  self.assertEqual(len(set(physion)),1)
  forbidden=['Approval is recorded verbatim','Hard supervisor stop','launch-permit handshake','kill-on-close Job Object','Vasi asked for in PR','Send Vasi','Upload the clip records','These are the changes I would like']
  for p in ROOT.rglob('*.md'):
   if 'historical' in p.relative_to(ROOT).parts or p.name.startswith('reproducibility'):continue
   text=p.read_text()
   for phrase in forbidden:self.assertNotIn(phrase,text,str(p))
def run_checks(verbosity=1):
 result=unittest.TextTestRunner(verbosity=verbosity).run(unittest.defaultTestLoader.loadTestsFromTestCase(PathContracts))
 if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':run_checks(2)
