import unittest,tempfile
from pathlib import Path
import numpy as np
from tensor_store import pack,unpack,tensor_info,save_tensor,load_tensor
from geometry_analysis import tables,fit,design
from physion_common import static_variant,appearance_variant,validate_population
from physion_statistics import hybrid_l1,bootstrap_pairs,rank_correlation
class Tests(unittest.TestCase):
 def test_exact_frame_roundtrip(self):
  x=np.zeros((1,16,3,256,256),np.float32);x[:,1,0,0,0]=-0.;x[:,2]=1.25;a,b=pack(x);self.assertEqual(x.tobytes(),unpack(a,b).tobytes());self.assertEqual(len(a),3)
 def test_corruption(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.npz';x=np.ones((1,16,3,256,256),np.float32);r=save_tensor(p,x);self.assertEqual(load_tensor(p,r).tobytes(),x.tobytes());p.write_bytes(p.read_bytes()+b'x')
   with self.assertRaises(AssertionError):load_tensor(p,r)
 def test_nonfinite_rejected(self):
  x=np.full((1,16,3,256,256),np.nan,np.float32)
  with self.assertRaises(AssertionError):pack(x)
 def test_bad_index_rejected(self):
  with self.assertRaises(AssertionError):unpack(np.zeros((1,3,256,256),np.float32),np.ones(16,np.uint8))
 def test_bad_dtype_rejected(self):
  with self.assertRaises(AssertionError):pack(np.zeros((1,16,3,256,256),np.float16))
 def test_future_context(self):
  clip=np.random.default_rng(1).integers(0,256,(16,8,8,3),dtype=np.uint8)
  for kind in ['hue','brightness']:
   p=appearance_variant(clip,'persistent_'+kind);t=appearance_variant(clip,'target_only_'+kind);self.assertTrue(np.array_equal(p[12:],t[12:]));self.assertTrue(np.array_equal(t[:12],clip[:12]))
 def test_static_frames(self):
  x=np.arange(16*8*8*3,dtype=np.uint8).reshape(16,8,8,3)
  for name,k in [('static_first',0),('static_middle',8),('static_last',15)]:self.assertTrue(np.array_equal(static_variant(x,name),np.repeat(x[k:k+1],16,axis=0)))
 def test_hybrid_identity(self):
  rng=np.random.default_rng(2);h=hybrid_l1(*[rng.normal(size=(8,8)) for _ in range(4)]);self.assertLess(h['reconstruction_abs_error'],1e-10)
 def test_tie_rank(self):self.assertAlmostEqual(rank_correlation([1,1,2,3],[2,2,4,6]),1.)
 def test_constant_rank(self):self.assertIsNone(rank_correlation([1,1],[2,3]))
 def test_population_reject(self):
  with self.assertRaises(AssertionError):validate_population([])
 def fixture(self):
  rng=np.random.default_rng(7);table=[]
  for i in range(5):
   for w in [0,1]:
    base=[w,0.,0.,*rng.normal(size=3)];d=rng.normal(size=5);table.append({'scene_id':str(i),'world':'gap' if w else 'solid','appearance':'clean','base':base,'displacement':d.tolist(),'indicator':float(w),'y':float(3+2*base[3]+d[0])})
  return table
 def test_dev_fit_no_test_labels(self):
  t=self.fixture();coeff=fit(t);p=design(t,0,'combined')@np.array(coeff['models']['block_09']['combined']['beta']);self.assertLess(np.max(np.abs(p-np.array([r['y'] for r in t]))),1e-10)
 def test_geometry_only_excludes_pixel_stats(self):
  t=self.fixture();a=design(t,0,'geometry_only')
  for r in t:r['base'][3:]=[100,200,300]
  self.assertTrue(np.array_equal(a,design(t,0,'geometry_only')))
 def test_shuffle_preserves_conditions(self):
  t=self.fixture();p=design(t,0,'combined',np.array([4,3,2,1,0]));a=design(t,0,'combined');self.assertTrue(np.array_equal(p[:,:7],a[:,:7]));self.assertTrue(np.array_equal(p[:,-2].reshape(5,2),a[:,-2].reshape(5,2)[::-1]))
 def test_bootstrap_pair_strata(self):
  rows=[{'pair_id':str(i),'scenario':s,'layout_group':str(i//2),'x':float(i)} for s in ['a','b','c'] for i in range(4)];o=bootstrap_pairs(rows,['x'],1,draws=100);self.assertEqual(o['metrics']['x']['mean'],1.5);self.assertEqual(o['stratum_groups'],{'a':4,'b':4,'c':4})
 def test_bootstrap_layout(self):
  rows=[{'pair_id':str(i),'scenario':'a','layout_group':str(i//2),'x':1.} for i in range(4)];o=bootstrap_pairs(rows,['x'],1,draws=100,layout=True);self.assertEqual(o['metrics']['x']['descriptive95'],[1.,1.])
 def test_native_adapter(self):
  from canonical_score import verify_adapter
  self.assertTrue(verify_adapter()['native_body_AST_exact'])
 def test_tables_reject_missing(self):
  with self.assertRaises((AssertionError,KeyError)):tables([{'clip_id':'x','scene_id':'a','world':'solid','physical_condition':'lawful','appearance_condition':'clean'}],np.ones((1,5,1408),np.float32))
if __name__=='__main__':unittest.main(verbosity=2)
