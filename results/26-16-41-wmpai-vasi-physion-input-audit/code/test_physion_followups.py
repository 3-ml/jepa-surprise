import unittest
import numpy as np
from physion_common import appearance_variant,static_variant,hue,validate_population,B_CONDITIONS
from physion_statistics import hybrid_l1,bootstrap_pairs,rank_correlation

class PhysionTests(unittest.TestCase):
 def setUp(self):self.clip=np.random.default_rng(4).integers(0,256,(16,8,8,3),dtype=np.uint8)
 def test_zero_hue_exact(self):self.assertTrue(np.array_equal(hue(self.clip,0),self.clip))
 def test_persistent_and_abrupt_share_future(self):
  for kind in ['hue','brightness']:
   p=appearance_variant(self.clip,'persistent_'+kind);a=appearance_variant(self.clip,'target_only_'+kind)
   self.assertTrue(np.array_equal(p[12:],a[12:]));self.assertTrue(np.array_equal(a[:12],self.clip[:12]));self.assertFalse(np.array_equal(p[:12],self.clip[:12]))
 def test_brightness_integer_definition(self):
  p=appearance_variant(self.clip,'persistent_brightness');self.assertTrue(np.array_equal(p,(self.clip.astype(np.uint16)*4//5).astype(np.uint8)))
 def test_original_identity(self):self.assertTrue(np.array_equal(appearance_variant(self.clip,'original'),self.clip))
 def test_static_frame_mapping(self):
  for name,index in [('static_first',0),('static_middle',8),('static_last',15)]:self.assertTrue(np.array_equal(static_variant(self.clip,name),np.repeat(self.clip[index:index+1],16,axis=0)))
 def test_hybrid_identity(self):
  rng=np.random.default_rng(5);x=[rng.normal(size=(8,12)) for _ in range(4)];r=hybrid_l1(*x);self.assertLess(r['reconstruction_abs_error'],1e-10)
 def test_encoder_only_contribution(self):
  p=np.ones((4,8));t=np.zeros((4,8));r=hybrid_l1(p,t,p,t+2);self.assertEqual(r['predictor_contribution'],0)
 def test_predictor_only_contribution(self):
  p=np.ones((4,8));t=np.zeros((4,8));r=hybrid_l1(p,t,p+2,t);self.assertEqual(r['target_contribution'],0)
 def test_rank_ties_and_missing(self):
  self.assertAlmostEqual(rank_correlation([1,1,3],[4,4,2]),-1);self.assertIsNone(rank_correlation([1,1],[1,2]))
 def test_pair_and_layout_bootstrap_constant(self):
  rows=[{'scenario':s,'layout_group':str(i//2),'x':2.0} for s in ['a','b','c'] for i in range(4)]
  for layout in [False,True]:
   r=bootstrap_pairs(rows,['x'],4,draws=100,family=6,layout=layout);self.assertEqual(r['metrics']['x']['mean'],2);self.assertEqual(r['metrics']['x']['descriptive95'],[2,2])
 def test_population_counts_and_pair_labels(self):
  pop=[{'relative_path':f'{i}/{j}','pair_id':str(i),'label':str(j),'scenario':str(i//32)} for i in range(96) for j in range(2)]
  self.assertEqual(len(validate_population(pop)),96)
  pop[1]['label']='0'
  with self.assertRaises(AssertionError):validate_population(pop)
 def test_duplicate_video_rejected(self):
  pop=[{'relative_path':f'{i}/{j}','pair_id':str(i),'label':str(j),'scenario':str(i//32)} for i in range(96) for j in range(2)];pop[1]['relative_path']=pop[0]['relative_path']
  with self.assertRaises(AssertionError):validate_population(pop)
if __name__=='__main__':unittest.main(verbosity=2)
