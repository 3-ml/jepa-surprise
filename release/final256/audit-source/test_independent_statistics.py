import unittest
import numpy as np
from independent_statistics import weighted_auc,compare
class Tests(unittest.TestCase):
 def test_auc_ties_and_weighted_replicates(self):
  rng=np.random.default_rng(5)
  for worlds in [1,2]:
   x=rng.integers(0,4,(7,worlds,2));draws=rng.integers(0,7,(19,7));weights=np.array([np.bincount(d,minlength=7) for d in draws]);actual=weighted_auc(x,weights)
   for i,idx in enumerate(draws):
    y=x[idx];p=y[:,:,1].ravel();n=y[:,:,0].ravel();expected=((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])).mean();self.assertAlmostEqual(actual[i],expected,places=15)
 def test_all_tied(self):self.assertEqual(weighted_auc(np.ones((5,2,2)),np.ones((1,5)))[0],.5)
 def test_perfect_and_reversed(self):
  x=np.tile([0.,1.],(5,2,1));self.assertEqual(weighted_auc(x,np.ones((1,5)))[0],1);self.assertEqual(weighted_auc(x[:,:,::-1],np.ones((1,5)))[0],0)
 def test_compare_rejects_changed_output(self):
  with self.assertRaises(AssertionError):compare({'a':[.1]},{'a':[.10001]})
if __name__=='__main__':unittest.main()
