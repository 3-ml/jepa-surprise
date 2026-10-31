import unittest
import numpy as np
from matched_common import CONDITIONS,OLD,NEW,PARENTS,contrasts,decompose
from matched_analyse import bootstrap
class Tests(unittest.TestCase):
 def test_grid(self):
  self.assertEqual(len(CONDITIONS),17);self.assertEqual(len(set(CONDITIONS)),17);self.assertEqual(len(OLD),6);self.assertEqual(len(NEW),11)
  self.assertFalse(set(OLD)&set(NEW));self.assertEqual(256*17*3,13056)
 def test_contrasts(self):
  c=contrasts();self.assertEqual(len(c),12);self.assertEqual(len(set(n for n,_ in c)),12)
  for _,weights in c:self.assertEqual(sum(weights.values()),0);self.assertTrue(set(weights)<=set(CONDITIONS))
 def test_parent_order(self):
  for name,parent in PARENTS.items():self.assertLess(CONDITIONS.index(parent),CONDITIONS.index(name))
 def test_decomposition(self):
  rng=np.random.default_rng(44)
  for n in [1,192,98304]:
   p,t,p0,t0=[rng.normal(size=n).astype(np.float32) for _ in range(4)];d=decompose(p,t,p0,t0)
   self.assertLess(d['reconstruction_abs_error'],1e-10)
   self.assertAlmostEqual(sum(d[k] for k in ['predictor_shift_sq','target_shift_sq','shift_cross','residual_predictor','residual_target']),d['delta_mse_float64'],places=10)
 def test_target_only(self):
  p=np.ones(192,np.float32);t=np.zeros(192,np.float32);d=decompose(p,t,p,t+1)
  self.assertEqual(d['predictor_shift_sq'],0);self.assertEqual(d['shift_cross'],0);self.assertEqual(d['residual_predictor'],0)
 def test_bootstrap_pairing(self):
  a=np.arange(8,dtype=np.float64);m=np.stack([a,-a,np.zeros(8)],axis=1);x=bootstrap(m,100,123)
  np.testing.assert_array_equal(x[:,0],-x[:,1]);np.testing.assert_array_equal(x[:,2],np.zeros(100));np.testing.assert_array_equal(x,bootstrap(m,100,123))
if __name__=='__main__':unittest.main()
