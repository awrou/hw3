#!/usr/bin/env python

import newton
import unittest
import functions as F
import numpy as N

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(2.0)
        self.assertEqual(x,-2.0)

    def testQuad(self):
        f = lambda x : x**2 + 3*x + 2
        solver = newton.Newton(f, tol=1.e-15, maxiter=20)
        x = solver.solve(-3.0)
        self.assertAlmostEqual(x,-2.0)
        x = solver.solve(0.0)
        self.assertAlmostEqual(x,-1.0)

# The root of this eq is at -2.0. At a guess >= 8 default exception is raised
#    def testmRangeExcp(self):
#        f = lambda x : 3.0 * x + 6.0
#        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
#        x0 = 8.0 #At a range > 10 from root exception is raised
#        x = solver.solve(x0)
#        self.assertEqual(x,-2.0)

        #Test that it moves toward root convergence
        #Note that this will fail to converge and therefore raise a value error
    def testSingleStep(self):
        f = lambda x : 4*x**2-4
        solver =  newton.Newton(f, tol=1.e-15, maxiter=1)
        x0 = 1.5
        x = solver.solve(x0)
        self.assertTrue(abs(-1-x)<abs(-1-x0))

          

if __name__ == "__main__":
    unittest.main()
