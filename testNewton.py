#!/usr/bin/env python

import newton
import unittest
import functions as F
import numpy as N
import math

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(2.0)
        self.assertEqual(x,-2.0)

    def testPoly(self):
        f = F.Polynomial([1,3,2])
        solver = newton.Newton(f, tol=1.e-15, maxiter=20)
        x = solver.solve(-3.0)
        self.assertAlmostEqual(x,-2.0)
        x = solver.solve(0.0)
        self.assertAlmostEqual(x,-1.0)

# The root of this eq is at -2.0. At a guess x0 >= 8 default exception is raised
    def testmRangeExcp(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=10, mRange=10)
        x0 = 9 #At a range > 10 from root exception is raised
        x = solver.solve(x0)
        self.assertEqual(x,-2.0)

    def testAnalyticJacobiPoly(self):
        f = F.Polynomial([1,3,2])
        _Df = lambda x: 2*x+3
        solver = newton.Newton(f, tol=1.e-15, maxiter=25, Df=_Df)
        x = solver.solve(-0.5)
        self.assertAlmostEqual(x,-1.0)
        x = solver.solve(-3.0)
        self.assertAlmostEqual(x,-2.0)

#    def testAnalyticJacobiLog(self):
#        f = F.Logarithmic([4, 6, 3])
#        _Df = lambda x: 4.0/x
#        solver = newton.Newton(f, tol=1.e-15, maxiter=10, Df=_Df)
#        x0 = 0.5
#        x = solver.solve(x0)
#        print(x)
#        self.assertAlmostEqual(x,1.0/(6.0*N.exp(3.0/4.0)))

        #Test convergence exception
    def testConverge(self):
        f = F.Polynomial([1,0,-1])
        solver = newton.Newton(f,tol=1.e-15, maxiter=3)
        x = solver.solve(2.0)
        self.assertEqual(x, 1.0)

        #Test that it moves toward root convergence for a single step
        #Note that this will fail to converge and therefore raise a value error
    def testSingleStep(self):
        f = lambda x : 4*x**2-4
        solver =  newton.Newton(f, tol=1.e-15, maxiter=1)
        x0 = 1.5
        x = solver.solve(x0)
        self.assertTrue(abs(-1-x)<abs(-1-x0))

    def test2dimApprox(self):
#        A = N.matrix("1. 2.; 3. 4.")
#        def f(x):
#            return A * x
        f = lambda x: N.matrix([[5*x[0,0]-2*x[1,0]-13],[2*x[0,0]+x[1,0]-7]])
        solver = newton.Newton(f, tol=1.e-8, maxiter=20)
        x0 = N.matrix([[4],[2]])
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, N.matrix([[3.],[1.]]))

    def test2dimAnalyt(self):
        f = lambda x: N.matrix([[5*x[0,0]-2*x[1,0]-13],[2*x[0,0]+x[1,0]-7]])
        _Df = lambda x: N.matrix([[5,-2],[2, 1]])
        solver = newton.Newton(f, tol=1.e-8, maxiter=20, Df=_Df)
        x0 = N.matrix([[4],[2]])
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, N.matrix([[3.],[1.]]))
        
if __name__ == "__main__":
    unittest.main()
