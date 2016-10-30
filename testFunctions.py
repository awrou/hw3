#!/usr/bin/env python

import functions as F
import numpy as N
import unittest
import math

class TestFunctions(unittest.TestCase):
    def testApproxJacobian1(self):
        slope = 3.0
        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (1,1))
        self.assertAlmostEqual(Df_x, slope)

    def testApproxJacobian2(self):
        A = N.matrix("1. 2.; 3. 4.")
        def f(x):
            return A * x
        x0 = N.matrix("5; 6")
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (2,2))
        N.testing.assert_array_almost_equal(Df_x, A)

         # Checking in higher dimension
    def testApproxJacobian3(self):
        A = N.matrix("1. 2. 7; 3. 4. 9; 3. 5. 1")
        def f(x):
            return A * x
        x0 = N.matrix("5; 6; 9")
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (3,3))
        N.testing.assert_array_almost_equal(Df_x, A)

    def testPolynomial(self):
        # p(x) = x^2 + 2x + 3
        p = F.Polynomial([1, 2, 3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p(x), x**2 + 2*x + 3)

#    def testAnalyticJacobi(self):

    def testLogarithmic(self):
        l = F.Logarithmic([4, 6, 3])
        for x in N.linspace(2,3,11):
            self.assertEqual(l(x), 4*N.log(6*x)+3)

    def testExponential(self):
        e = F.Exponential([2, 2, -1])
        for x in N.linspace(2,3,11):
            self.assertEqual(e(x), 2*N.exp(2*x)-1)

    def testAnalyticAccu(self):
        slope = 3.0
        def f(x):
            return slope * x + 5.0 #Write desired function here
        x0 = 2.0
        dx = 1.e-3
        DfApprox = F.ApproximateJacobian(f, x0, dx)
        DfAnalytic = slope #Write analytical Jacobian here
        self.assertAlmostEqual(DfAnalytic,DfApprox)
        
if __name__ == '__main__':
    unittest.main()

    



