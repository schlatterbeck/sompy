#!/usr/bin/python3

import numpy as np
from scipy.special import jv as bessel, hankel1 as hankel

# Constants
c    = 299.8
mu0  = 4e-7 * np.pi
eps0 = 1e-12 / (c**2 * mu0)

class Sommerfeld:

    def __init__ (self, eps_r, sigma = None, fmhz = None):
        """ The original implementation used the case where sigma < 0 to
            specify the imag component of eps_r directly (and ignore fmhz)
            We make sigma optional and allow eps_r to be complex.
            Note: The original implementation in somnec uses 59.96 as an
            approximation of mu0 * c / (2*pi)
        """
        # epscf is the *relative* epsilon (divided by eps0)
        if sigma is not None:
            assert fmhz is not None
            f = fmhz * 1e6
            self.epscf = eps_r -1j * sigma / (2 * np.pi * f * eps0)
        else:
            self.epscf = eps_r
        # In the fortran implementation the common block cgrid contains
        # the three arrays ar1, ar2, and ar2 with the values. We compute
        # the xyz input values here replacing DXA, DYA (the increment
        # value), XSA, YSA (the start values), NXA, NYA (the number of
        # increments aka the dimension)
        # The X values are R (R = sqrt (rho ** 2 + (z + h) ** 2))
        # The Y values are THETA (THETA = atan ((z + h) / rho)
        self.dxa = np.array ([.02, .05, .1])
        self.dya = np.array ([np.pi / 18, np.pi / 36, np.pi / 18])
        self.nxa = np.array ([11, 17, 9])
        self.nya = np.array ([10,  5, 8])
        self.xsa = np.array ([0., .2, .2])
        self.ysa = np.array ([0., 0., np.pi / 9])
        xx    = (self.xsa, self.xsa + (self.nxa - 1) * self.dxa, self.nxa)
        yy    = (self.ysa, self.ysa + (self.nya - 1) * self.dya, self.nya)
        xgrid = [np.linspace (*x) for x in zip (*xx)]
        ygrid = [np.linspace (*y) for y in zip (*yy)]
        # avoid computing things twice:
        xgrid [0] = xgrid [0][:-1]
        ygrid [2] = ygrid [2][1:]
        xygrid = [np.meshgrid (x, y) for x, y in zip (xgrid, ygrid)]
        r     = np.concatenate ([x [0].flatten () for x in xygrid])
        theta = np.concatenate ([x [1].flatten () for x in xygrid])
        rho   = self.rho = r * np.cos (theta)
        zph   = self.zph = r * np.sin (theta)
        rho [rho < 1e-7] = 1e-8
        zph [zph < 1e-7] = 0.0
        self.is_hankel = self.zph < 2 * self.rho

        self.ck2sq = 4 * np.pi ** 2
        self.ck1sq = self.ck2sq * np.conjugate (self.epscf)
        self.ck1   = np.sqrt (self.ck1sq)
        self.tkmag = 100 * abs (self.ck1)
        self.tsmag = 100 * self.ck1 * np.conjugate (self.ck1)
        self.cksm  = self.ck2sq / (self.ck1sq + self.ck2sq)
        self.ct1   = .5    * (self.ck1sq - self.ck2sq)
        self.ct2   = .125  * (self.ck1sq ** 2 - self.ck2sq ** 2)
        self.ct3   = .0625 * (self.ck1sq ** 3 - self.ck2sq ** 3)
        # We do not define ck2 = 2*np.pi and ck1r = ck1.real
        # We also do not define JH which is 1 for the hankel form an 0
        # for bessel, we use is_hankel above
    # end def __init__

    def evlua (erv, ezv, erh, eph):
        """ Controls the integration contour in the complex lambda plane for
            evaluation of the Sommerfeld integrals
        """
        dlt = np.max (self.zph, self.rho)
        cp1 = 0            +.8j * np.pi
        cp2 =  1.2 * np.pi -.4j * np.pi
        cp3 = 2.04 * np.pi -.4j * np.pi
    # end def evlua

    def gshank (start, dela, sum, nans, seed, ibk, bk, delb):
        """ Integrates the 6 Sommerfeld integrals from start to infinity
            (until convergence) in lambda. At the break point, bk, the
            step increment may be changed from dela to delb. Shank's
            algorithm to accelerate convergence of a slowly converging
            series is used.
        """
        crit = 1e-4
        maxh = 20
        dlt  = dela
    # end def gshank

    def rom1 (self, n, nx):
        nm    = 1 << 17
        rx    = 1e-4
        lstep = 0
        z     = np.zeros (self.rho.shape)
        ze    = 1.
        ep    = 1 / (1e4 * nm) # for epsilon comparison
        zend  = ze - ep
        sum   = np.zeros (self.rho.shape + (n,), dtype = complex)
        todo  = np.ones  (self.rho.shape, dtype = bool)
        ns    = np.ones  (self.rho.shape, dtype = int) * nx
        nt    = np.zeros (self.rho.shape)
        g1    = self.saoa (z)
        g2    = np.zeros (g1.shape, dtype = complex)
        g3    = np.zeros (g1.shape, dtype = complex)
        g4    = np.zeros (g1.shape, dtype = complex)
        g5    = np.zeros (g1.shape, dtype = complex)
        # This used to be label 2
        while True:
            dz = 1 / ns
            cnd = z + dz > ze
            dz [cnd] = (ze - z) [cnd]
            todo [dz < ep] = False
            if not todo.any ():
                break
            dzot  = dz * .5
            dzotn = dzot [..., np.newaxis]
            g3 [todo] = self.saoa ((z + dzot) [todo])
            g5 [todo] = self.saoa ((z + dz) [todo])
            # This used to be label 4
            ngo2 = todo
            while ngo2.any ():
                nogo = np.zeros (g3.shape, dtype = bool)
                t00 = (g1 + g5) * dzotn
                t01 = (t00 + dz [..., np.newaxis] * g3) * .5
                t10 = (4 * t01 - t00) / 3
                # test convergence of 3 point romberg result
                tri = self.test (t01, t10, 0.)
                nogo [(tri.real > rx) | (tri.imag > rx)] = 1
                ngo = np.sum (nogo, axis = 1, dtype = bool)
                go  = np.logical_not (ngo)
                sum [go] = sum [go] + t10 [go]
                nt  [go] = nt  [go] + 2
                # This is only be called for ngo == 1:
                # It won't do anything if the prev produced nogo=0
                g2 [ngo] = self.saoa (z + dz * .25, ngo)
                g4 [ngo] = self.saoa (z + dz * .75, ngo)
                t02 = np.zeros (t01.shape, dtype = complex)
                t11 = np.zeros (t01.shape, dtype = complex)
                t20 = np.zeros (t01.shape, dtype = complex)
                t02 [ngo] = (t01 [ngo] + dzotn * (g2 + g4)) [ngo] * .5
                t11 [ngo] = ( 4 * t02 [ngo] - t01 [ngo]) / 3
                t20 [ngo] = (16 * t11 [ngo] - t10 [ngo]) / 15
                nogo2 = np.zeros (g3.shape, dtype = bool)
                # test convergence of 5 point romberg result
                tri = np.zeros (g3.shape, dtype = complex)
                tri [ngo] = self.test (t11 [ngo], t20 [ngo], 0.)
                nogo2 [(tri.real > rx) | (tri.imag > rx)] = 1
                ngo2 = np.sum (nogo2, axis = 1, dtype = bool)
                go   = np.logical_not (ngo2)
                sum [go] = sum [go] + t20 [go]
                nt  [go] = nt  [go] + 1

                if ngo2.any:
                    u, c = np.unique (ngo2, return_counts = True)
                    d = dict (zip (u, c))
                    print ('ngo2: %d' % d [True])
                # Here the ngo2 part should be handled (goto 13)
                if (ngo2 & (ns >= nm)).any ():
                    # The part in 'if lstep' is an error message where we
                    # would produce incorrect results. We raise an exception.
                    raise ValueError \
                        ( 'rom1: step size limited at a=%s, b=%s'
                        % (self.a, self.b)
                        )
                nt [ngo2 ] = 0
                ns [ngo2] = ns [ngo2] * 2
                dz [ngo2] = 1 / ns [ngo2]
                dzot [ngo2] = dz [ngo2] * .5
                dzotn = dzot [..., np.newaxis]
                g5 [ngo2] = g3 [ngo2]
                g3 [ngo2] = g2 [ngo2]
            z = z + dz
            if (z > zend).all ():
                break
            g1 = g5
            cnd = (nt >= 4) & (ns > nx)
            ns [cnd] = ns [cnd] / 2
            nt [cnd] = 1
            # implicit continue here (goto 2)
    # end def rom1

    def saoa (self, t, cond = None):
        """ Computes the integrand for each of the 6 Sommerfeld
            integrals for source and observer above ground
            The is_hankel boolean matrix indicates which of the elements
            should be computed with the hankel variant.
        """
        if cond is None:
            cond = np.ones (self.is_hankel.shape, dtype = bool)
        if self.is_hankel.shape != cond.shape:
            import pdb; pdb.set_trace ()
        is_hankel = self.is_hankel & cond
        is_bessel = np.logical_not (self.is_hankel) & cond
        dxl = (self.b - self.a)
        xl  = self.a + dxl * t
        cgam1 = np.zeros (self.is_hankel.shape, dtype = complex)
        cgam2 = np.zeros (self.is_hankel.shape, dtype = complex)
        b0    = np.zeros (self.is_hankel.shape, dtype = complex)
        b0p   = np.zeros (self.is_hankel.shape, dtype = complex)
        b, bp, cg1, cg2 = self.saoa_bessel \
            (xl [is_bessel], self.rho [is_bessel])
        cgam1 [is_bessel] = cg1
        cgam2 [is_bessel] = cg2
        b0    [is_bessel] = b
        b0p   [is_bessel] = bp
        b, bp, cg1, cg2 = self.saoa_hankel \
            (xl [is_hankel], self.rho [is_hankel])
        cgam1 [is_hankel] = cg1
        cgam2 [is_hankel] = cg2
        b0    [is_hankel] = b
        b0p   [is_hankel] = bp
        cgam1 = cgam1 [cond]
        cgam2 = cgam2 [cond]
        b0    = b0    [cond]
        b0p   = b0p   [cond]
        xl    = xl [cond]
        xlr  = xl * np.conjugate (xl)
        dgam = np.zeros (xlr.shape, dtype = complex)
        dgam [xlr < self.tsmag] = cgam2 - cgam1
        sign = np.ones (xlr.shape, dtype = bool)
        sign [xlr < self.tsmag] = 0
        sign [(xl.imag >= 0) & (xl.real < 2 * np.pi)] = -1
        cnd = (sign == 1) & (xl.real <= self.ck1.real)
        dgam [cnd] = cgam2 [cnd] - cgam1 [cnd]
        sign [cnd] = 0
        cnd = (sign != 0)
        xxl = (1 / (xl * xl)) [cnd]
        dgam [cnd] = \
            ( sign [cnd]
            * ((self.ct3 * xxl + self.ct2) * xxl + self.ct1)
            / xl [cnd]
            )
        den2 = self.cksm * dgam \
             / (cgam2 * (self.ck1sq * cgam2 + self.ck2sq * cgam1))
        den1 = 1 / (cgam1 + cgam2) - self.cksm / cgam2
        com  = dxl * xl * np.exp (-cgam2 * self.zph [cond])
        ans = np.zeros ((6,) + com.shape, dtype = complex)
        ans [5] = com * b0 * den1 / self.ck1
        com  = com * den2
        ans [0] = np.zeros (ans [5].shape)
        ans [3] = np.zeros (ans [5].shape)
        nr0  = self.rho [cond] != 0
        r0   = np.logical_not (nr0)
        xxl = xl [r0] * xl [r0] * .5
        ans [0][r0] = ans [3][r0] = -com [r0] * xxl
        b0p [nr0] = b0p [nr0] / self.rho [cond][nr0]
        xxl = xl [nr0]
        ans [0][nr0] = -com [nr0] * xxl * (b0p [nr0] + b0 [nr0] * xxl)
        ans [3][nr0] = com [nr0] * xxl * b0p [nr0]
        ans [1] = com * cgam2 * cgam2 * b0
        ans [2] = -ans [3] * cgam2 * self.rho [cond]
        ans [4] = com * b0
        return np.array (ans).T
    # end def saoa

    def saoa_bessel (self, xl, rho):
        b0    =  2 * bessel (0, xl * rho)
        b0p   = -2 * bessel (1, xl * rho)
        cgam1 = np.sqrt (xl * xl - self.ck1sq)
        cgam2 = np.sqrt (xl * xl - self.ck2sq)
        cond  = cgam1.real == 0
        cgam1 [cond] = 0 - 1j * np.abs (cgam1 [cond].imag)
        cond  = cgam2.real == 0
        cgam2 [cond] = 0 - 1j * np.abs (cgam2 [cond].imag)
        return b0, b0p, cgam1, cgam2
    # end def saoa_bessel

    def saoa_hankel (self, xl, rho):
        b0    =  hankel (0, xl * rho)
        b0p   = -hankel (1, xl * rho)
        com   = xl - self.ck1
        cgam1 = np.sqrt (xl + self.ck1) * np.sqrt (com)
        cond = (com.real < 0) & (com.imag > 0)
        cgam1 [cond] = -cgam1 [cond]
        com   = xl - 2 * np.pi
        cgam2 = np.sqrt (xl + 2 * np.pi) * np.sqrt (com)
        cond = (com.real < 0) & (com.imag > 0)
        cgam2 [cond] = -cgam2 [cond]
        return b0, b0p, cgam1, cgam2
    # end def saoa_hankel

    def test (self, f1, f2, dmin):
        den = np.abs (f2.real)
        tr  = np.abs (f2.imag)
        ti  = np.zeros (tr.shape, dtype = float)
        den [den < tr]   = tr [den < tr]
        den [den < dmin] = dmin
        cond = den >= 1e-37
        tr [cond] = np.abs ((f1.real [cond] - f2.real [cond]) / den [cond])
        ti [cond] = np.abs ((f1.imag [cond] - f2.imag [cond]) / den [cond])
        cond = np.logical_not (cond)
        tr [cond] = 0
        ti [cond] = 0
        return tr + 1j * ti
    # end def test

# end class Sommerfeld

if __name__ == '__main__':
    s = Sommerfeld (4.0, .001, 10.0)
