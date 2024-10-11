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
        nts   = 4
        rx    = 1e-4
        lstep = 0
        z     = 0.
        ze    = 1.
        s     = 1.
        ep    = s / (1e4 * nm)
        zend  = ze - ep
        sum   = np.zeros (self.rho.shape + (n,))
        ns    = nx
        nt    = np.zeros (self.rho.shape)
        g1    = self.saoa (z)
        g2    = np.zeros (g1.shape)
        g3    = np.zeros (g1.shape)
        g4    = np.zeros (g1.shape)
        g5    = np.zeros (g1.shape)
        while True:
            dz = s / ns
            if z + dz > ze:
                dz = ze - z
                if dz < ep:
                    break
            dzot = dz * .5
            g3 = saoa (z + dzot)
            g5 = saoa (z + dz)
            #while something:
            nogo = np.zeros (g3.shape, dtype = bool)
            t00 = (g1 + g5) * dzot
            t01 = (t00 + dz * g3) * .5
            t10 = (4 * t01 - t00) / 3
            # test convergence of 3 point romberg result
            tri = test (t01, t10, 0.)
            # FIXME: nogo must be the *or* of all 6 conditions
            nogo [np.logical_or (tri.real > rx, tri.imag > rx)] = 1
            # FIXME: And here we again need a nogo with dimension added
            go = np.logical_not (nogo)
            sum [go] = sum [go] + t10 [go]
            nt [go]  = nt [go] + 2
            # This should only be called for nogo == 1:
            if 1:
                g2 [nogo] = self.saoa (z + dz * .25, nogo)
                g4 [nogo] = self.saoa (z + dz * .75, nogo)
                t02 [nogo] = (t01 [nogo] + dzot * (g2 + g4)) [nogo] * .5
                t11 [nogo] = ( 4 * t02 [nogo] - t01 [nogo]) / 3
                t20 [nogo] = (16 * t11 [nogo] - t10 [nogo]) / 15
                nogo2 = np.zeros (g3.shape, dtype = bool)
                # test convergence of 5 point romberg result
                tri = np.zeros (g3.shape, dtype = complex)
                tri [nogo] = test (t11 [nogo], t20 [nogo], 0.)
                nogo2 [np.logical_or (tri.real > rx, tri.imag > rx)] = 1
                go = np.logical_not (nogo2)
                sum [go] = sum [go] + t20 [go]
                nt [go] = nt [go] + 1
                # Here the nogo2 part should be handled (goto 13)
                nt = 0
                if ns < np: #hMMM goto 15
                    ns = ns * 2
                    dz = s / ns
                    dzot = dz * .5
                    #g5 [] = g3 [] # fixme
                    #g3 [] = g2 [] # fixme
                    # goto 4 ??
                if lstep != 1:
                    lstep = 1
                    # Hmpf. printint. defer to end
                    # hmpf lambda?
                    #t00, t11 = ...
                    #print t00
                    #print z, dz, self.a, self.b
                    #for g in range (len (g1)):
            z = z + dz
            if z > zend:
                # we might return here
                break
            #g1 [..] = g5 [..] # FIXME
            if nt >= nts and ns > nx:
                ns = ns / 2
                nt = 1
            # implicit continue here (goto 2)
    # end def rom1

    def saoa (self, t, cond = True):
        """ Computes the integrand for each of the 6 Sommerfeld
            integrals for source and observer above ground
            The is_hankel boolean matrix indicates which of the elements
            should be computed with the hankel variant.
        """
        is_hankel = np.logical_and (self.is_hankel, cond)
        is_bessel = np.logical_and (np.logical_not (self.is_hankel), cond)
        dxl = (self.b - self.a)
        xl  = self.a + dxl * t
        cgam1 = np.zeros (self.is_hankel.shape, dtype = complex)
        cgam2 = np.zeros (self.is_hankel.shape, dtype = complex)
        b0    = np.zeros (self.is_hankel.shape, dtype = complex)
        b0p   = np.zeros (self.is_hankel.shape, dtype = complex)
        b, bp, cg1, cg2 = self.saoa_bessel (xl, self.rho [is_bessel])
        cgam1 [is_bessel] = cg1
        cgam2 [is_bessel] = cg2
        b0    [is_bessel] = b
        b0p   [is_bessel] = bp
        b, bp, cg1, cg2 = self.saoa_hankel (xl, self.rho [is_hankel])
        cgam1 [is_hankel] = cg1
        cgam2 [is_hankel] = cg2
        b0    [is_hankel] = b
        b0p   [is_hankel] = bp
        cgam1 = cgam1 [cond]
        cgam2 = cgam2 [cond]
        b0    = b0    [cond]
        b0p   = b0p   [cond]
        xlr  = xl * np.conjugate (xl)
        dgam = None
        if xlr < self.tsmag:
            dgam = cgam2 - cgam1
        elif xl.imag < 0:
            sign = 1
        elif xl.real < 2 * np.pi:
            sign = -1
        elif x.real > self.ck1.real:
            sign = 1
        else:
            dgam = cgam2 - cgam1
        if dgam is None:
            xxl  = 1 / (xl * xl)
            dgam = sign * ((self.ct3 * xxl + self.ct2) * xxl + self.ct1) / xl
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
        ans [0][r0] = ans [3][r0] = -com [r0] * xl * xl * .5
        b0p [nr0] = b0p [nr0] / self.rho [cond][nr0]
        ans [0][nr0] = -com [nr0] * xl * (b0p [nr0] + b0 [nr0] * xl)
        ans [3][nr0] = com [nr0] * xl * b0p [nr0]
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
        if cgam1.real == 0:
            cgam1 = 0 - 1j * np.abs (cgam1.imag)
        if cgam2.real == 0:
            cgam2 = 0 - 1j * np.abs (cgam2.imag)
        return b0, b0p, cgam1, cgam2
    # end def saoa_bessel

    def saoa_hankel (self, xl, rho):
        b0    =  hankel (0, xl * rho)
        b0p   = -hankel (1, xl * rho)
        com   = xl - self.ck1
        cgam1 = np.sqrt (xl + self.ck1) * np.sqrt (com)
        if com.real < 0 and com.imag > 0:
            cgam1 = -cgam1
        com   = xl - 2 * np.pi
        cgam2 = np.sqrt (xl + 2 * np.pi) * np.sqrt (com)
        if com.real < 0 and com.imag > 0:
            cgam2 = -cgam2
        return b0, b0p, cgam1, cgam2
    # end def saoa_hankel

    def test (self, f1, f2, dmin):
        den = np.abs (f2r)
        tr  = np.abs (f2i)
        ti  = np.zeros (tr.shape, dtype = float)
        den [den < tr]   = tr [den < tr]
        den [den < dmin] = dmin
        tr [den >= 1e-37] = np.abs ((f1.real - f2.real) / den)
        ti [den >= 1e-37] = np.abs ((f1.imag - f2.imag) / den)
        tr [den < 1e-37] = 0
        ti [den < 1e-37] = 0
        return tr + 1j * ti
    # end def test

# end class Sommerfeld

if __name__ == '__main__':
    s = Sommerfeld (4.0, .001, 10.0)
