#!/usr/bin/python3

import sys
import numpy as np
from scipy.special import jv as bessel, hankel1 as hankel

# Constants
c    = 299.8
mu0  = 4e-7 * np.pi
eps0 = 1e-12 / (c**2 * mu0)

class Sommerfeld:

    verbose = False
    debug   = False

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
        dxa = np.array ([.02, .05, .1])
        dya = np.array ([np.pi / 18, np.pi / 36, np.pi / 18])
        nxa = np.array ([10, 17, 9]) #([11, 17, 9])
        nya = np.array ([10,  5, 8])
        xsa = np.array ([0.02, .2, .2])
        ysa = np.array ([0., 0., np.pi / 9])
        xx    = (xsa, xsa + (nxa - 1) * dxa, nxa)
        yy    = (ysa, ysa + (nya - 1) * dya, nya)
        xgrid = [np.linspace (*x) for x in zip (*xx)]
        ygrid = [np.linspace (*y) for y in zip (*yy)]
        # avoid computing things twice:
        xgrid [0] = xgrid [0][:-1]
        ygrid [2] = ygrid [2][1:]
        xygrid = [np.meshgrid (x, y, indexing = 'ij')
                  for x, y in zip (xgrid, ygrid)]
        r     = np.concatenate ([x [0].flatten () for x in xygrid])
        theta = np.concatenate ([x [1].flatten () for x in xygrid])
        #for a, b in zip (r, theta):
        #    print ('r:', a, 'theta:', b)
        rho   = self.rho = r * np.cos (theta)
        zph   = self.zph = r * np.sin (theta)
        rho [rho < 1e-7] = 1e-8
        zph [zph < 1e-7] = 0.0
        #for a, b in zip (rho, zph):
        #    print ('rho:', a, 'zph:', b)
        self.is_hankel = self.zph < 2 * self.rho
        self.is_bessel = np.logical_not (self.is_hankel)

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

    def evlua (self):
        """ Controls the integration contour in the complex lambda plane for
            evaluation of the Sommerfeld integrals
        """
        dlt = np.maximum (self.zph, self.rho)
        ans = self.evlua_bessel (dlt)
        ans [self.is_hankel] = self.evlua_hankel (dlt) [self.is_hankel]
        ans.T [5] = ans.T [5] * self.ck1
        # Conjugate since NEC uses exp (+jwt)
        erv = np.conjugate (self.ck1sq * ans.T [2])
        ezv = np.conjugate (self.ck1sq * (ans.T [1] + self.ck2sq * ans.T [4]))
        erh = np.conjugate (self.ck2sq * (ans.T [0] + ans.T [5]))
        eph = -np.conjugate (self.ck2sq * (ans.T [3] + ans.T [5]))
        return erv, ezv, erh, eph
    # end def evlua

    def evlua_bessel (self, dlt):
        self.sum = np.zeros (self.rho.shape, dtype = complex)
        self.a   = np.zeros (self.rho.shape, dtype = complex)
        dlt = 1 / dlt
        tkm = (1-1j) * .1 * self.tkmag
        self.b = np.ones (self.rho.shape, dtype = complex) * tkm
        cnd  = (dlt <  self.tkmag) & self.is_bessel
        cnd2 = (dlt >= self.tkmag) & self.is_bessel
        sum = self.rom1 (6, 2, cnd)
        tmp = self.b
        dd  = (1-1j) * dlt
        self.b = np.ones (self.rho.shape, dtype = complex) * dd
        sum [cnd2] = self.rom1 (6, 2, cnd2) [cnd2]
        self.a = tmp
        sum [cnd] = sum [cnd] + self.rom1 (6, 2, cnd) [cnd]
        return self.gshank (self.b, np.pi / 5 * dlt, 6, sum, self.is_bessel)
    # end def evlua_bessel

    def evlua_hankel (self, dlt):
        ones = np.ones (self.rho.shape, dtype = complex)
        cp1 = ones * (0            +.8j * np.pi)
        cp2 = ones * ( 1.2 * np.pi -.4j * np.pi)
        cp3 = ones * (2.04 * np.pi -.4j * np.pi)
        self.a = cp1
        self.b = cp2
        sum = self.rom1 (6, 2, self.is_hankel)
        self.a = self.b
        self.b = cp3
        sum = -(sum + self.rom1 (6, 2, self.is_hankel))
        slope = np.ones (self.rho.shape) * 1000
        cnd   = (self.zph > .001 * self.rho)
        slope [cnd] = self.rho [cnd] / self.zph [cnd]
        dlt = np.pi / 5 / dlt
        delta  = (-1 + 1j * slope) * dlt / np.sqrt (1 + slope * slope)
        delta2 = -np.conjugate (delta)
        bk   = np.zeros (self.rho.shape, dtype = complex)
        ans  = self.gshank (cp1, delta, 6, sum, self.is_hankel)
        rmis = self.rho * (self.ck1.real - 2 * np.pi)
        # This used to be the conditions for goto 8
        cnd8 = (rmis < 4 * np.pi) | (self.rho < 1e-10)
        # This used to be the condition for goto 6
        cnd6 = (self.zph < 1e-10) & np.logical_not (cnd8)
        not68 = np.logical_not (cnd8 | cnd6)
        bk [not68] = ((-self.zph + 1j * self.rho) * (self.ck1 - cp3)) [not68]
        rmis [not68] = -bk [not68].real / np.abs (bk [not68].imag)
        # Another goto 8
        tmp  = np.logical_not (cnd6) & np.logical_not (cnd8)
        cnd8 [tmp] = rmis [tmp] > 4 * self.rho [tmp] / self.zph [tmp]
        cnd8 = cnd8 & self.is_hankel
        # Finally all that wasn't 8 is 6
        # Integrate up between branch cuts, then to + infinity
        cnd6 = np.logical_not (cnd8) & self.is_hankel
        cp1 [cnd6] = self.ck1 - (.1 + .2j)
        cp2 [cnd6] = cp1 [cnd6] + .2
        bk  [cnd6] = 0 + 1j * dlt [cnd6]
        sum [cnd6] = self.gshank (cp1, bk, 6, ans, cnd6) [cnd6]
        self.a = cp1
        self.b = cp2
        ans [cnd6] = self.rom1 (6, 1, cnd6) [cnd6]
        ans [cnd6] = ans [cnd6] - sum [cnd6]
        sum [cnd6] = self.gshank (cp3, bk, 6, ans, cnd6) [cnd6]
        ans [cnd6] = self.gshank (cp2, delta2, 6, sum, cnd6) [cnd6]
        # cnd8
        # Integrate below branch points, the to + infinity
        sum  [cnd8] = -ans [cnd8]
        rmis [cnd8] = self.ck1.real * 1.01
        rmis [cnd8 & (2 * np.pi + 1 > rmis)] = 2 * np.pi + 1
        bk   [cnd8] = rmis [cnd8] + .99j * self.ck1.imag
        delta [cnd8] = bk [cnd8] - cp3 [cnd8]
        delta [cnd8] = delta [cnd8] * dlt [cnd8] / np.abs (delta [cnd8])
        ans [cnd8] = self.gshank (cp3, delta, 6, sum, cnd8, bk, delta2) [cnd8]
        return ans
    # end def evlua_hankel

    def gshank (self, start, dela, nans, seed, cond, bk = None, delb = None):
        """ Integrates the 6 Sommerfeld integrals from start to infinity
            (until convergence) in lambda. At the break point, bk, the
            step increment may be changed from dela to delb. Shank's
            algorithm to accelerate convergence of a slowly converging
            series is used.
            Note that no breakpoint checking is performed if bk is None.
            If delb is None and a breakpoint is encountered, dela is
            used.
        """
        crit = 1e-4
        maxh = 20
        shape = cond.shape + (nans,)
        if seed.shape == shape:
            seed = seed [cond]
        if dela.shape [0] == seed.shape [0]:
            dlt = dela
        else:
            dlt = dela [cond]
        ans1 = np.zeros (shape, dtype = complex)
        ans2 = np.zeros (shape, dtype = complex)
        ans2 [cond] = seed
        if cond.shape == start.shape:
            self.b = np.array (start)
        else:
            self.b = np.zeros (cond.shape, dtype = complex)
            self.b [cond] = start
        if delb is None:
            delb = dela
        # Label here was 2:
        i = 0
        q1 = np.zeros ((maxh,) + seed.shape, dtype = complex)
        q2 = np.zeros ((maxh,) + seed.shape, dtype = complex)
        # label 2
        todo = np.array (cond, dtype = bool)
        done = np.logical_not (todo)
        sum  = np.zeros (cond.shape + (nans,), dtype = complex)
        if bk is None:
            ibx = np.ones (cond.shape, dtype = bool)
        else:
            ibx = np.logical_not (cond)

        for inti in range (maxh):
            inx = inti
            self.a = np.array (self.b)
            self.b [cond] = self.b [cond] + dlt
            if bk is not None:
                hitb1 = (ibx == 0) & (self.b.real > bk.real) & cond
                self.b [hitb1] = bk [hitb1]
                ibx    [hitb1] = True
                sum    [hitb1] = self.rom1 (nans, 2, hitb1) [hitb1]
                ans2   [hitb1] = ans2 [hitb1] + sum [hitb1]
                todo   [hitb1] = False
            sum  [todo] = self.rom1 (nans, 2, todo) [todo]
            ans1 [todo] = ans2 [todo] + sum [todo]
            self.a = np.array (self.b)
            self.b [cond] = self.b [cond] + dlt
            if bk is not None:
                hitb2 = (ibx == 0) & (self.b.real > bk.real) & cond
                self.b [hitb2] = bk [hitb2]
                ibx    [hitb2] = True
                sum    [hitb2] = self.rom1 (nans, 2, hitb2) [hitb2]
                ans2   [hitb2] = ans1 [hitb2] + sum [hitb2]
                todo   [hitb2] = False
            sum  [todo] = self.rom1 (nans, 2, todo) [todo]
            ans2 [todo] = ans1 [todo] + sum [todo]
            if not todo.any ():
                converged = np.array (todo)
                break
            # 11
            as1 = ans1 [cond]
            as2 = ans2 [cond]
            for j in range (1, inti + 1):
                jm = j - 1
                aa = q2 [jm]
                a1 = q1 [jm] + as1 - 2 * aa
                cn = (a1 == 0j)
                a1 [cn] = q1 [jm][cn]
                cn = np.logical_not (cn)
                a2 = np.zeros (aa.shape, dtype = complex)
                a2 [cn] = aa [cn] - q1 [jm][cn]
                a1 [cn] = q1 [jm][cn] - a2 [cn] * a2 [cn] / a1 [cn]
                a2 = aa + as2 - 2 * as1
                cn = (a2 == 0j)
                a2 [cn] = aa [cn]
                cn = np.logical_not (cn)
                a2 [cn] = aa [cn] - (as1 [cn] - aa [cn]) ** 2 / a2 [cn]
                q1 [jm] = as1
                q2 [jm] = as2
                as1 = a1
                as2 = a2
            # This happens before the loop above for inti == 0
            q1 [inti] = as1
            q2 [inti] = as2
            amg = np.abs (ans2.real) + np.abs (ans2.imag)
            den = np.max (amg, axis = 1)

            # goto 17
            as1  = ans1
            as2  = ans2
            jm   = max (inti - 3, 1)
            converged = np.array (todo)
            denm = np.zeros (todo.shape, dtype = complex)
            denm [todo] = 1e-3 * den [todo] * crit
            den  = np.zeros (ans2.shape, dtype = complex)
            denm = np.reshape (np.repeat (denm, 6), den.shape)
            for j in range (jm - 1, inti + 1):
                a1  = q2 [j]
                den [cond] = (abs (a1.real) + abs (a1.imag)) * crit
                den [(den < denm)] = denm [(den < denm)]
                a1  = q1 [j] - a1
                amg = np.zeros (den.shape)
                amg [cond] = abs (a1.real) + abs (a1.imag)
                cnd = converged & np.sum (amg > den, axis = 1, dtype = bool)
                converged [cnd] = False
                if not converged.any ():
                    break
            if converged.any ():
                val = np.zeros (sum.shape, dtype = complex)
                val [cond] = .5 * (q1 [inx] + q2 [inx])
                sum [converged] = val [converged]
        if todo.any () and not converged.any ():
            raise ValueError ('No convergence in gshank')
        # Recursive call where we hit the breakpoint
        if bk is not None and ibx.any ():
            ibx = ibx & cond
            sum [ibx] = self.gshank \
                (bk [ibx], delb, nans, ans2 [ibx], ibx) [ibx]
        return sum
    # end def gshank

    def rom1 (self, n, nx, todo):
        nm    = 1 << 17
        rx    = 1e-4
        z     = np.zeros (self.rho.shape)
        ze    = 1. # end of integration
        ep    = 1 / (1e4 * nm) # for epsilon comparison
        zend  = ze - ep
        sum   = np.zeros (self.rho.shape + (n,), dtype = complex)
        todo  = np.array (todo, dtype = bool)
        ns    = np.ones  (self.rho.shape, dtype = int) * nx
        nt    = np.zeros (self.rho.shape)
        g1    = np.zeros (sum.shape, dtype = complex)
        g1 [todo] = self.saoa (z, todo)
        g2    = np.zeros (sum.shape, dtype = complex)
        g3    = np.zeros (sum.shape, dtype = complex)
        g4    = np.zeros (sum.shape, dtype = complex)
        g5    = np.zeros (sum.shape, dtype = complex)
        if self.debug:
            count = np.zeros (self.rho.shape, dtype = int)
        # This used to be label 2
        while True:
            dz = 1 / ns
            cnd = z + dz > ze
            dz [cnd] = (ze - z [cnd])
            todo [dz <= ep] = False
            if not todo.any ():
                break
            dzot  = dz * .5
            dzotn = dzot [..., np.newaxis]
            g3 [todo] = self.saoa ((z + dzot), todo)
            g5 [todo] = self.saoa ((z + dz), todo)
            # This used to be label 4
            ngo2 = todo
            while ngo2.any ():
                if self.debug:
                    count [ngo2] = count [ngo2] + 1
                nogo = np.zeros (g1.shape, dtype = bool)
                t00  = np.zeros (g1.shape, dtype = complex)
                t01  = np.zeros (g1.shape, dtype = complex)
                t10  = np.zeros (g1.shape, dtype = complex)
                t00 [ngo2] = (g1 [ngo2] + g5 [ngo2]) * dzotn [ngo2]
                t01 [ngo2] = ( t00 [ngo2]
                             + dz [..., np.newaxis][ngo2] * g3 [ngo2]) * .5
                t10 [ngo2] = (4 * t01 [ngo2] - t00 [ngo2]) / 3
                # test convergence of 3 point romberg result
                tri = np.zeros (g1.shape, dtype = complex)
                tri [ngo2] = self.test (t01 [ngo2], t10 [ngo2], 0.)
                nogo [(tri.real > rx) | (tri.imag > rx)] = 1
                ngo = np.sum (nogo, axis = 1, dtype = bool)
                go  = np.logical_not (ngo) & ngo2
                sum [go] = sum [go] + t10 [go]
                nt  [go] = nt  [go] + 2
                # This is only be called for ngo == 1:
                # It won't do anything if the prev produced nogo=0
                g2 [ngo] = self.saoa (z + dz * .25, ngo)
                g4 [ngo] = self.saoa (z + dz * .75, ngo)
                t02 = np.zeros (t01.shape, dtype = complex)
                t11 = np.zeros (t01.shape, dtype = complex)
                t20 = np.zeros (t01.shape, dtype = complex)
                t02 [ngo] = (t01 [ngo] + dzotn [ngo] * (g2 + g4) [ngo]) * .5
                t11 [ngo] = ( 4 * t02 [ngo] - t01 [ngo]) / 3
                t20 [ngo] = (16 * t11 [ngo] - t10 [ngo]) / 15
                nogo2 = np.zeros (g1.shape, dtype = bool)
                # test convergence of 5 point romberg result
                tri = np.zeros (g1.shape, dtype = complex)
                tri [ngo] = self.test (t11 [ngo], t20 [ngo], 0.)
                nogo2 [(tri.real > rx) | (tri.imag > rx)] = 1
                ngo2 = np.sum (nogo2, axis = 1, dtype = bool)
                go   = np.logical_not (ngo2) & ngo
                sum [go] = sum [go] + t20 [go]
                nt  [go] = nt  [go] + 1

                if self.verbose and ngo2.any ():
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
                # here we had a goto 4
            z = z + dz
            if (z > zend).all ():
                break
            g1 [todo] = g5 [todo] # copy!
            cnd = (nt >= 4) & (ns > nx)
            ns [cnd] = ns [cnd] / 2
            nt [cnd] = 1
            # here we had a goto 2
        if self.debug:
            d = {}
            for k in count:
                if k not in d:
                    d [k] = 0
                d [k] += 1
            import pdb; pdb.set_trace ()
        return sum
    # end def rom1

    def saoa (self, t, cond = None):
        """ Computes the integrand for each of the 6 Sommerfeld
            integrals for source and observer above ground
            The is_hankel boolean matrix indicates which of the elements
            should be computed with the hankel variant.
        """
        if cond is None:
            cond = np.ones (self.is_hankel.shape, dtype = bool)
        is_hankel = self.is_hankel & cond
        is_bessel = self.is_bessel & cond
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
        xl    = xl  [cond]
        dxl   = dxl [cond]
        xlr   = xl * np.conjugate (xl)
        dgam  = np.zeros (xlr.shape, dtype = complex)
        cnd   = (xlr < self.tsmag)
        dgam [cnd] = cgam2 [cnd] - cgam1 [cnd]
        sign = np.ones (xlr.shape, dtype = bool)
        sign [cnd] = 0
        sign [(xl.imag >= 0) & (xl.real < 2 * np.pi)] = -1
        cnd = (sign == 1) & (xl.real <= self.ck1.real)
        dgam [cnd] = cgam2 [cnd] - cgam1 [cnd]
        sign [cnd] = 0
        cnd = (sign != 0)
        xxl = (1 / (xl [cnd] * xl [cnd]))
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
