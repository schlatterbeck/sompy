#!/usr/bin/python3

import sys
import struct
import numpy as np
from operator import mul
from scipy.special import jv as bessel, hankel1 as hankel
from scipy.interpolate import RegularGridInterpolator
from rsclib.iter_recipes import batched
from argparse import ArgumentParser

# Constants
c    = 299.8
mu0  = 4e-7 * np.pi
mu0  = 1.25663706127e-6
eps0 = 1e-12 / (c**2 * mu0)
#eps0 = 8.8541878188e-12
eta  = np.sqrt (mu0 / eps0)

class Sommerfeld:

    verbose = False
    debug   = False
    # Fundamental constants
    nfunc   = 6 # The 6 Sommerfeld integrals (unlikely to change)

    def __init__ (self, eps_r, sigma = None, fmhz = None):
        """ The original implementation used the case where sigma < 0 to
            specify the imag component of eps_r directly (and ignore fmhz)
            We make sigma optional and allow eps_r to be complex.
            Note: The original implementation in somnec uses 59.96 as an
            approximation of mu0 * c / (2*pi)
            In fortran there is a common segment /gnd/ which contains
            the variables frati and zrati:
             - frati: (k_1**2 - k_2**2) / (k_1**2 + k_2**2) where
               k2 = ω * sqrt(μ_0 ε_0)
               k1 = k2 / zrati
             - zrati: [ε_r - j σ/ωε_0] ** -(1/2)
               where σ is ground conductivity (mhos/meter),
               ε_0 is the permittivity of free space (farads/meter),
               and ω = 2π f
             The code, however computes directly (which is equivalent):
             FRATI=( EPSC-1.)/( EPSC+1.)
             where EPSC is our epscf.  We compute these as member
             variables, too (using the simplified formula from the code).
        """
        self.eps_r = eps_r
        self.sigma = sigma
        self.fmhz  = fmhz
        # epscf is the *relative* epsilon (divided by eps0)
        if sigma is not None:
            assert fmhz is not None
            f = fmhz * 1e6
            self.epscf = eps_r -1j * sigma / (2 * np.pi * f * eps0)
        else:
            self.epscf = eps_r
        self.zrati = 1 / np.sqrt (self.epscf)
        self.frati = (self.epscf - 1) / (self.epscf + 1)
        # In the fortran implementation the common block cgrid contains
        # the three arrays ar1, ar2, and ar2 with the values. We compute
        # the xyz input values here replacing DXA, DYA (the increment
        # value), XSA, YSA (the start values), NXA, NYA (the number of
        # increments aka the dimension)
        # The X values are R (R = sqrt (rho ** 2 + (z + h) ** 2))
        # The Y values are THETA (THETA = atan ((z + h) / rho)
        dxa = self.dxa = np.array ([.02, .05, .1])
        dya = self.dya = np.array ([np.pi / 18, np.pi / 36, np.pi / 18])
        nxa = self.nxa = np.array ([11, 17, 9], dtype = int)
        nya = self.nya = np.array ([10,  5, 8], dtype = int)
        xsa = self.xsa = np.array ([0., .2, .2])
        ysa = self.ysa = np.array ([0., 0., np.pi / 9])
        xx    = (xsa, xsa + (nxa - 1) * dxa, nxa)
        yy    = (ysa, ysa + (nya - 1) * dya, nya)
        xgrid = [np.linspace (*x) for x in zip (*xx)]
        ygrid = [np.linspace (*y) for y in zip (*yy)]
        # A copy:
        self.xgrid = xgrid [:]
        self.ygrid = ygrid [:]
        self.grids = [np.meshgrid (x, y, indexing = 'ij')
                      for x, y in zip (xgrid, ygrid)]
        # avoid computing things twice:
        xgrid [0] = np.array (xgrid [0][1:-1])
        ygrid [2] = np.array (ygrid [2][1:])
        xygrid = [np.meshgrid (x, y, indexing = 'ij')
                  for x, y in zip (xgrid, ygrid)]
        r     = self.r     = np.concatenate ([x [0].flatten () for x in xygrid])
        theta = self.theta = np.concatenate ([x [1].flatten () for x in xygrid])
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

    @classmethod
    def from_file (cls, filename = 'somnec.out', byte_order = '='):
        """ Read somnec grid from file
        """
        with open (filename, 'rb') as f:
            content = f.read ()
        # single precision version:
        cl = len (content)
        if cl == 8632:
            fmtc = 'f'
            flen = 4
        elif cl == 17232:
            fmtc = 'd'
            flen = 8
        else:
            raise ValueError ('Unknown file format (invalid length: %s)' % l)
        fmti   = 'L'
        som    = cls (1.0)
        som.ar = [[], [], []]
        # Unpack byte count (4-byte unsigned int)
        rl = struct.unpack (byte_order + fmti, content [:4]) [0]
        assert cl - rl == 8
        off = 4
        for n, g in enumerate (som.grids):
            l   = 2 * mul (*g [0].shape)
            bl  = l * flen
            fmt = byte_order + str (l) + fmtc
            for e in range (4):
                # ar arrays are complex
                flt = list (struct.unpack (fmt, content [off:off+bl]))
                itr = iter (flt)
                cpx = [a + 1j * b for a, b in zip (itr, itr)]
                som.ar [n].append (np.reshape (cpx, g [0].T.shape).T)
                off += bl
        fmt_c = byte_order + '2' + fmtc
        rl, im = struct.unpack (fmt_c, content [off:off+flen*2])
        som.epscf = som.eps_r = rl + 1j * im
        off += flen * 2
        fmt_3 = byte_order + '3' + fmtc
        fl3 = 3 * flen
        for x in 'dxa', 'dya', 'xsa', 'ysa':
            cnt = content [off:off+fl3]
            setattr (som, x, np.array (list (struct.unpack (fmt_3, cnt))))
            off += fl3
        fmt_i = byte_order + '3' + fmti
        som.nxa = list (struct.unpack (fmt_i, content [off:off+12]))
        som.nxa = np.array (som.nxa, dtype = int)
        off += 12
        som.nya = list (struct.unpack (fmt_i, content [off:off+12]))
        som.nya = np.array (som.nya, dtype = int)
        off += 12

        # At the end we have the length again
        rl = struct.unpack (byte_order + fmti, content [off:off+4]) [0]
        assert cl - rl == 8
        return som
    # end def from_file

    def _fmt_complex (self, c):
        if c.imag:
            v = '%.5E% .5E' % (c.real, c.imag)
        else:
            v = '%.5E' % c.real
        return v
    # end def _fmt_complex

    def as_text (self):
        r = []
        if isinstance (self.eps_r, complex):
            v = '%.6f % .6fj' % (self.eps_r.real, self.eps_r.imag)
        else:
            v = '%.6f' % self.eps_r
        r.append ('Relative Dielectric Constant = %s' % v)
        if self.sigma is not None:
            r.append ('Conductivity (Mhos/Meter) = %13g' % self.sigma)
        if self.fmhz is not None:
            r.append ('Frequency, MHz = %s' % self.fmhz)
        r.append (' NEC GROUND INTERPOLATION GRID')
        r.append \
            ( ' DIELECTRIC CONSTANT= %.5E% .5E'
            % (self.epscf.real, self.epscf.imag)
            )
        r.extend (('', '', ''))
        names = ('ERV', 'EZV', 'ERH', 'EPH')
        for n, (g, ar) in enumerate (zip (self.grids, self.ar)):
            r.append (' GRID %d' % (n + 1))
            r.append \
                ( '    R(1)= %.4f   DR=  %.4f   NR=  %d'
                % (self.xsa [n], self.dxa [n], self.nxa [n])
                )
            r.append \
                ( ' THET(1)= %.4f   DTH= %.4f   NTH= %d'
                % (self.ysa [n], self.dya [n], self.nya [n])
                )
            for k, sect in enumerate (ar):
                r.extend (('', '', ''))
                r.append (' ' + names [k])
                for s, line in enumerate (sect):
                    r.append (' IR= %d' % (s + 1))
                    for b in batched (iter (line), 5):
                        r.append \
                            (' ' + ' '.join (self._fmt_complex (v) for v in b))
        return '\n'.join (r)
    # end def as_text

    def compute (self):
        erv, ezv, erh, eph = self.evlua ()
        rk  = 2 * np.pi * self.r
        # FIXME: magic constant, eta / (8*pi**2)?
        con = - (0 +4.77147j) * self.r / (np.cos (rk) -1j * np.sin (rk))
        erv = erv * con
        ezv = ezv * con
        erh = erh * con
        eph = eph * con
        self.ar = [[], [], []]
        self.ar [0].append (np.reshape (erv [:90],        ( 9, 10)))
        self.ar [1].append (np.reshape (erv [90:90+5*17], (17,  5)))
        self.ar [2].append (np.reshape (erv [90+5*17:],   ( 9,  7)))
        self.ar [0].append (np.reshape (ezv [:90],        ( 9, 10)))
        self.ar [1].append (np.reshape (ezv [90:90+5*17], (17,  5)))
        self.ar [2].append (np.reshape (ezv [90+5*17:],   ( 9,  7)))
        self.ar [0].append (np.reshape (erh [:90],        ( 9, 10)))
        self.ar [1].append (np.reshape (erh [90:90+5*17], (17,  5)))
        self.ar [2].append (np.reshape (erh [90+5*17:],   ( 9,  7)))
        self.ar [0].append (np.reshape (eph [:90],        ( 9, 10)))
        self.ar [1].append (np.reshape (eph [90:90+5*17], (17,  5)))
        self.ar [2].append (np.reshape (eph [90+5*17:],   ( 9,  7)))

        # rows not computed for grid1
        a10 = list (self.ar [1][0][0][[0,2,4]]) + list (self.ar [2][0][0])
        a11 = list (self.ar [1][1][0][[0,2,4]]) + list (self.ar [2][1][0])
        a12 = list (self.ar [1][2][0][[0,2,4]]) + list (self.ar [2][2][0])
        a13 = list (self.ar [1][3][0][[0,2,4]]) + list (self.ar [2][3][0])

        # cols not computed for grid2
        a30 = list (self.ar [1][0][[0,2,4,6,8,10,12,14,16]].T [4])
        a31 = list (self.ar [1][1][[0,2,4,6,8,10,12,14,16]].T [4])
        a32 = list (self.ar [1][2][[0,2,4,6,8,10,12,14,16]].T [4])
        a33 = list (self.ar [1][3][[0,2,4,6,8,10,12,14,16]].T [4])

        # Fill grid1 for r equal to zero
        cl2  = -(0 +188.370j) * (self.epscf - 1) / (self.epscf + 1)
        cl1  = cl2 / (self.epscf + 1)
        ysa, nya, dya = self.ysa [0], self.nya [0], self.dya [0]
        thet = np.linspace (ysa, ysa + (nya - 1) * dya, nya)
        ezv  = np.ones  (thet.shape, dtype = complex) * self.epscf * cl1
        erv  = np.zeros (thet.shape, dtype = complex)
        erh  = np.zeros (thet.shape, dtype = complex)
        eph  = np.zeros (thet.shape, dtype = complex)
        cnd  = np.ones  (thet.shape, dtype = bool)
        cnd [-1] = 0
        tfac2 = np.cos (thet [cnd])
        tfac1 = (1 - np.sin (thet [cnd])) / tfac2
        tfac2 = tfac1 / tfac2
        erv [cnd] = self.epscf * cl1 * tfac1
        erh [cnd] = cl1 * (tfac2 - 1) + cl2
        eph [cnd] = cl1 * tfac2 - cl2
        erv [-1] = 0j
        erh [-1] = cl2 - .5 * cl1
        eph [-1] = -erh [-1]
        # Insert missing rows into ar [0]
        self.ar [0][0] = np.vstack ((erv, self.ar [0][0], a10))
        self.ar [0][1] = np.vstack ((ezv, self.ar [0][1], a11))
        self.ar [0][2] = np.vstack ((erh, self.ar [0][2], a12))
        self.ar [0][3] = np.vstack ((eph, self.ar [0][3], a13))
        # Insert missing col into ar [2]
        self.ar [2][0] = np.vstack ((a30, self.ar [2][0].T)).T
        self.ar [2][1] = np.vstack ((a31, self.ar [2][1].T)).T
        self.ar [2][2] = np.vstack ((a32, self.ar [2][2].T)).T
        self.ar [2][3] = np.vstack ((a33, self.ar [2][3].T)).T
        self.interpolators = []
        for k in range (len (self.ar)):
            self.ar [k] = np.array (self.ar [k])
            self.interpolators.append ([])
            tpl = (self.xgrid [k], self.ygrid [k])
            for i in range (4):
                gi = RegularGridInterpolator \
                    (tpl, self.ar [k][i], method = 'cubic')
                self.interpolators [k].append (gi)
        self.interpolators = np.array (self.interpolators)
    # end def compute

    def compute_incom (self, dirvec):
        """ Compute values that used to be in fortran common segment /incom/
            This returns sn, xsn, ysn
        """
        sn  = np.linalg.norm (dirvec [..., :2], axis = 1)
        sn [sn < 1e-5] = 0
        xsn = np.ones  (sn.shape)
        ysn = np.zeros (sn.shape)
        cnd = sn != 0
        xsn [cnd] = dirvec [..., 0][cnd] / sn [cnd]
        ysn [cnd] = dirvec [..., 1][cnd] / sn [cnd]
        return sn, xsn, ysn
    # end def compute_incom

    def direct_field (self, p, dirvec, obs):
        """ E-field at a point -- not integrated over a segment
            p is the source point, obs is the observation point
            dirvec is the direction vector of the infinitesimal source
            This incorporates the logic in the thin wire kernel in EKSC.
        """
        dist  = obs [None, ...] - p [:, None, ...]
        zp    = np.zeros (dist.shape [:-1])
        for k in range (dist.shape [0]):
            zp [k] = dist [k].dot (dirvec [k])
        rho   = dist - dirvec [:, None, ...] * zp [..., None]
        rh    = np.linalg.norm (rho, axis = -1)
        cnd   = rh > 1e-10
        rho [cnd] = rho [cnd] / rh [cnd, None]
        cnd   = np.logical_not (cnd)
        rho  [cnd] = 0
        r     = np.sqrt (zp * zp + rh * rh)
        # Here we would use the lumped current approximation for r > 1
        # see below
        tezk = self.eksc (zp, rh)
        terk = np.zeros (tezk.shape)
        # This is the lumped current element approximation for large r
        # We instead *always* use the exact version because we do not
        # need to integrate.
        if 0:
            rmag  = 2 * np.pi * r
            cth   = zp / r
            px    = rh / r
            txk   = np.exp (-1j * rmag)
            py    = 2 * np.pi * r * r
            tyk   = eta * cth * txk * (-1j / rmag + 1) / py
            tzk   = eta * px  * txk * (1j * rmag - 1j / rmag + 1) / 2 * py
            tezk  = tyk * cth - tzk * px
            terk  = tyk * px + tzk * cth
        tk    = tezk [..., None] * dirvec [:, None, :] \
              + terk [..., None] * rho
        # Finally compute projection of field onto observation segment
        # See line ww77 in function CMWW, original comment:
        # Electric field tangent to segment I is computed
        # etk = exk * cabi + eyk * sabi + ezk * salpi
        return tk
    # end def direct_field

    def efld (self, p, dirvec, obs):
        """ This computes the sum of three components, the direct field,
            the reflected field, and the Sommerfeld contribution. The
            vector dist from p to obs is (xij, yij, zij) in fortran.
        """
        field = None
        for refl in (1, -1):
            kvec = np.array ([1, 1, refl])
            fld  = self.direct_field (p * kvec, dirvec, obs)
            if refl > 0:
                field = fld
            else:
                field = field - fld * self.frati
        field = field + self.sflds (p, dirvec, obs)
        return field
    # end def efld

    def eksc (self, z, rh):
        """ This is more or less the original in the fortran code but
            without taking the segment length into account. We also do
            not compute a contribution of segment ends -- our
            infinitesimal dipole doesn't have a radius.
            So the contribution of the field in rho-direction is zero,
            we return only the ezk component.
            We also do not integrate over the segment length (performed
            by INTX in fortran).
            z: z coordinate of field point
            rh: rho coordinate of field point
            # not passed as arguments:
            xk: 2 pi / lambda where lambda = 1
            ij: indicates if field point is on source segment, not
                needed here
        """
        con  = 1j * eta / (8 * np.pi ** 2)
        xk   = 2 * np.pi
        zpk  = xk * z
        rhk  = xk * rh
        rkb2 = rhk * rhk
        # this is done by routine GX for z1 and z2 originally
        # where z1 and z2 are the segment ends, then gz below is
        # integrated from z1 to z2.
        r2   = z * z + rh * rh
        r    = np.sqrt (r2)
        rkz  = xk * r
        gz   = np.exp (-1j * rkz) / r
        ezk  = con * xk * xk * gz
        return ezk
    # end def eksc

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
        self.a   = np.zeros (self.rho.shape, dtype = complex)
        dlt = 1 / dlt
        tkm = (1-1j) * .1 * self.tkmag
        self.b = np.ones (self.rho.shape, dtype = complex) * tkm
        cnd  = (dlt <  self.tkmag) & self.is_bessel
        cnd2 = (dlt >= self.tkmag) & self.is_bessel
        all = np.zeros (dlt.shape + (self.nfunc,), dtype = complex)
        if cnd2.any ():
            all [cnd2] = self.rom1 (2, cnd2) [cnd2]
        tmp = self.b
        dd  = (1-1j) * dlt
        self.b = np.ones (self.rho.shape, dtype = complex) * dd
        all [cnd] = self.rom1 (2, cnd) [cnd]
        self.a = tmp
        if cnd2.any ():
            all [cnd2] = all [cnd2] + self.rom1 (2, cnd2) [cnd2]
        return self.gshank (self.b, np.pi / 5 * dlt, all, self.is_bessel)
    # end def evlua_bessel

    def evlua_hankel (self, dlt):
        ones = np.ones (self.rho.shape, dtype = complex)
        cp1 = ones * (0            +.8j * np.pi)
        cp2 = ones * ( 1.2 * np.pi -.4j * np.pi)
        cp3 = ones * (2.04 * np.pi -.4j * np.pi)
        self.a = cp1
        self.b = cp2
        all = self.rom1 (2, self.is_hankel)
        self.a = self.b
        self.b = cp3
        all = -(all + self.rom1 (2, self.is_hankel))
        slope = np.ones (self.rho.shape) * 1000
        cnd   = (self.zph > .001 * self.rho)
        slope [cnd] = self.rho [cnd] / self.zph [cnd]
        dlt = np.pi / 5 / dlt
        delta  = (-1 + 1j * slope) * dlt / np.sqrt (1 + slope * slope)
        delta2 = -np.conjugate (delta)
        bk   = np.zeros (self.rho.shape, dtype = complex)
        ans  = self.gshank (cp1, delta, all, self.is_hankel)
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
        if cnd6.any ():
            cp1 [cnd6] = self.ck1 - (.1 + .2j)
            cp2 [cnd6] = cp1 [cnd6] + .2
            bk  [cnd6] = 0 + 1j * dlt [cnd6]
            all [cnd6] = self.gshank (cp1, bk, ans, cnd6) [cnd6]
            self.a = cp1
            self.b = cp2
            ans [cnd6] = self.rom1 (1, cnd6) [cnd6]
            ans [cnd6] = ans [cnd6] - all [cnd6]
            all [cnd6] = self.gshank (cp3, bk, ans, cnd6) [cnd6]
            ans [cnd6] = self.gshank (cp2, delta2, all, cnd6) [cnd6]
        # cnd8
        # Integrate below branch points, the to + infinity
        if cnd8.any ():
            all  [cnd8] = -ans [cnd8]
            rmis [cnd8] = self.ck1.real * 1.01
            rmis [cnd8 & (2 * np.pi + 1 > rmis)] = 2 * np.pi + 1
            bk   [cnd8] = rmis [cnd8] + .99j * self.ck1.imag
            delta [cnd8] = bk [cnd8] - cp3 [cnd8]
            delta [cnd8] = delta [cnd8] * dlt [cnd8] / np.abs (delta [cnd8])
            ans [cnd8] = self.gshank \
                (cp3, delta, all, cnd8, bk, delta2) [cnd8]
        return ans
    # end def evlua_hankel

    def fbar (self, p):
        """ Sommerfeld attenuation function for Norton's asymptotic
            field approximation for numerical distance p
            accs: relative convergence test value (1e-12)
            sp: sqrt(pi)
            tosp 2 / sqrt (pi)
            fj: 1j
        """
        z    = np.sqrt (p + 0j) * 1j
        fbar = np.zeros (z.shape, dtype = complex)
        cond = np.abs (z) > 3
        fbar [cond] = self.fbar_asymptotic (z [cond])
        cond = np.logical_not (cond)
        fbar [cond] = self.fbar_series (z [cond])
        return fbar
    # end def fbar

    def fbar_asymptotic (self, z):
        minus = z.real < 0
        z [minus] = -z [minus]
        zs    = .5 / (z * z)
        res   = np.zeros (z.shape, dtype = complex)
        term  = np.ones  (z.shape, dtype = complex)
        sp    = np.sqrt (np.pi)
        for i in range (6):
            term = -term * (2 * (i + 1) - 1) * zs
            res  = res + term
            res [minus] = \
                ( res [minus]
                - 2 * sp * z [minus] * np.exp (z [minus] * z [minus])
                )
        return -res
    # end def fbar_asymptotic

    def fbar_series (self, z):
        zs   = z * z
        res  = z
        pow  = z
        term = np.zeros (z.shape, dtype = complex)
        tms  = np.zeros (z.shape, dtype = complex)
        sms  = np.zeros (z.shape, dtype = complex)
        cnd  = np.ones  (z.shape, dtype = bool)
        for i in range (100):
            pow  [cnd] = -pow [cnd] * zs [cnd] / (i + 1)
            term [cnd] = pow [cnd] / (2 * (i + 1) + 1)
            res  [cnd] = res [cnd] + term [cnd]
            tms  [cnd] = (term [cnd] * np.conjugate (term [cnd])).real
            sms  [cnd] = (res  [cnd] * np.conjugate (res  [cnd])).real
            cnd [tms / sms < 1e-12] = False
            if not cnd.any ():
                break
        sp = np.sqrt (np.pi)
        return 1 - (1 - res * (2 / sp)) * z * np.exp (zs) * sp
    # end def fbar_series

    def gshank (self, start, dela, seed, cond, bk = None, delb = None):
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
        shape = cond.shape + (self.nfunc,)
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
        all  = np.zeros (cond.shape + (self.nfunc,), dtype = complex)
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
                all    [hitb1] = self.rom1 (2, hitb1) [hitb1]
                ans2   [hitb1] = ans2 [hitb1] + all [hitb1]
                todo   [hitb1] = False
            all  [todo] = self.rom1 (2, todo) [todo]
            ans1 [todo] = ans2 [todo] + all [todo]
            self.a = np.array (self.b)
            self.b [cond] = self.b [cond] + dlt
            if bk is not None:
                hitb2 = (ibx == 0) & (self.b.real > bk.real) & cond
                self.b [hitb2] = bk [hitb2]
                ibx    [hitb2] = True
                all    [hitb2] = self.rom1 (2, hitb2) [hitb2]
                ans2   [hitb2] = ans1 [hitb2] + all [hitb2]
                todo   [hitb2] = False
            all  [todo] = self.rom1 (2, todo) [todo]
            ans2 [todo] = ans1 [todo] + all [todo]
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
                val = np.zeros (all.shape, dtype = complex)
                val [cond] = .5 * (q1 [inx] + q2 [inx])
                all [converged] = val [converged]
        if todo.any () and not converged.any ():
            raise ValueError ('No convergence in gshank')
        # Recursive call where we hit the breakpoint
        if bk is not None and ibx.any ():
            ibx = ibx & cond
            all [ibx] = self.gshank (bk [ibx], delb, ans2 [ibx], ibx) [ibx]
        return all
    # end def gshank

    def gwave (self, xx1, xx2, r1, r2, zmh, zph):
        """ Compute the electric field, including ground wave, of a
            current element over a ground plane at intermediate
            distances, including the surface wave field.
            Using formulas of K. A. Norton
            (Proc. IRE, Sept., 1937, pp. 1203-1236)
            fortran commons:
            u, u2, xx1, xx2, r1, r2, zmh, zph are common /gwav/
            - common /gwav/:
              - u: (ε_r - j σ/ωε_0) ** -(1/2) [zrati]
              - u2: u**2
              - xx1, xx2: defined in gfld and sflds
              - r1: distance from current element to point at which field
                is evaluated
              - r2: distance from image of current element to point at
                which field is evaluated
              - zmh: Z - Z'
              - zph: Z + Z' where Z is height of the field evaluation
                point and Z' is the height of the current element
            fj = 1j, fjx = [fj.real, fj.imag]
            tpj = 2j * np.pi, tpjx = [tpj.real, tpj.imag]
        """
        econ  = 0 -188.367j
        u2    = self.zrati * self.zrati
        sppp  = zmh / r1
        sppp2 = sppp * sppp
        cppp2 = 1 - sppp2
        cppp2 [cppp2 < 1e-20] = 1e-20
        cppp  = np.sqrt (cppp2)
        spp   = zph / r2
        spp2  = spp * spp
        cpp2  = 1 - spp2
        cpp2 [cpp2 < 1e-20] = 1e-20
        cpp  = np.sqrt (cpp2)
        rk1  = -2j * np.pi * r1
        rk2  = -2j * np.pi * r2
        t1   = 1 - u2 * cpp2
        t2   = np.sqrt (t1)
        t3   = (1 - 1 / rk1) / rk1
        t4   = (1 - 1 / rk2) / rk2
        p1   = rk2 * u2 * t1 / (2 * cpp2)
        rv   = (spp - self.zrati * t2) / (spp + self.zrati * t2)
        omr  = 1 - rv
        w    = 1 / omr
        w    = (4 + 0j) * p1 * w * w
        f    = self.fbar (w)
        q1   = rk2 * t1 / (2 * u2 * cpp2)
        rh   = (t2 - self.zrati * spp) / (t2 + self.zrati * spp)
        v    = 1 / (1 + rh)
        v    = (4 +0j) * q1 * v * v
        g    = self.fbar (v)
        xr1  = xx1 / r1
        xr2  = xx2 / r2
        x1   = cppp2 * xr1
        x2   = rv * cpp2 * xr2
        x3   = omr * cpp2 * f * xr2
        x4   = self.zrati * t2 * spp * 2 * xr2 / rk2
        x5   = xr1 * t3 * (1 - 3 * sppp2)
        x6   = xr2 * t4 * (1 - 3 * spp2)
        ezv  = (x1 + x2 + x3 - x4 - x5 - x6) * econ
        x1   = sppp * cppp * xr1
        x2   = rv * spp * cpp * xr2
        x3   = cpp * omr * self.zrati * t2 * f * xr2
        x4   = spp * cpp * omr * xr2 / rk2
        x5   = 3 * sppp * cppp * t3 * xr1
        x6   = cpp * self.zrati * t2 * omr * xr2 / rk2 * .5
        x7   = 3 * spp * cpp * t4 * xr2
        erv  = -(x1 + x2 - x3 + x4 - x5 + x6 - x7) * econ
        ezh  = -(x1 - x2 + x3 - x4 - x5 - x6 + x7) * econ
        x1   = sppp2 * xr1
        x2   = rv * spp2 * xr2
        x4   = u2 * t1 * omr * f * xr2
        x5   = t3 * (1 - 3 * cppp2) * xr1
        x6   = t4 * (1 - 3 * cpp2) * (1 - u2 * (1 + rv) - u2 * omr * f) * xr2
        x7   = u2 * cpp2 * omr * (1 - 1 / rk2) \
             * (f * (u2 * t1 - spp2 - 1 / rk2) + 1 / rk2) * xr2
        erh  = (x1 - x2 - x4 - x5 + x6 + x7) * econ
        x1   = xr1
        x2   = rh * xr2
        x3   = (rh + 1) * g * xr2
        x4   = t3 * xr1
        x5   = t4 * (1 - u2 * (1 + rv) - u2 * omr * f) * xr2
        x6   = .5 * u2 * omr \
             * (f * (u2 * t1 - spp2 - 1 / rk2) + 1 / rk2) * xr2 / rk2
        eph  = -(x1 - x2 + x3 - x4 + x5 + x6) * econ
        return np.array ([erv, ezv, erh, ezh, eph])
    # end def gwave

    def intrp (self, xy):
        """ Evaluate the Sommerfeld integral contributions to the field
            of a source over ground by interpolation in precomputed
            tables. For a given x and y the values of I_rho^V, I_z^V,
            I_rho^H, and I_phi^H are found by bivariate cubic
            interpolation.
            The parameter is an np.array of [x, y] values or a single
            [x,y] value.
        """
        # determine grids
        l = xy.shape [0]
        isscalar = False
        if len (xy.shape) == 1:
            l = 1
            isscalar = True
            xy = np.array ([xy])
        idx = np.zeros (l, dtype = int)
        idx [xy [..., 0] <= self.xsa [1]] = 0
        idx [(xy [..., 0] > self.xsa [1]) & (xy [..., 1] <= self.ysa [2])] = 1
        idx [(xy [..., 0] > self.xsa [1]) & (xy [..., 1] >  self.ysa [2])] = 2
        intpol = self.interpolators [idx, :]
        r = np.zeros ((l, 4), dtype = complex)
        if (idx == 0).any ():
            r [idx == 0, 0] = self.interpolators [0, 0](xy [idx == 0])
            r [idx == 0, 1] = self.interpolators [0, 1](xy [idx == 0])
            r [idx == 0, 2] = self.interpolators [0, 2](xy [idx == 0])
            r [idx == 0, 3] = self.interpolators [0, 3](xy [idx == 0])
        if (idx == 1).any ():
            r [idx == 1, 0] = self.interpolators [1, 0](xy [idx == 1])
            r [idx == 1, 1] = self.interpolators [1, 1](xy [idx == 1])
            r [idx == 1, 2] = self.interpolators [1, 2](xy [idx == 1])
            r [idx == 1, 3] = self.interpolators [1, 3](xy [idx == 1])
        if (idx == 2).any ():
            r [idx == 2, 0] = self.interpolators [2, 0](xy [idx == 2])
            r [idx == 2, 1] = self.interpolators [2, 1](xy [idx == 2])
            r [idx == 2, 2] = self.interpolators [2, 2](xy [idx == 2])
            r [idx == 2, 3] = self.interpolators [2, 3](xy [idx == 2])
        if isscalar:
            return r [0]
        return r
    # end def intrp

    def rom1 (self, nx, todo):
        nm    = 1 << 17
        rx    = 1e-4
        z     = np.zeros (self.rho.shape)
        ze    = 1. # end of integration
        ep    = 1 / (1e4 * nm) # for epsilon comparison
        zend  = ze - ep
        all   = np.zeros (self.rho.shape + (self.nfunc,), dtype = complex)
        todo  = np.array (todo, dtype = bool)
        ns    = np.ones  (self.rho.shape, dtype = int) * nx
        nt    = np.zeros (self.rho.shape)
        g1    = np.zeros (all.shape, dtype = complex)
        g1 [todo] = self.saoa (z, todo)
        g2    = np.zeros (all.shape, dtype = complex)
        g3    = np.zeros (all.shape, dtype = complex)
        g4    = np.zeros (all.shape, dtype = complex)
        g5    = np.zeros (all.shape, dtype = complex)
        if self.debug:
            count = np.zeros (self.rho.shape, dtype = int)
            ish   = (todo & self.is_hankel).any ()
            isb   = (todo & self.is_bessel).any ()
            assert not (ish and isb)
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
            t00 = np.zeros (g1.shape, dtype = complex)
            t01 = np.zeros (g1.shape, dtype = complex)
            t10 = np.zeros (g1.shape, dtype = complex)
            t02 = np.zeros (g1.shape, dtype = complex)
            t11 = np.zeros (g1.shape, dtype = complex)
            t20 = np.zeros (g1.shape, dtype = complex)
            while ngo2.any ():
                if self.debug:
                    count [ngo2] = count [ngo2] + 1
                nogo = np.zeros (g1.shape, dtype = bool)
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
                all [go] = all [go] + t10 [go]
                nt  [go] = nt  [go] + 2
                # This is only be called for ngo == 1:
                # It won't do anything if the prev produced nogo=0
                g2 [ngo] = self.saoa (z + dz * .25, ngo)
                g4 [ngo] = self.saoa (z + dz * .75, ngo)
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
                all [go] = all [go] + t20 [go]
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
            n = 'BH' [ish]
            print ('%s:' % n, d, file = sys.stderr)
            big = count >= 100
            if big.any ():
                print ('BIG:', list (np.where (big) [0]), file = sys.stderr)
        return all
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

    def sflds (self, px, dirvec, obs, current = 1.0):
        """ Evaluate the Sommerfeld-integral field components due to an
            infinitesimal current element on a segment.
            Compute the  field due to ground for a current element at
            px with direction dirvec.

            The current source element has a position at px (in fortran
            this used to be xj, yj, zj), a unit direction vector dirvec
            (in fortran this was cabj, sabj, salpj) and an optional
            current (which by default is 1). The position is shifted by
            t into the direction of the direction vector.
            The observation point obs (in fortran this was xo, yo, zo)
            is the point at which the E-field is computed.

            In the fortran version:
            - s, xj, yj, zj, cabj, sabj, salpj are common /dataj/
            - xo, yo, zo, sn, xsn, ysn, isnor are common /incom/
            - outputs xx1, xx2, r1, r2, zmh, zph are common /gwav/
            - frati is from common /gnd/
            The fortran version used to return the field value
            multiplied by the segment length when using the Norton
            approximation. For the Non-Norton case it returned the field
            value and the caller integrated over the segment length.
            We do not multiply by the segment length inside sflds

            Docs on common segments:
            - /dataj/ is used to pass the parameters of the source segment
              or patch to the routines tht compute the E or H field and
              to return the field components.
              - s: segment length
              - xj, yj, zj: coordinates of segment center
              - cabj, sabj, salpj: x, y, z, respectively, of the unit
                vector in the direction of the segment
            - /gnd/ contains parameters of the ground including the
              two-medium ground and radial-wire ground-screen cases.
              - frati: (k_1**2 - k_2**2) / (k_1**2 + k_2**2) where
                k2 = ω * sqrt(μ_0 ε_0)
                k1 = k2 / zrati
              - zrati: [ε_r - j σ/ωε_0] ** -(1/2)
                where σ is ground conductivity (mhos/meter),
                ε_0 is the permittivity of free space (farads/meter),
                and ω = 2π f
            - /gwav/
              - xx1, xx2: defined in gfld and sflds
              - r1: distance from current element to point at which field
                is evaluated
              - r2: distance from image of current element to point at
                which field is evaluated
              - zmh: Z - Z'
              - zph: Z + Z' where Z is height of the field evaluation
                point and Z' is the height of the current element
            - /incom/
              - xo, yo, zo: point at which field due to ground will be
                evaluated
              - sn: cos \alpha (see figure 11)
                alpha is angle from x-y plane upwards
              - xsn: cos \beta
                beta is angle from x-axis to y-axis
              - ysn: sin \beta
              - isnor: 1 to evaluate field due to ground by
                interpolation, 0 to use Norton's approximation
                [wtf: would expect it the other way round]
                This seems to be wrong, too, the fortran code sets isnor
                to 1 or 2 (not zero).
                Note that in the python implementation we just use the
                boolean isnor to determine if we use the Norton
                approximation.
        """
        sn, xsn, ysn = self.compute_incom (dirvec)
        rhx   = obs [None, ..., 0] - px [..., None, 0]
        rhy   = obs [None, ..., 1] - px [..., None, 1]
        rhs   = rhx * rhx + rhy * rhy
        rho   = np.sqrt (rhs)
        cnd   = rho <= 0
        rhx [cnd] = 1.
        rhy [cnd] = 0.
        cnd   = rho > 0
        rhx [cnd] = rhx [cnd] / rho [cnd]
        rhy [cnd] = rhy [cnd] / rho [cnd]
        phx   = -rhy
        phy   = rhx
        cph   = rhx * xsn [..., None] + rhy * ysn [..., None]
        sph   = rhy * xsn [..., None] - rhx * ysn [..., None]
        cnd   = np.abs (cph) < 1e-10
        cph [cnd] = 0
        cnd   = np.abs (sph) < 1e-10
        cph [cnd] = 0
        zph   = obs [None, ..., 2] + px [..., None, 2]
        zphs  = zph * zph
        r2s   = rhs + zphs
        r2    = np.sqrt (r2s)
        isnor = r2 > .95
        rk    = r2 * 2 * np.pi
        xx2   = np.cos (rk) -1j * np.sin (rk)
        e     = np.zeros (isnor.shape + (3,), dtype = complex)
        # Use Norton approximation for field due to ground. Current is
        # lumped at segment center with current moment for constant,
        # sine, or cosine distribution.
        r1    = np.ones  (isnor.shape) [isnor]
        xx1   = np.zeros (isnor.shape) [isnor]
        gwv   = self.gwave (xx1, xx2 [isnor], r1, r2 [isnor], r1, zph [isnor])
        erv, ezv, erh, ezh, eph = gwv
        # FIXME: magic constant
        # p 101 in nec2prt2 tells us 4.771341189 = eta/(8 pi**2)
        # it is really 4.771345158604122
        # But fails too many tests if we change this
        #magic = eta / (8 * np.pi ** 2)
        magic = -4.77134j
        et    = magic * self.frati * xx2 [isnor] / (r2s [isnor] * r2 [isnor])
        er    = 2 * et * (1 + 1j * rk [isnor])
        et    = et * (1 - (rk [isnor] * rk [isnor]) + 1j * rk [isnor])
        hrv   = (er + et) * rho [isnor] * zph [isnor] / r2s [isnor]
        hzv   = (zphs [isnor] * er - rhs [isnor] * et) / r2s [isnor]
        hrh   = (rhs [isnor] * er - zphs [isnor] * et) / r2s [isnor]
        erv   = erv - hrv
        ezv   = ezv - hzv
        erh   = erh + hrh
        ezh   = ezh + hrv
        eph   = eph + et
        ddd   = dirvec [..., 2, None] * isnor
        erv   = erv * ddd [isnor]
        ezv   = ezv * ddd [isnor]
        tmp   = (sn [..., None] * isnor) [isnor]
        erh   = erh * tmp * cph [isnor]
        ezh   = ezh * tmp * cph [isnor]
        eph   = eph * tmp * sph [isnor]
        erh   = erv + erh
        e [..., 0][isnor] = (erh * rhx [isnor] + eph * phx [isnor])
        e [..., 1][isnor] = (erh * rhy [isnor] + eph * phy [isnor])
        e [..., 2][isnor] = (ezv + ezh)
        # Interpolation in Sommerfeld field tables
        nisnor = np.logical_not (isnor)
        thet   = np.ones (zph [nisnor].shape) * np.pi / 2
        cnd    = rho [nisnor] >= 1e-12
        thet [cnd] = np.arctan (zph [nisnor][cnd] / rho [nisnor][cnd])
        # Combine vertical and horizontal components and convert to
        # x,y,z components. Multiply by exp(-jkr)/r.
        erv, ezv, erh, eph = self.intrp (np.array ([r2 [nisnor], thet]).T).T
        xx2  = xx2 [nisnor] / r2 [nisnor]
        sfac = (sn [..., None] * nisnor)[nisnor] * cph [nisnor]
        erh  = xx2 \
             * ((dirvec [..., 2, None] * nisnor)[nisnor] * erv + sfac * erh)
        ezh  = xx2 \
             * ((dirvec [..., 2, None] * nisnor)[nisnor] * ezv - sfac * erv)
        # x,y,z fields for constant current
        eph  = (sn [..., None] * nisnor) [nisnor] * sph [nisnor] * xx2 * eph
        e [..., 0][nisnor] = erh * rhx [nisnor] + eph * phx [nisnor]
        e [..., 1][nisnor] = erh * rhy [nisnor] + eph * phy [nisnor]
        e [..., 2][nisnor] = ezh
        return e
    # end def sflds

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

    def to_file (self, fn, fmtc = 'f', byte_order = '='):
        """ Dump computed (or read) somnec grid to file
            By default we use single precision float (compatible with
            the Fortran version), if double precisions is desired, use
            fmtc = 'd', note that the computation is accurate only to
            about 1e-3, so we do not gain anything with double
            precision.
        """
        result = []
        fmti   = 'L'

        for ar in self.ar:
            for a in ar:
                a   = a.T.flatten ()
                a   = np.array ([a.real, a.imag]).T.flatten ()
                fmt = byte_order + str (len (a)) + fmtc
                result.append (struct.pack (fmt, *a))
        fmt = byte_order + '2' + fmtc
        result.append (struct.pack (fmt, self.epscf.real, self.epscf.imag))
        fmt = byte_order + '3' + fmtc
        for x in (self.dxa, self.dya, self.xsa, self.ysa):
            result.append (struct.pack (fmt, *x))
        fmt = byte_order + '3' + fmti
        for x in (self.nxa, self.nya):
            result.append (struct.pack (fmt, *x))
        result = b''.join (result)
        lp = struct.pack (byte_order + fmti, len (result))
        with open (fn, 'wb') as f:
            f.write (lp + result + lp)
    # end def to_file

# end class Sommerfeld

def main (argv = sys.argv [1:]):
    cmd = ArgumentParser (argv)
    cmd.add_argument \
        ( '-c', '--complex_epsilon'
        , help    = 'Complex relative dielectric constant epsilon_r,'
                    ' already including the imaginary part due to loss;'
                    ' when specified, epsilon and frequency are ignored'
        , type    = complex
        )
    cmd.add_argument \
        ( '-e', '--epsilon'
        , help    = 'Ground relative dielectric constant epsilon_r'
        , default = 4.0
        , type    = float
        )
    cmd.add_argument \
        ( '-f', '--frequency'
        , help    = 'Frequency in MHz'
        , default = 10.0
        , type    = float
        )
    cmd.add_argument \
        ( '--float-format'
        , help    = 'Format for writing float values to binary format'
        , default = 'f'
        , choices = ['f', 'd']
        )
    cmd.add_argument \
        ( '-w', '--write-file'
        , help    = 'Write sommerfeld data to file'
        )
    cmd.add_argument \
        ( '-r', '--read-file'
        , help    = 'Read sommerfeld data from file'
        )
    cmd.add_argument \
        ( '-s', '--sigma'
        , help    = 'Ground conductivity in S/m'
        , default = 0.001
        , type    = float
        )
    args = cmd.parse_args ()
    if args.read_file:
        s = Sommerfeld.from_file (args.read_file)
    elif args.complex_epsilon:
        s = Sommerfeld (complex_epsilon)
    else:
        s = Sommerfeld (args.epsilon, args.sigma, args.frequency)
    if not args.read_file:
        s.compute ()
    if args.write_file:
        s.to_file (args.write_file, fmtc = args.float_format)
    print (s.as_text ())
# end def main

if __name__ == '__main__':
    main (sys.argv [1:])
