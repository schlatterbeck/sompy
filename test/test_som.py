#!/usr/bin/python3

import pytest
import numpy as np
from sompy import Sommerfeld

class Test_Base:

    def test_saoa_bessel (self):
        vals = \
            [ [ -3.11803722    +1.3262701j
              ,  5.93318653    -3.36045551j
              ,  0.139051959   -0.414679796j
              , -3.11248446    +1.33915186j
              ,  0.00753160566 +0.0176054034j
              ,  2.58885336    -0.97795552j
              ]
            , [ -2.64690518    +0.917380214j
              ,  5.22539663    -2.02546954j
              ,  0.294826061   -0.657950521j
              , -2.63087296    +0.960900962j
              ,  0.0013268512  +0.00372831593j
              ,  2.20666766    -0.518677175j
              ]
            ]
        vals = np.array (vals)
        s = Sommerfeld (4.0, .001, 10.0)
        s.a = 0j
        s.b = 53.2088928-53.2088928j
        cond = np.zeros (s.rho.shape, dtype = bool)
        cond [7] = 1
        result = []
        t = np.ones (s.rho.shape) * .25
        r = s.saoa (t, cond = cond)
        result.append (r [0])
        r = s.saoa (t * 2, cond = cond)
        result.append (r [0])
        result = np.array (result)
        assert result.shape == vals.shape
        assert result == pytest.approx (vals, rel = 1e-5)
    # end def test_saoa_bessel

    def test_saoa_hankel (self):
        vals = \
            [ [ -2.138304      -1.14035702j
              ,  0.121376038   +0.0647298098j
              , -0.153132051   +0.287140638j
              ,  2.12156224    +1.13142872j
              , -0.00265042298 -0.0014134699j
              ,  0.0364189744  +0.029970834j
              ]
            , [ -2.17637038    -1.22720146j
              ,  0.120408714   +0.0678955019j
              , -0.162322715   +0.2878699j
              ,  2.15976238    +1.2178365j
              , -0.00262929988 -0.0014825972j
              ,  0.0358796231  +0.0309694745j
              ]
            ]
        vals = np.array (vals)
        s = Sommerfeld (4.0, .001, 10.0)
        s.a = 2.51327419j
        s.b = 3.76991153 -1.2566371j
        cond = np.zeros (s.rho.shape, dtype = bool)
        cond [0] = cond [1] = 1
        t = np.zeros (s.rho.shape)
        r = s.saoa (t, cond = cond)
        assert r.shape == vals.shape
        assert r == pytest.approx (vals, rel = 1e-5)
    # end def test_saoa_hankel

    def test_rom1_bessel (self):
        vals = \
            [ [ 0j, 0j, 0j, 0j, 0j, 0j ] # not computed for bessel
            , [ -2.31588697    +0.480371684j
              ,  4.71199608    -1.50895941j
              ,  0.324989647   -0.507604897j
              , -2.30867672    +0.528231323j
              , -0.00221469323 +0.012674178j
              ,  1.96858811    -0.311186165j
              ]
            ]
        vals = np.array (vals)
        s = Sommerfeld (4.0, .001, 10.0)
        s.a = 0j
        s.b = 53.2088928-53.2088928j
        r = s.rom1 (6, 2, s.is_bessel)
        cond = np.zeros (s.rho.shape, dtype = bool)
        cond [0] = cond [7] = 1
        r = r [cond]
        assert r.shape == vals.shape
        assert r == pytest.approx (vals, rel = 1e-3)
    # end def test_rom1_bessel

    def test_rom1_hankel (self):
        vals = \
            [ [ -1.44701922    +1.97692227j
              ,  0.126062781   -0.0513910204j
              ,  0.224961758   +0.193945825j
              ,  1.45205915    -1.99802136j
              , -0.00332087232 +0.00183619559j
              ,  0.044633951   -0.00629247818j
              ]
            , [ 0j, 0j, 0j, 0j, 0j, 0j ] # not computed for hankel
            ]
        vals = np.array (vals)
        s = Sommerfeld (4.0, .001, 10.0)
        s.a = 2.51327419j
        s.b = 3.76991153 -1.2566371j
        r = s.rom1 (6, 2, s.is_hankel)
        cond = np.zeros (s.rho.shape, dtype = bool)
        cond [0] = cond [7] = 1
        r = r [cond]
        assert r.shape == vals.shape
        assert r == pytest.approx (vals, rel = 1e-3)
    # end def test_rom1_hankel

    def test_gshank_bessel (self):
        seed = \
            [ [ -2.31588697    +0.480371684j
              ,  4.71199608    -1.50895941j
              ,  0.324989647   -0.507604897j
              , -2.30867672    +0.528231323j
              , -0.00221469323 +0.012674178j
              ,  1.96858811    -0.311186165j
              ]
            ]
        seed = np.array (seed)
        vals = \
            [ [ -2.80080628  -0.413829148j
              ,  5.88886356  +0.292651534j
              ,  1.12654698  -0.116916478j
              , -2.99543357  -0.385962725j
              , -0.002336937 +0.0128434291j
              ,  2.34222507  +0.4738428j
              ]
            ]
        vals = np.array (vals)
        s = Sommerfeld (4.0, .001, 10.0)
        b = 53.2088928 -53.2088928j
        d = 33.4321327 +0j
        cond = np.zeros (s.rho.shape, dtype = bool)
        cond [7] = 1
        r = s.gshank (b, d, 6, seed, 0, b, b, cond)
        assert r.shape == vals.shape
        assert r == pytest.approx (vals, rel = 1e-3)
    # end def test_gshank_bessel

# end class Test_Base
