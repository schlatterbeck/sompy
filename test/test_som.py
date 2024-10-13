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
        s.a = np.zeros (s.rho.shape, dtype = complex)
        s.b = np.ones  (s.rho.shape, dtype = complex) * (53.2088928-53.2088928j)
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
        s.a = np.ones (s.rho.shape, dtype = complex) * 2.51327419j
        s.b = np.ones (s.rho.shape, dtype = complex) * (3.76991153 -1.2566371j)
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
        s.a = np.zeros (s.rho.shape, dtype = complex)
        s.b = np.ones  (s.rho.shape, dtype = complex) * (53.2088928-53.2088928j)
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
        s.a = np.ones (s.rho.shape, dtype = complex) * 2.51327419j
        s.b = np.ones (s.rho.shape, dtype = complex) * (3.76991153 -1.2566371j)
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
            , [ -2.18897271    +0.478487819j
              ,  4.46816778    -1.46256638j
              ,  0.149093568   -0.235078514j
              , -2.18731856    +0.48920086j
              , -0.00232726359 +0.0125354007j
              ,  1.86808836    -0.306106269j
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
            , [ -2.87119198    -0.397920609j
              ,  5.88670492    +0.286414981j
              ,  0.559056878   -0.0576702692j
              , -2.91910267    -0.391062587j
              , -0.00244203862 +0.0127298869j
              ,  2.34124136    +0.470980108j
              ]
            ]
        vals = np.array (vals)
        s = Sommerfeld (4.0, .001, 10.0)
        b = np.zeros (s.rho.shape, dtype = complex)
        b [7] = 53.2088928 -53.2088928j
        b [8] = 50.7713356 -50.7713356j
        d = np.array ([33.4321327 +0j, 31.9005718 +0j])
        cond = np.zeros (s.rho.shape, dtype = bool)
        cond [7] = cond [8] = 1
        r = s.gshank (b, d, 6, seed, cond) [cond]
        assert r.shape == vals.shape
        assert r == pytest.approx (vals, rel = 1e-3)
    # end def test_gshank_bessel

    def test_gshank_hankel (self):
        seed = \
            [ [  2.00776601   -8.01187611j
              , -0.272398651  +0.0381847322j
              , -0.706325531  -0.50113225j
              , -2.06688285   +8.25211906j
              ,  0.0083973892 -0.00705266185j
              , -0.0908726901 -0.0253810994j
              ]
            , [  2.19408679    -8.17548943j
              , -0.27332291    +0.0343347378j
              , -0.7048949     -0.517745733j
              , -2.25558758    +8.4144516j
              ,  0.00848118961 -0.00692269811j
              , -0.0907073617  -0.0266431328j
              ]
            ]
        seed = np.array (seed)
        vals = \
            [ [  13.0748482   -10.4512615j
              , -2.36912513   +0.433109283j
              , -1.41920185   -4.10852337j
              , -11.2865829   +10.357749j
              ,  0.0147127882 -0.00860253721j
              , -0.933717251  +0.0110758897j
              ]
            , [  13.7067928   -9.71621895j
              , -2.41830635   +0.102204993j
              , -0.954755962  -4.22943687j
              , -11.8771076   +9.93634701j
              ,  0.0149100628 -0.00816507265j
              , -0.934709907  -0.12284977j
              ]
            ]
        vals = np.array (vals)
        s = Sommerfeld (4.0, .001, 10.0)
        st = np.zeros (s.rho.shape, dtype = complex)
        st [0] = 2.51327419j
        st [1] = 2.51327419j
        d = np.array ([-0.0314159133 +31.4159126j, -5.53947687 +31.4159298j])
        cond = np.zeros (s.rho.shape, dtype = bool)
        cond [0] = cond [1] = 1
        r = s.gshank (st, d, 6, seed, cond) [cond]
        assert r.shape == vals.shape
        assert r == pytest.approx (vals, rel = 1e-3)
    # end def test_gshank_hankel

    def test_gshank_hankel_recursive (self):
        seed = \
            [ [ -13.0748482   +10.4512615j
              ,  2.36912513   -0.433109283j
              ,  1.41920185   +4.10852337j
              ,  11.2865829   -10.357749j
              , -0.0147127882 +0.00860253721j
              ,  0.933717251  -0.0110758897j
              ]
            , [ -13.7067928   +9.71621895j
              ,  2.41830635   -0.102204993j
              ,  0.954755962  +4.22943687j
              ,  11.8771076   -9.93634701j
              , -0.0149100628 +0.00816507265j
              ,  0.934709907  +0.12284977j
              ]
            ]
        seed = np.array (seed)
        vals = \
            [ [ 0.103320166     -0.988833129j
              ,  5.96449757     +0.454928935j
              ,  6.3460722      -0.759163558j
              , -6.10379219     -0.0741284937j
              ,  0.000911501935 +0.0154020973j
              ,  2.37479591     +0.548333287j
              ]
            , [ -0.784671187    -0.825073123j
              ,  5.94271421     +0.419626474j
              ,  5.33457327     -0.620028794j
              , -5.16232586     -0.182956994j
              ,  0.000108209628 +0.0149045251j
              ,  2.36532521     +0.532098413j
              ]
            ]
        vals = np.array (vals)
        s = Sommerfeld (4.0, .001, 10.0)
        st = np.zeros (s.rho.shape, dtype = complex)
        st [0] = 6.40884924 -1.2566371j
        st [1] = 6.40884924 -1.2566371j
        d  = np.array ([26.8740673   +16.2709866j, 27.2886486  +16.5219936j])
        d2 = np.array ([0.0314159133 +31.4159126j,  5.53947687 +31.4159298j])
        bk = np.zeros (s.rho.shape, dtype = complex)
        bk [0] = 12.9941263 +2.73043895j
        bk [1] = 12.9941263 +2.73043895j
        cond = np.zeros (s.rho.shape, dtype = bool)
        cond [0] = cond [1] = 1
        r = s.gshank (st, d, 6, seed, cond, bk, d2) [cond]
        assert r.shape == vals.shape
        assert r == pytest.approx (vals, rel = 1e-3)
    # end def test_gshank_hankel_recursive

    def test_evlua (self):
        vals = \
            [ [  1056.00671 -330.47583j
              ,  872.122375 -593.691833j
              ,  1150.5542  -498.038666j
              , -905.507202 +534.14978j
              ]
            , [  886.40332  -280.664612j
              ,  867.573914 -581.21875j
              ,  1112.45508 -495.226624j
              , -939.632324 +520.576355j
              ]
            ]
        vals = np.array (vals)
        s = Sommerfeld (4.0, .001, 10.0)
        erv, ezv, erh, eph = s.evlua ()
        result = []
        result.append ([erv [0], ezv [0], erh [0], eph [0]])
        result.append ([erv [1], ezv [1], erh [1], eph [1]])
        result = np.array (result)
        assert result.shape == vals.shape
        assert result == pytest.approx (vals, rel = 1e-3)
    # end def test_evlua

# end class Test_Base
