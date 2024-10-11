#!/usr/bin/python3

import pytest
import numpy as np
from sompy import Sommerfeld

class Test_Base:

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
        s.is_hankel = np.ones (s.rho.shape, dtype = bool)
        cond = np.zeros (s.rho.shape, dtype = bool)
        cond [1] = cond [11] = 1
        r = s.saoa (0, cond = cond)
        assert r.shape == vals.shape
        assert r == pytest.approx (vals, rel = 1e-5)
    # end def test_saoa_hankel

# end class Test_Base
