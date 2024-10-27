Computing Wave Pattern with Sommerfeld Integrals in Python
==========================================================

:Author: Ralf Schlatterbeck <rsc@runtux.com>

.. |--| unicode:: U+2013   .. en dash

This package |--| called ``sompy`` is a re-implementation of the
Sommerfeld integral [Som09]_ computation and the approximation by
Norton [Nor37]_ found in the public domain Numerical Electromagnetics
Code (NEC) in version 2 [BP81]_. In particular this computes an
interpolation grid, in the original Fortran implementation this was in a
separate program named SOMNEC. The interpolation code that uses the grid
computed by somnec is found in the nec2 source code. Nowadays the
re-implementation of NEC2, in C called nec2c [Kyr]_ or Tim Moltenos C++
implementation [Mol.a]_ and Python wrapper [Mol.b]_ are more useful than
the original Fortran code.

The current code uses an infinitesimal dipole for generating the field.
The observation points are also thought as infinitesimal, so we can get
away with skipping a lot of integrations along antenna segments. The
interpolation grid used by NEC is re-implemented but for the (cubic)
interpolation a ready-made implementation from ``scipy.interpolate`` is
used. The grid can be exported |--| and read |--| from the original
fortran format (when compiled with the GNU Fortran compiler, the file
format seems to be compiler specific).

The code is mainly useful in antenna simulation. When used in that way
in a Method of Moments code, the field contributions need to be
integrated over the antenna segments.

In the current version the code can be used to display the waves created
by an infinitesimal dipole. In the example the lower edge is the ground,
what is seen as the Y-axis in the plot is in fact the Z-axis (height
above ground). The plot shows the field around the infinitesimal dipole
located at X=0, Z=1, lengths are in wavelength, i.e. the dipole is
located 1 wavelengths above ground. The program which has been used
for the plot is in the file ``display_sommerfeld.py``.

The code currently doesn't have a python installer and is not on Pypi,
this may change if I make use of the code myself or you drop me a note
that you do something with it and need an installer.

The code is licensed under the MIT license, see ``LICENSE`` file.

.. figure:: https://raw.githubusercontent.com/schlatterbeck/sompy/master/plot.png
    :align: center

Bibliography
------------

.. [Som09] A. Sommerfeld. Über die Ausbreitung der Wellen in der
       drahtlosen Telegraphie. Annalen der Physik, 28(4):665–736, 1909.
.. [Nor37] K. A. Norton. The propagation of radio waves over the surface
       of the earth and in the upper atmosphere, part II: The
       propagation from vertical, horizontal, and loop antennas over a
       plane earth of finite conductivity. Proceedings of the Institute
       of Radio Engineers, 25(9):1203–1236, September 1937.
.. [BP81] G. J. Burke and A. J. Poggio. Numerical electromagnetics code
       (NEC) – method of moments, part I-III, January 1981. All three
       parts available as `ADA956129`_
.. [Kyr] Neoklis Kyriazis: `NEC2 in C`_
.. [Mol.a] T. Molteno: `NEC2++ Numerical Electromagnetic Code in C++`_
.. [Mol.b] T. Molteno: `Python NEC2++ Module`_

.. _`ADA956129`: https://apps.dtic.mil/sti/tr/pdf/ADA956129.pdf
.. _`NEC2++ Numerical Electromagnetic Code in C++`:
    https://github.com/tmolteno/necpp
.. _`Python NEC2++ Module`: https://pypi.org/project/PyNEC/
.. _`NEC2 in C`: https://github.com/KJ7LNW/nec2c
