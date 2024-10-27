#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from math import prod
from argparse import ArgumentParser
from sompy import Sommerfeld, c

plotly_default = dict \
    ( layout = dict
        ( scene = dict
            ( aspectratio = dict (x=2, y=1, z=1)
            )
        )
    )

def main (argv = sys.argv [1:]):
    cmd = ArgumentParser (argv)
    cmd.add_argument \
        ( '-d', '--direction'
        , help    = 'Source direction, either vertical (default) or'
                    ' horizontal (in Y-direction)'
        , default = 'horizontal'
        , choices = ('vertical', 'horizontal')
        )
    cmd.add_argument \
        ( '-e', '--epsilon'
        , help    = 'Dielectric constant of earth, default=%(default)s,'
                    ' salt water has about 81, very poor ground has 3'
        , default = 14
        , type    = float
        )
    cmd.add_argument \
        ( '-f', '--field-type'
        , help    = 'Field type: all: Combined direct wave, reflected wave,'
                    ' and Sommerfeld contribution, sommerfeld: only'
                    ' Sommerfeld contribution, direct: only direct wave'
        , default = 'all'
        , choices = ('all', 'direct', 'direct+reflected', 'sommerfeld')
        )
    cmd.add_argument \
        ( '-s', '--sigma'
        , help    = 'Earth conductivity (S/m), default=%(default)s,'
                    ' use 5 for salt water, 0.001 for very poor ground'
        , default = 0.002
        , type    = float
        )
    cmd.add_argument \
        ( '--use-matplotlib'
        , help    = 'Use matplotlib instead of plotly for plotting'
        , action  = 'store_true'
        )
    args = cmd.parse_args ()
    f    = c # 1m wavelen (lambda)
    som = Sommerfeld (args.epsilon, args.sigma, f)
    som.compute ()
    h   = 1
    src = np.array ([[0, 0, h]]) # at [0, 0] and height h lambda
    if args.direction == 'horizontal':
        sdi  = np.array ([[0, 1, 0]]) # horizontal in y-direction
        eidx = 1
    else:
        sdi  = np.array ([[0, 0, 1]]) # vertical in z-direction
        eidx = 2
    nx  = 201
    nz  = 101
    x   = np.linspace (-6, 6, nx)
    y   = np.array ([0])
    z   = np.linspace (0.0, 6.1, nz)
    xx, yy, zz = np.meshgrid (x, y, z)
    l   = prod (xx.shape)
    obs = np.reshape (np.array ([xx, yy, zz]), (3, l)).T
    if args.field_type == 'direct':
        e  = som.direct_field (src, sdi, obs)
    elif args.field_type == 'sommerfeld':
        e  = som.sflds (src, sdi, obs)
    elif args.field_type == 'direct+reflected':
        kvec = np.array ([1, 1, -1])
        ed = som.direct_field (src,        sdi, obs)
        er = som.direct_field (src * kvec, sdi, obs)
        e  = ed - (er * som.frati)
    elif args.field_type == 'all':
        e  = som.efld (src, sdi, obs)
    else:
        assert 0
    ef  = np.reshape (e [0, ..., eidx].real, xx [0].shape)
    title = 'Field: %s \u03b5=%s \u03c3=%s' \
          % (args.field_type, args.epsilon, args.sigma)
    if args.use_matplotlib:
        fig = plt.figure ()
        ax  = fig.add_subplot (111, projection = '3d')
        ax.set_title (title)
        ax.plot_surface (xx [0], zz [0], ef)
        plt.show ()
    else:
        fig = go.Figure (data = [go.Surface(x = xx[0], y = zz [0], z = ef)])
        layout = dict (plotly_default)
        layout ['layout'].update (title = dict (text = title))
        fig.update (layout)
        fig.show ()

if __name__ == '__main__':
    main (sys.argv [1:])
