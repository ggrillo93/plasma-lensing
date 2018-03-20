from __future__ import division
from numba import jit
import numpy as np
import sympy as sym

c = 2.998e10
pctocm = 3.0856e18
GHz = 1e9
re = 2.8179e-13
kpc = 1e3
autocm = 1.4960e13
pi = np.pi

ux, uy = sym.symbols('ux uy')
lensfun = sym.exp(-ux**2 - uy**2)
lensg = np.array([sym.diff(lensfun, ux), sym.diff(lensfun, uy)])
lensh = np.array([sym.diff(lensfun, ux, ux), sym.diff(lensfun, uy, uy), sym.diff(lensfun, ux, uy)])
lensfun = sym.lambdify([ux, uy], lensfun, "numpy")
lensg = sym.lambdify([ux, uy], lensg, "numpy")
lensh = sym.lambdify([ux, uy], lensh, "numpy")

@jit(nopython=True)
def alpha(dso, dsl, f, dm):
    """ Returns alpha coefficient. """
    dlo = dso - dsl
    return -c**2*re*dsl*dlo*dm/(2*pi*f**2*dso)

@jit(nopython=True)
def rFsqr(dso, dsl, f):
    """ Returns the square of the Fresnel scale. """
    dlo = dso - dsl
    return c*dsl*dlo/(2*pi*f*dso)

@jit(nopython=True)
def lensc(dm, f):
    """ Returns the coefficient that determines the phase perturbation due to the lens. """
    return -c*re*dm/f

@jit(nopython=True)
def tg0coeff(dso, dsl):
    dlo = dso - dsl
    return dso/(2*c*dsl*dlo)

@jit(nopython=True)
def tdm0coeff(dm, f):
    return c*re*dm/(2*pi*f**2)

def mapToUp(uvec, alp, ax, ay):
    """ Maps points in the u-plane to points in the u'-plane. """
    ux, uy  = uvec
    fun = lensfun(*uvec)
    upx = ux*(1 - 2*alp*fun/ax**2)
    upy = uy*(1 - 2*alp*fun/ay**2)
    return np.array([upx, upy])
