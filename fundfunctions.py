from __future__ import division
from numba import jit
import numpy as np

c = 2.998e10
pctocm = 3.0856e18
GHz = 1e9
re = 2.8179e-13
kpc = 1e3
autocm = 1.4960e13
pi = np.pi
exp = np.exp

# Gaussian Lenses

@jit(nopython=True)
def gauss(ux, uy):
    return exp(-ux**2 - uy**2)

@jit(nopython=True)
def gauss10(ux, uy):
    return -2*ux*gauss(ux, uy)

@jit(nopython=True)
def gauss01(ux, uy):
    return -2*uy*gauss(ux, uy)

@jit(nopython=True)
def gauss20(ux, uy):
    return 2*gauss(ux, uy)*(2*ux**2 - 1)

@jit(nopython=True)
def gauss02(ux, uy):
    return 2*gauss(ux, uy)*(2*uy**2 - 1)

@jit(nopython=True)
def gauss11(ux, uy):
    return 4*ux*uy*gauss(ux, uy)

@jit(nopython=True)
def gauss30(ux, uy):
    return -4*ux*(2*ux**2 - 3)*gauss(ux, uy)

@jit(nopython=True)
def gauss03(ux, uy):
    return -4*uy*(2*uy**2 - 3)*gauss(ux, uy)

@jit(nopython=True)
def gauss21(ux, uy):
    return -4*uy*(2*ux**2 - 1)*gauss(ux, uy)

@jit(nopython=True)
def gauss12(ux, uy):
    return -4*ux*(2*uy**2 - 1)*gauss(ux, uy)

@jit(nopython=True)
def gauss22(ux, uy):
    return 4*gauss(ux, uy)*(2*ux**2 - 1)*(2*uy**2 - 1)

@jit(nopython=True)
def gauss13(ux, uy):
    return 8*gauss(ux, uy)*ux*uy*(2*uy**2 - 3)

@jit(nopython=True)
def gauss31(ux, uy):
    return 8*gauss(ux, uy)*ux*uy*(2*ux**2 - 3)

@jit(nopython=True)
def gauss40(ux, uy):
    return 4*gauss(ux, uy)*(3 - 12*ux**2 + 4*ux**4)

@jit(nopython=True)
def gauss04(ux, uy):
    return 4*gauss(ux, uy)*(3 - 12*uy**2 + 4*uy**4)

@jit(nopython=True)
def gamma(dso, dsl, f, dm):
    """ Returns gamma coefficient. """
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

def mapToUp(uvec, gam, ax, ay):
    """ Maps points in the u-plane to points in the u'-plane. """
    ux, uy = uvec
    g = gauss(ux, uy)
    upx = ux*(1 - 2*gam*g/ax**2)
    upy = uy*(1 - 2*gam*g/ay**2)
    return np.array([upx, upy])
