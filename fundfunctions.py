from numba import jit
import numpy as np
import sympy as sym
from scipy.special import airy
from scipy.special import ai_zeros
from scipy.spatial.distance import *
from matplotlib import gridspec as gs
from matplotlib import pyplot as plt
from decimal import Decimal
from scipy.interpolate import *
import pypulse as pp

c = 2.998e10
pctocm = 3.0856e18
GHz = 1e9
re = 2.8179e-13
kpc = 1e3
autocm = 1.4960e13
pi = np.pi

u_x, u_y = sym.symbols('u_x u_y')
# A, B = 1e-2, 5
lensf = sym.exp(-u_x**2 - u_y**2) # sym.sinc(u_x+u_y)**2 # 0.5/((u_x+u_y)**2 + 0.25)*(1./pi) #1./(sym.exp(u_x + u_y) + sym.exp(-u_x-u_y)))**2. #*(1. - A*(sym.sin(B*u_x)+sym.sin(B*u_y)))
lensg = np.array([sym.diff(lensf, u_x), sym.diff(lensf, u_y)])
lensh = np.array([sym.diff(lensf, u_x, u_x), sym.diff(lensf, u_y, u_y), sym.diff(lensf, u_x, u_y)])
lensgh = np.array([sym.diff(lensf, u_x, u_x, u_x), sym.diff(lensf, u_x, u_x, u_y), sym.diff(lensf, u_x, u_y, u_y), sym.diff(lensf, u_y, u_y, u_y)])
lensfun = sym.lambdify([u_x, u_y], lensf, "numpy")
lensg = sym.lambdify([u_x, u_y], lensg, "numpy")
lensh = sym.lambdify([u_x, u_y], lensh, "numpy")
lensgh = sym.lambdify([u_x, u_y], lensgh, "numpy")

airsqrenv = interp1d(ai_zeros(100)[1], airy(ai_zeros(100)[1])[0]**2/2., kind = 'cubic', fill_value = 'extrapolate')

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
    ux, uy = uvec
    grad = lensg(ux, uy)
    upx = ux + alp*grad[0]/ax**2
    upy = uy + alp*grad[1]/ay**2
    return np.array([upx, upy])
    
def assignColor(allroots, nreal):
    nzones = len(allroots)
    colors = list(np.zeros(nzones))
    allcolors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'yellow', 'burlywood', 'chartreuse', 'firebrick', 'darksalmon', 'aqua', 'fuchsia', 'pink', 'steelblue', 'indianred', 'dimgrey']
    center = np.argmax(nreal)
    colors[center] = allcolors[:nreal[center]]
    left = range(center)
    right = range(center + 1, nzones)
    for i in np.flipud(left):
        roots1 = allroots[i][-1].real
        roots0 = allroots[i + 1][0].real
        nroots1 = nreal[i]
        nroots0 = nreal[i + 1]
        regcolor = []
        for j in range(nroots1):
            dist = []
            ind = []
            for k in range(nroots0):
                dist.append(pdist([roots1[j], roots0[k]])[0])
                ind.append([j, k])
            closest = ind[np.argmin(dist)]
            regcolor.append(colors[i+1][closest[1]])
        colors[i] = regcolor
    for i in right:
        roots1 = allroots[i][0].real
        roots0 = allroots[i-1][-1].real
        nroots1 = nreal[i]
        nroots0 = nreal[i-1]
        regcolor = []
        for j in range(nroots1):
            dist = []
            ind = []
            for k in range(nroots0):
                dist.append(pdist([roots1[j], roots0[k]])[0])
                ind.append([j, k])
            closest = ind[np.argmin(dist)]
            regcolor.append(colors[i-1][closest[1]])
        colors[i] = regcolor
    return colors
