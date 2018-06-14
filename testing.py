from upslice import *
from fslice import *
from kdi import *
from dspectra import *
from toapertopt import *
from dspecpulse import *

dso, dsl, f, dm, ax, ay = 1.*kpc*pctocm, 0.5*kpc*pctocm, 0.8*GHz, -7e-4*pctocm, 0.5*autocm, 1.1*autocm

# fsliceGfull([-4.90282938,  0.04858531], 5., 5., 0.1*GHz, 1.5*GHz, dso, dsl, dm, ax, ay, 0.5, 2.5, N=200, npoints=3500, comp=True)

# Test toapert.py
# path = '/home/gian/Documents/Research/NANOGrav/Lensing/Scripts/Simulation/Templates/J1713+0747.Rcvr_800.GUPPI.9y.x.sum.sm'
# template = pp.Archive(path).getData()
# fslicepert([0.1, 0.1], 0.3*GHz, 3.*GHz, dso, dsl,dm, ax, ay, 5e3, spacing = 5e4, plot = True, noise = 0.1, chw = 1.5e6)
# pulsedynspec(dso, dsl, 0.3*GHz, 3.*GHz, dm, np.array([-0.5, 0.25]), 5., ax, ay, template = None, spacing = 5e4, noise = 0.2, chw = 0.5e6)

# m, n = 0.5, 0.
alp = alpha(dso, dsl, f, dm)
# lc = lensc(dm, f)
# print(lc)
# print(alp*np.array([1./ax**2, 1./ay**2]))
# causPlotter(5., 5., alp, ax, ay)
# dspectra(0.1*GHz, 4.*GHz, 5., 5., dso, dsl, dm, ax, ay, 1., 3.)
# coeff = alp*np.array([1./ax**2, 1./ay**2])
# polishedRoots(lensEq, 5., 5., args=([2., 2.], coeff), plot = True)
# fsliceG([0.15, 0.15], 0.3*GHz, 3.*GHz, dso, dsl, dm, ax, ay, npoints=1000, plot=True)
fsliceGfull([0.1, 0.1], 5., 5., 0.3*GHz, 3.*GHz, dso, dsl, dm, ax, ay, 1., 0., comp=True, spacing = 5e4, chw = 1.5e6)
# planeSliceTOA(3., 3., dso, dsl, f, dm, m, n, ax, ay, 1000)
# upx = -2.
# roots = findRoots(causEqFreq, 5., 5., args = (upx, ax, ay, 0.5, 0.5), plot = True, N = 1000)
# print(roots)
# freqcaustics = []
# dlo = dso - dsl
# coeff = dsl*dlo*re*dm/(2*pi*dso)
# causCurveFreq(5., 5., ax, ay, dso, dsl, dm, 0.5, 0.5, N=200)

# Test cases
def runTests():
    dso, dsl = np.array([1.1, 0.55])*kpc*pctocm
    f = 0.8*GHz
    dm = pctocm*np.array([-1e-6, -2e-6, -5e-6, -1e-5])
    m = np.array([0.1, 0.5, 1., 1.2])
    n = np.array([0., 0.5, 1., 2.5, 3., 3.2, 4.])
    gsize = np.array([2048, 1.5*2048, 2*2048])
    lims = np.array([1.5, 2., 3., 4., 4.5, 8.])
    scales = autocm*np.array([0.02, 0.021, 0.022, 0.025, 0.03, 0.04, 0.06, 0.08])
    planeSliceG(lims[3], lims[3], dso, dsl, f, dm[0], m[2], n[2], scales[0], scales[0], gsizex = gsize[-1], gsizey = gsize[-1])
    planeSliceG(lims[2], lims[2], dso, dsl, f, dm[1], m[2], n[2], scales[0], scales[0])
    planeSliceG(lims[-1], lims[-1], dso, dsl, f, dm[-2], m[2], n[4], scales[0], scales[0], gsizex = gsize[1], gsizey = gsize[1])
    planeSliceG(lims[-2], lims[-2], dso, dsl, f, dm[-1], m[2], n[-3], scales[-3], scales[-3], gsizex = gsize[-1], gsizey = gsize[-1])
    planeSliceG(lims[0], lims[0], dso, dsl, f, dm[0], m[1], n[1], scales[0], scales[1])
    planeSliceG(lims[0], lims[0], dso, dsl, f, dm[0], m[1], n[1], scales[0], scales[2])
    planeSliceG(lims[0], lims[0], dso, dsl, f, dm[0], m[1], n[0], scales[0], scales[3])
    planeSliceG(lims[2], lims[1], dso, dsl, f, dm[1], m[1], n[0], scales[0], scales[3])
    planeSliceG(lims[3], lims[2], dso, dsl, f, dm[2], m[1], n[0], scales[4], scales[5], gsizex = gsize[1], gsizey = gsize[1])
    planeSliceG(lims[-2], lims[-2], dso, dsl, f, -dm[1], m[1], n[0], scales[0], scales[0])
    planeSliceG(lims[-2], lims[-2], dso, dsl, f, -dm[1], m[1], n[1], scales[0], scales[0])
    planeSliceG(lims[-2], lims[-2], dso, dsl, f, -dm[2], m[1], n[0], scales[-3], scales[-3], gsizex = gsize[1], gsizey = gsize[1])
    planeSliceG(lims[-3], lims[-4], dso, dsl, f, -dm[2], m[0], n[0], scales[-3], scales[-1], gsizex = gsize[2], gsizey = gsize[2])
    planeSliceG(lims[-3], lims[-4], dso, dsl, f, -dm[2], m[-1], n[3], scales[-3], scales[-1], gsizex = gsize[2], gsizey = gsize[2])
    planeSliceG(lims[2], lims[2], dso, dsl, f, -dm[0], m[2], n[-3], scales[0], scales[0])
    planeSliceG(lims[-2], lims[-2], dso, dsl, f, -dm[1], m[2], n[-1], scales[0], scales[0])
    planeSliceG(lims[-2], lims[-2], dso, dsl, f, 4e-6*pctocm, m[2], n[-1], scales[-4], scales[-4], gsizex = gsize[1], gsizey = gsize[1])
    planeSliceG(lims[2], lims[2], dso, dsl, f, -dm[-1], m[2], n[-2], scales[-2], scales[-2], gsizex = gsize[-1], gsizey = gsize[-1])
    return


# findRoots(causticEqSlice, 4., 4., args=(alp, 1., 0.5, ax, ay), plot = True)
# runTests()
# solveKDI(2., 2., dso, dsl, f, dm, ax, ay, 2048, 2048, m = 1, n = 0.5)
# tdm0 = c*re*dm/(2*pi*f**2)
# # print(tdm0)
# m, n = 1., 0.5
# planeSliceG(2., 2., dso, dsl, f, dm, m, n, ax, ay, gsizex = 1.5*2048, gsizey = 1.5*2048)
# rF2 = rFsqr(dso, dsl, f)
# # print(rF2)
# lc = lensc(dm, f)
# # uF2x = rF2/ax**2
# # uF2y = rF2/ay**2
# # print(uF2x)
# # print(uF2y)
# coeff = alp*np.array([1./ax**2, 1./ay**2])
# print(coeff)
# print(lc)
# uxmax = 2
# uymax = 1.5
