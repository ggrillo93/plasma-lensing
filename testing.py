from solvers import *
from observables import *
from upslice import *
from kdi import *

# dso, dsl, f, dm, ax, ay = 1.1*kpc*pctocm, 0.55*kpc*pctocm, 0.8*GHz, -5e-6*pctocm, 0.02*autocm, 0.02*autocm
# alp = alpha(dso, dsl, f, dm)

# planeSliceG(8., 8., dso, dsl, f, dm, 1., 3., ax, ay, gsizex = 1.5*2048, gsizey = 1.5*2048)

# Test cases
@profile
def runTests():
    dso, dsl = np.array([1.1, 0.55])*kpc*pctocm
    f = 0.8*GHz
    dm = pctocm*np.array([-1e-6, -2e-6, -5e-6, -1e-5])
    m = np.array([0.1, 0.5, 1., 1.2])
    n = np.array([0., 0.5, 1., 2.5, 3., 3.2, 4.])
    gsize = np.array([2048, 1.5*2048, 2*2048])
    lims = np.array([1.5, 2., 3., 4., 4.5, 8.])
    scales = autocm*np.array([0.02, 0.021, 0.022, 0.025, 0.03, 0.04, 0.06, 0.08])
    # planeSliceG(lims[1], lims[1], dso, dsl, f, dm[0], m[2], n[2], scales[0], scales[0])
    # planeSliceG(lims[2], lims[2], dso, dsl, f, dm[1], m[2], n[2], scales[0], scales[0])
    # print(gsize[1])
    # planeSliceG(lims[-1], lims[-1], dso, dsl, f, dm[-2], m[2], n[4], scales[0], scales[0], gsizex = gsize[1], gsizey = gsize[1])
    # planeSliceG(lims[-2], lims[-2], dso, dsl, f, dm[-1], m[2], n[-3], scales[-3], scales[-3], gsizex = gsize[-1], gsizey = gsize[-1])
    # planeSliceG(lims[0], lims[0], dso, dsl, f, dm[0], m[1], n[1], scales[0], scales[1])
    # planeSliceG(lims[0], lims[0], dso, dsl, f, dm[0], m[1], n[1], scales[0], scales[2])
    # planeSliceG(lims[0], lims[0], dso, dsl, f, dm[0], m[1], n[0], scales[0], scales[3])
    # planeSliceG(lims[2], lims[1], dso, dsl, f, dm[1], m[1], n[0], scales[0], scales[3])
    # planeSliceG(lims[3], lims[2], dso, dsl, f, dm[2], m[1], n[0], scales[4], scales[5], gsizex = gsize[1], gsizey = gsize[1])
    # planeSliceG(lims[-2], lims[-2], dso, dsl, f, -dm[1], m[1], n[0], scales[0], scales[0])
    # planeSliceG(lims[-2], lims[-2], dso, dsl, f, -dm[1], m[1], n[1], scales[0], scales[0])
    # planeSliceG(lims[-2], lims[-2], dso, dsl, f, -dm[2], m[1], n[0], scales[-3], scales[-3], gsizex = gsize[1], gsizey = gsize[1])
    planeSliceG(lims[-3], lims[-4], dso, dsl, f, -dm[2], m[0], n[0], scales[-3], scales[-1], gsizex = gsize[1], gsizey = gsize[1])
    planeSliceG(lims[-3], lims[-4], dso, dsl, f, -dm[2], m[-1], n[3], scales[-3], scales[-1], gsizex = gsize[1], gsizey = gsize[1])
    planeSliceG(lims[2], lims[2], dso, dsl, f, -dm[0], m[[2], n[-3], scales[0], scales[0]])
    planeSliceG(lims[-1], lims[-1], dso, dsl, f, -dm[1], m[2], n[-1], scales[0], scales[0])
    planeSliceG(lims[-1], lims[-1], dso, dsl, f, 4e-6*pctocm, m[2], n[-1], gsizex = gsize[1], gsizey = gsize[1])
    planeSliceG(lims[2], lims[2], dso, dsl, f, -dm[-1], m[2], n[-2], scales[-2], scales[-2], gsizex = gsize[-1], gsizey = gsize[-1])
    return

runTests()
# tdm0 = c*re*dm/(2*pi*f**2)
# # print(tdm0)
# m, n = 1., 4.
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
# print(4*ax**2*uxmax**2/(pi*rF2))
# print(4*ay**2*uymax**2/(pi*rF2))
# Test KDI code
# solveKDI(2., 2., dso, dsl, f, dm, ax, ay, 2048, 2048, m = m, n = n)

# distances = np.linspace(0.1, 0.5, 10)*kpc*pctocm
# # Test plotting of caustic surfaces and caustic intersections
# causPlotter(2.5, 2.5, alp, ax, ay, m = m, n = n)
# planeSliceTOA(4., 4., dso, dsl, f, dm, m, n, ax, ay, 1000)
# planeSliceG(4.5, 4.5, dso, dsl, f, dm, m, n, ax, ay, npoints = 2000, gsizex = 2*2048, gsizey = 2*2048)
# Test slice.py

# Test findRoots for complex quantities
# upvec = [-3 + 0j, 0 + 0j]
# roots = findRoots(compLensEq, 5. + 5j, 5. + 5j, args = (upvec, coeff), plot = True)
# print(roots)

# Test infite GO gain at caustics
# roots = polishedRoots(causticEqSlice, 5, 5, args = (alp, m, n, ax, ay), plot = True)
# for root in roots:
#     print(GOAmplitude(root, rF2, lc, ax, ay))

# Test whether phiA and phiB are equivalent

# upvec = [-0.1, m*-0.1 + n]
# print(upvec)
# roots = polishedRoots(lensEq, 1., 1., args = (upvec, coeff))
# print(roots)
# for root in roots:
#     print(phiB(root, upvec, rF2, lc, ax, ay))
#     print(phiA(root, rF2, lc, ax, ay))

# Test physGain

# roots = polishedRoots(causticEqSlice, 5, args = (gam, 0.5, 0.3, ax, ay), plot = True)
# for root in roots:
#     # g = gauss(root[0], root[1])
#     # upx = root[0]*(1 - 2*gam*g/ax**2)
#     # upy = root[1]*(1 - 2*gam*g/ay**2)
#     # upvec = np.array([upx, upy])
#     #print('Real gradient = ' + str([ax**2/rF2*(root[0] - upx) + lc*gaussx(*root), ay**2/rF2*(root[1] - upy) + lc*gaussy(*root)]))
#     print("GO gain = " + str(GOgain(root, gam, ax, ay)))
#     # print(physGainA(root, upvec, rF2, lc, ax, ay))
#     # print(physGainB(root, rF2, lc, ax, ay))
#     print('Phys gain 1 = ' + str(physGainA(root, rF2, lc, ax, ay)))
#     #print('Phys gain 2 = ' + str(physGainB(root, rF2, lc, ax, ay)))

# Test whether deltatA and deltatB are equivalent
# upvec = [2.5, 1.]
# roots = polishedRoots(lensEq, 5., args = (upvec, coeff))
# tdm0 = c*re*dm/(2*pi*f**2)
# dlo = dso - dsl
# tg0 = dso/(2*c*dsl*dlo)
#
# for root in roots:
#     print(deltatA(root, tg0, tdm0, gam, ax, ay))
#     print(deltatB(root, upvec, tg0, tdm0, ax, ay))

# Test whether deltaDMA and deltaDMB are equivalent
# upvec = [0.1, 0.1]
# roots = polishedRoots(lensEq, 5., args = (upvec, coeff), plot = True)
# print(roots)
# tdm0 = c*re*dm/(2*pi*f**2)
# dlo = dso - dsl
# tg0 = dso/(2*c*dsl*dlo)
#
# for root in roots:
#     gain = GOgain(root, gam, ax, ay, absolute = False)
#     print(deltaDMA(root, tg0, tdm0, gam, ax, ay, f, gain))
#     print(deltaDMB(root, upvec, tg0, tdm0, gam, ax, ay, f, gain))
