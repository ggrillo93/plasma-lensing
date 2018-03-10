from solvers import *
from observables import *
from upslice import *
from kdi import *

dso, dsl, f, dm, ax, ay = 1.1*kpc*pctocm, 0.55*kpc*pctocm, 0.8*GHz, 5e-6*pctocm, 0.04*autocm, 0.04*autocm
alp = alpha(dso, dsl, f, dm)

# tdm0 = c*re*dm/(2*pi*f**2)
# print(tdm0)
m, n = 1., 1.
# rF2 = rFsqr(dso, dsl, f)
# print(rF2)
lc = lensc(dm, f)
# uF2x = rF2/ax**2
# uF2y = rF2/ay**2
# print(uF2x)
# print(uF2y)
coeff = alp*np.array([1./ax**2, 1./ay**2])
print(coeff)
print(lc)
# uxmax = 2
# uymax = 1.5
# print(4*ax**2*uxmax**2/(pi*rF2))
# print(4*ay**2*uymax**2/(pi*rF2))
# Test KDI code
# solveKDI(2., 2., dso, dsl, f, dm, ax, ay, 2048, 2048, m = m, n = n)

# distances = np.linspace(0.1, 0.5, 10)*kpc*pctocm
# # Test plotting of caustic surfaces and caustic intersections
# causPlotter(4, 4, alp, ax, ay, m = m, n = n)
# planeSlice(1.5, dso, dsl, f, dm, m, n, ax, ay, npoints = 150)
planeSliceG(3., 3., dso, dsl, f, dm, m, n, ax, ay, npoints = 5000, gsizex = 2*2048, gsizey = 2*2048)
# Test slice.py

# Test findRoots for complex quantities
# upvec = [-3 + 0j, 0 + 0j]
# roots = findRoots(compLensEq, 5. + 5j, 5. + 5j, args = (upvec, coeff), plot = True)
# print(roots)

# Test infite GO gain at caustics
# roots = polishedRoots(causticEqSlice, 5, args = (gam, 0.1, 0.1, ax, ay), plot = True)
# for root in roots:
#     print(gain(roots, gam, ax, ay))

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
