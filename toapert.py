from observables import *
from solvers import *

def fslicepert(upvec, fmin, fmax, dso, dsl, dm, ax, ay, template, period, npoints = 3000, plot = True):
    
        # Calculate coefficients
    fcoeff = dsl*(dso - dsl)*re*dm/(2*pi*dso)
    alpp = alpha(dso, dsl, 1., dm)
    coeff = alpp*np.array([1./ax**2, 1./ay**2])
    rF2p = rFsqr(dso, dsl, 1.)
    lcp = lensc(dm, 1.)
    
    upx, upy = upvec
    ucross = polishedRoots(causEqFreq, np.abs(upx) + 3., np.abs(upy) + 3., args = (upx, ax, ay, upy/upx, 0))
    fcross = []
    ucrossb = []
    for uvec in ucross:
        ux, uy = uvec
        arg = fcoeff*lensg(ux, uy)[0]/(ux - upx)
        if arg > 0:
            freq = c*np.sqrt(arg)/ax
            if fmin < freq < fmax:
                fcross.append(freq)
                ucrossb.append([ux, uy])
    fcross = np.asarray(fcross)
    p = np.argsort(fcross)
    fcross = fcross[p]
    ucrossb = np.asarray(ucrossb)[p]
    ncross = len(fcross)
    
    # Calculate sign of second derivative at caustics
    sigs = np.zeros(ncross)
    for i in range(ncross):
        rF2 = rFsqr(dso, dsl, fcross[i])
        lc = lensc(dm, fcross[i])
        sigs[i] = np.sign(ax**2/rF2 + lc*lensh(ucrossb[i][0], ucrossb[i][1])[0])
        
    cdist = 1e6

    # Set up boundaries
    bound = np.insert(fcross, 0, fmin)
    bound = np.append(bound, fmax)
    midpoints = [(bound[i] + bound[i+1])/2. for i in range(len(bound) - 1)] # find middle point between boundaries
    nzones = len(midpoints)
    nreal = np.zeros(nzones)
    for i in range(nzones):
        mpoint = midpoints[i]
        leqcoeff = coeff/mpoint**2
        nreal[i] = len(findRoots(lensEq, np.abs(upx) + 3., np.abs(upy) + 3., args = (upvec, leqcoeff), N = 1000))
    segs = np.array([np.linspace(bound[i-1] + cdist, bound[i] - cdist, npoints) for i in range(1, ncross + 2)])
    print(nreal)
    
    # Solve lens equation at each coordinate
    allroots = rootFinderFreq(segs, nreal, ncomplex, npoints, ucrossb, upvec, coeff)
    
    # Calculate fields
    allfields = []
    for l in range(nzones):
        nroots = len(allroots[l][0])
        fvec = segs[l]
        roots = allroots[l]
        fields = np.zeros([nroots, 3, npoints], dtype = complex)
        for i in range(npoints):
            freq = fvec[i]
            rF2 = rF2p/freq
            lc = lcp/freq
            for j in range(nroots):
                ans = GOfield(roots[i][j], rF2, lc, ax, ay)
                for k in range(3):
                    fields[j][k][i] = ans[k]
        allfields.append(fields)
    
    merged = []
    for i in range(nzones):
        if nreal[i] > 1:
            merged.append(uniAsympTOA(allroots[i], allfields[i], nreal[i], npoints, sigs[i]))
    
    combfields = []
    for i in range(nzones):
        for j in range()