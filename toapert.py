from observables import *
from solvers import *

def fslicepert(upvec, fmin, fmax, dso, dsl, dm, ax, ay, template, period, npoints = 3000, plot = True):
    
    # Calculate coefficients
    fcoeff = dsl*(dso - dsl)*re*dm/(2*pi*dso)
    alpp = alpha(dso, dsl, 1., dm)
    coeff = alpp*np.array([1./ax**2, 1./ay**2])
    rF2p = rFsqr(dso, dsl, 1.)
    lcp = lensc(dm, 1.)
    tg0 = tg0coeff(dso, dsl)
    tdm0p = tdm0coeff(dm, 1.)
    
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
    nreal = np.zeros(nzones, dtype = int)
    for i in range(nzones):
        mpoint = midpoints[i]
        leqcoeff = coeff/mpoint**2
        nreal[i] = int(len(findRoots(lensEq, np.abs(upx) + 3., np.abs(upy) + 3., args = (upvec, leqcoeff), N = 1000)))
    segs = np.array([np.linspace(bound[i-1] + cdist, bound[i] - cdist, npoints) for i in range(1, ncross + 2)])
    ncomplex = np.zeros(nzones)
    df = (fmax - fmin - 2*cdist)/npoints
    dt = period/2048.
    print(nreal)
    
    # Solve lens equation at each coordinate
    allroots = rootFinderFreq(segs, nreal, ncomplex, npoints, ucrossb, upvec, coeff)
    
    # Calculate field components, TOAs
    allfields = []
    alltoas = []
    for l in range(nzones):
        nroots = len(allroots[l][0])
        fvec = segs[l]
        roots = allroots[l]
        fields = np.zeros([nroots, 3, npoints], dtype = complex)
        toas = np.zeros([nroots, npoints])
        for i in range(npoints):
            freq = fvec[i]
            rF2 = rF2p/freq
            lc = lcp/freq
            alp = rF2*lc
            tdm0 = tdm0p/freq**2
            for j in range(nroots):
                ans = GOfield(roots[i][j], rF2, lc, ax, ay)
                toas[j][i] = deltat(roots[i][j].real, tg0, tdm0, alp, ax, ay)
                for k in range(3):
                    fields[j][k][i] = ans[k]
        allfields.append(fields)
        alltoas.append(toas)
    
    # # Calculate combined fields for merging roots using uniform asymptotics
    # merged = []
    # for i in range(nzones):
    #     if nreal[i] > 1:
    #         merged.append(uniAsympTOA(allroots[i], allfields[i], nreal[i], npoints, sigs[i]))
    
    # # Combine field components for all roots
    # combfields = []
    # for i in range(nzones):
    #     arrsh = allroots[i].shape
    #     totfield = np.zeros(arrsh, dtype = complex)
    #     nroots = nreal[i]
    #     if nreal[i] == 1:
    #         totfield = constructField(*allfields[i])
    #     else:
    #         for j in range(nreal[i]):
    #             totfield[j] = constructField(*allfields[i][j])
    #     combfields.append(totfield)
    
    for i in range(len(segs)):
        zone = alltoas[i]
        for j in range(len(zone)):
            plt.plot(segs[i], zone[j], color='black')
    plt.yscale('symlog')
    plt.show()
    
    return
    
    # # Create pulses and calculate TOAs
    # taxis = np.linspace(-period/2., period/2., 2048)
    # toapert = []
    # for i in range(nzones):
    #     nroots = nreal[i]
    #     if nroots == 1:
    #         toapert.append(alltoas[i])
    #     else:
    #         fields = combfields[i]
    #         # far from caustics
    #         h = 100*cdist/df
    #         far = fields[h:-h]
    #         for j in range(npoints):
    #             plt.figure()
    #             tpulse = np.zeros(2048)
    #             for k in range(nroots):
    #                 pulse = np.roll(template*fields[k][j], alltoas[i][k][j]/dt)
    #                 plt.plot(taxis, np.abs(pulse)**2, color = 'red')
    #                 tpulse = tpulse + pulse
    #             plt.plot(taxis, tpulse**2, color = 'blue')
