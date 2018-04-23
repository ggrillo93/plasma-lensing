from observables import *
from solvers import *
from scipy.ndimage import uniform_filter1d as f1d
import time

def fslicepert(upvec, fmin, fmax, dso, dsl, dm, ax, ay, template, period, npoints = 3000, tsize = 2048, plot = True):
    
    start = time.time()
    sqrtempl = template**2
    
    def tempmatch(data):
        pulse = pp.SinglePulse(data)
        # pulse.plot()
        shift = pulse.fitPulse(sqrtempl)[1]
        ans = shift*dt
        # print(ans)
        return ans
    
    # Calculate coefficients
    fcoeff = dsl*(dso - dsl)*re*dm/(2*pi*dso)
    alpp = alpha(dso, dsl, 1., dm)
    coeff = alpp*np.array([1./ax**2, 1./ay**2])
    rF2p = rFsqr(dso, dsl, 1.)
    lcp = lensc(dm, 1.)
    tg0 = tg0coeff(dso, dsl)
    tdm0p = tdm0coeff(dm, 1.)
    
    # Find frequency caustics
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
    
    taxis = np.linspace(-period/2., period/2., 2048)
    
    # Calculate sign of second derivative at caustics
    sigs = np.zeros(ncross)
    for i in range(ncross):
        rF2 = rFsqr(dso, dsl, fcross[i])
        lc = lensc(dm, fcross[i])
        sigs[i] = np.sign(ax**2/rF2 + lc*lensh(ucrossb[i][0], ucrossb[i][1])[0])
        
    cdist = 1e5 # set minimum caustic distance

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
    
    # Calculate coefficient for each frequency
    alp = alpp/segs**2
    rF2 = rF2p/segs
    lc = lcp/segs
    tdm0 = tdm0p/segs**2
    
    ncomplex = np.zeros(nzones) # only real rays
    
    df = (fmax - fmin - 2*cdist)/npoints # frequency grid spacing
    dt = period/tsize # time axis spacing
    
    print(nreal)
    
    # Solve lens equation at each coordinate
    allroots = rootFinderFreq(segs, nreal, ncomplex, npoints, ucrossb, upvec, coeff)
    
    singleim = np.argwhere(nreal == 1).flatten()
    multiim = np.argwhere(nreal > 1).flatten()
    
    allgains = list(np.zeros(nzones))
    alltoas = list(np.zeros(nzones))
    toapert = np.zeros([nzones, npoints])
    
    # Calculate gain and TOA perturbation for regions with a single image
    for i in singleim:
        fvec = segs[i]
        gain = np.zeros(npoints)
        roots = allroots[i]
        for j in range(npoints):
            gain[j] = GOAmp(roots[j].real[0], rF2[i][j], lc[i][j], ax, ay)**2
            toapert[i][j] = deltat(roots[j].real[0], tg0, tdm0[i][j], alp[i][j], ax, ay)
        allgains[i] = [gain]
        alltoas[i] = [toapert[i]]
    
    # Calculate field components, TOAs, gains for regions with more than one image
    allfields = list(np.zeros(nzones))
    for i in multiim:
        nroots = nreal[i]
        roots = allroots[i]
        fields = np.zeros([nroots, 3, npoints], dtype = complex)
        toas = np.zeros([nroots, npoints])
        gains = np.zeros([nroots, npoints])
        for j in range(npoints):
            for k in range(nroots):
                field = GOfield(roots[j][k], rF2[i][j], lc[i][j], ax, ay)
                gains[k][j] = np.abs(field[0])**2
                toas[k][j] = deltat(roots[j][k].real, tg0, tdm0[i][j], alp[i][j], ax, ay)
                for l in range(3):
                    fields[k][l][j] = field[l]
        allfields[i] = fields
        alltoas[i] = toas
        allgains[i] = gains
    
    # Create pulses and calculate TOAs
    # h = int(500*cdist/df) # inner boundary
    orpulse = pp.SinglePulse(template)
    
    for i in multiim:
        if i == 0:
            sig = [sigs[0], sigs[0]]
        elif  i == nzones - 1:
            sig = [sigs[-1], sigs[-1]]
        else:
            sig = [sigs[i - 1], sigs[i]]
        
        fields, toas = uniAsympTOA(allroots[i], allfields[i], alltoas[i], nreal[i], npoints, sig)
        zonepert = np.array([])
        for j in range(len(fields)):
            regfields = fields[j]
            regtoas = toas[j]
            nptsreg = len(regfields[0])
            nimreg = len(regfields)
            regpert = np.zeros(nptsreg)
            for k in range(nptsreg):
                tpulse = np.zeros(tsize)
                for l in range(nimreg):
                    pulse = np.abs(regfields[l][k])*orpulse.shiftit(regtoas[l][k]/dt)
                    tpulse = tpulse + pulse
                regpert[k] = tempmatch(np.abs(tpulse)**2)
            zonepert = np.append(zonepert, regpert)
        toapert[i] = zonepert
    
    toapert = toapert.flatten()
    print 'It took', time.time()-start, 'seconds.'
    sband = (segs.flatten() > 0.7*GHz) * (segs.flatten() < 0.9*GHz)
    print 'Av. TOA pert for 700-900 MHz = ', np.average(toapert[np.asarray(np.where(sband)).flatten()])
    
    if plot:
        fig = plt.figure(figsize = (18, 15))
        gs = gs.GridSpec(2, 2, width_ratios=[4, 1])
        axarr[2].plot(segs.flatten()/GHz, toapert, color = 'black')
        # interp = UnivariateSpline(segs.flatten(), toapert.flatten())
        # axarr[2].plot(segs.flatten()/GHz, interp(segs.flatten()), color = 'blue')
        axarr[2].set_ylabel(r'$\Delta t_{comb}$ ($\mu s$)')
        axarr[2].set_xlabel(r'$\nu$ (GHz)')
        axarr[2].plot([-1, 10], [0, 0], ls = 'dashed', color = 'black')
        axarr[2].set_xlim([fmin/GHz, fmax/GHz])
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        for i in range(len(segs)):
            gains = allgains[i]
            toas = alltoas[i]
            for j in range(len(gains)):
                axarr[0].plot(segs[i]/GHz, gains[j], color = colors[j])
                axarr[1].plot(segs[i]/GHz, toas[j], color = colors[j])
        axarr[0].set_yscale('log')
        axarr[0].set_ylabel('G')
        axarr[1].set_ylabel(r'$\Delta t_{ind}$ ($\mu s$)')
        axarr[1].set_yscale('symlog')
        for i in range(len(fcross)):
            for j in range(3):
                axarr[j].plot([fcross[i]/GHz, fcross[i]/GHz], [-1000, 1000], color = 'black', ls = 'dashed', scaley = False, scalex = False)
        plt.show()
        
    return
