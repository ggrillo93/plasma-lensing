from observables import *
from solvers import *
import time
        
def fslicepert(dso, dsl, dm, upvec, T, ax, ay, fc, nch, bw, spacing, template = None, plot = True, noise = 0.1, comp = True):
    
    start = time.time()
    nsam = findN(bw, nch, T)
    taxis = np.linspace(-T/2., T/2., nsam)
    
    if template == None:
        template = gausstemp(np.linspace(-T/2., T/2., 2048), T/10., 1., 0.)
        
    delta = unit_impulse(nsam, 'mid')
    ft = fftshift(fft(delta))
    freqs = fftfreq(nsam, d= T/nsam)
    nfactor = 0.25
    fmin = min(freqs) + fc
    fmax = max(freqs) + fc
    
    # Calculate coefficients
    fcoeff = dsl*(dso - dsl)*re*dm/(2*pi*dso)
    alpp = alpha(dso, dsl, 1., dm)
    coeff = alpp*np.array([1./ax**2, 1./ay**2])
    rF2p = rFsqr(dso, dsl, 1.)
    lcp = lensc(dm, 1.)
    tdm0p = tdm0coeff(dm, 1.)
    tg0 = tg0coeff(dso, dsl)
    
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
    print(fcross/GHz)
        
    cdist = 1e5 # set minimum caustic distance

    # Set up boundaries
    if ncross != 0:
        bound = np.insert(fcross, 0, fmin)
        bound = np.append(bound, fmax)
        midpoints = [(bound[i] + bound[i+1])/2. for i in range(len(bound) - 1)] # find middle point between boundaries
        nzones = len(midpoints)
        nreal = np.zeros(nzones, dtype = int)
        for i in range(nzones):
            mpoint = midpoints[i]
            leqcoeff = coeff/mpoint**2
            nreal[i] = len(findRoots(lensEq, np.abs(upx) + 3., np.abs(upy) + 3., args = (upvec, leqcoeff), N = 1000))
        segs = np.array([np.arange(bound[i-1] + cdist, bound[i] - cdist, spacing) for i in range(1, ncross + 2)])
        if comp == True:
            diff = difference(nreal)
            ncomplex = np.ones(nzones)*100
            for i in range(nzones):
                if diff[i] == 0 or diff[i] == -2:
                    ncomplex[i] = 1
                elif diff[i] == -4:
                    ncomplex[i] = 2
                elif diff[i] == 4 or diff[i] == 2:
                    ncomplex[i] = 0
        else:
            ncomplex = np.zeros(nzones)
    else:
        segs = [np.arange(fmin, fmax, spacing)]
        mpoint = segs[0][len(segs[0])/2]
        leqcoeff = coeff/mpoint**2
        nreal = [len(findRoots(lensEq, np.abs(upx) + 3., np.abs(upy) + 3., args = (upvec, leqcoeff), N = 1000))]
        ncomplex = [0]
        nzones = 1
    print(nreal)
    print(ncomplex)
    
    # Solve lens equation at each coordinate
    allroots = rootFinderFreq(segs, nreal, ncomplex, ucrossb, upvec, coeff)
    
    # Calculate fields
    allfields = []
    allgains = []
    alltoas = []
    for l in range(nzones):
        nroots = len(allroots[l][0])
        fvec = segs[l]
        roots = allroots[l]
        npoints = len(fvec)
        fields = np.zeros([nroots, 2, npoints], dtype=complex)
        gains = np.zeros([nroots, npoints])
        toas = np.zeros([nroots, npoints])
        for i in range(npoints):
            freq = fvec[i]
            rF2 = rF2p/freq
            lc = lcp/freq
            tdm0 = tdm0p/freq**2
            for j in range(nroots):
                ans = GOfield(roots[i][j], rF2, lc, ax, ay)
                gains[j][i] = np.abs(ans[0])**2
                toas[j][i] = deltat(roots[i][j].real, tg0, tdm0, rF2*lc, ax, ay)
                for k in range(2):
                    fields[j][k][i] = ans[k]
        allfields.append(fields)
        allgains.append(gains)
        alltoas.append(toas)
    
    asympfuncs = uniAsymp(allfields, segs, nreal, ncomplex)
    fvec = fftshift(freqs) + fc
    if ncross != 0:
        splitat = fvec.searchsorted(fcross)
        splfvec = np.array_split(fvec, splitat)
        splft = np.array_split(ft, splitat)
        lspec = np.array([], dtype = complex)
        for i in range(nzones):
            seg = splfvec[i]
            orzf = splft[i]
            zf = np.zeros(len(seg), dtype = complex)
            for root in asympfuncs[i]:
                amp = root[0](seg)
                phase = root[1](seg)
                zf = zf + orzf*amp*np.exp(1j*phase)
            lspec = np.append(lspec, zf)
    else:
        lspec = np.zeros(len(fvec), dtype = complex)
        for root in asympfuncs[0]:
            amp = root[0](fvec)
            phase = root[1](fvec)
            lspec = lspec + ft*amp*np.exp(1j*phase)
            
    chlspec = np.split(lspec, nch)
    chfvec = np.split(fvec, nch)
    chft = np.split(ft, nch)
    dspec = np.zeros([nch, 2048])
    toapert = np.zeros(nch)
    noisevec = noise*randn(nch, 2048)
    
    # Calculate initial TOA perturbation
    # ift = ifft(ifftshift(chft[0]))
    # I = groupedAvg(np.abs(ift)**2, N=nsam/(2048*nch))/nfactor
    # I = fftconvolve(I, template, mode = 'same')
    # toapertor = tempmatch(I, template, T/2048.)[0]*1e6
    # print(toapertor)
    toapertor = 2.4414
    
    # Calculate lens TOA perturbation
    for i in range(nch):
        ift = ifft(ifftshift(chlspec[i]))
        I = groupedAvg(np.abs(ift)**2, N = nsam/(2048*nch))/nfactor
        I = fftconvolve(I, template, mode = 'same') + noisevec[i]
        # I = np.flipud(I)
        toapert[i] = toapertor - tempmatch(I, template, T/2048.)[0]*1e6

    cfreqvec = np.mean(chfvec, axis = 1)
    
    plt.close()

    print 'It took', time.time()-start, 'seconds.'
    band1 = (cfreqvec/GHz > 0.73) * (cfreqvec/GHz < 0.91)
    band2 = (cfreqvec/GHz > 1.15) * (cfreqvec/GHz < 1.88)
    band3 = (cfreqvec/GHz > 2.05) * (cfreqvec/GHz < 2.4)
    av1 = np.average(toapert[np.asarray(np.where(band1)).flatten()])
    av2 = np.average(toapert[np.asarray(np.where(band2)).flatten()])
    av3 = np.average(toapert[np.asarray(np.where(band3)).flatten()])
    print([av1, av2, av3])
    
    if plot:
        fig = plt.figure(figsize = (12, 8), dpi = 100)
        grid = gs.GridSpec(3, 1)
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0], sharex = ax1)
        ax3 = fig.add_subplot(grid[2, 0], sharex = ax1)
        plt.setp(ax1.get_xticklabels(), visible = False)
        plt.setp(ax2.get_xticklabels(), visible = False)
        colors = assignColor(allroots, nreal)
        for i in range(len(segs)):
            gains = allgains[i]
            toas = alltoas[i]
            for j in range(nreal[i]):
                ax1.plot(segs[i]/GHz, gains[j], color = colors[i][j])
                ax2.plot(segs[i]/GHz, toas[j], color = colors[i][j])
        ax1.set_yscale('log')
        ax1.axhline(y = 1, ls='dashed', color='black')
        ax1.set_xlim([fmin/GHz, fmax/GHz])
        ax1.set_ylabel(r'$G_j$', fontsize = 20)
        ax2.set_ylabel(r'$\Delta t_{j}$ ($\mu s$)', fontsize = 20)
        ax2.axhline(y = 0, ls='dashed', color='black')
        ax2.set_yscale('symlog')
        axes = [ax1, ax2, ax3]
        for i in range(len(fcross)):
            for axis in axes:
                axis.axvline(x = fcross[i]/GHz, color = 'black', ls = ':')
        ax3.scatter(cfreqvec/GHz, toapert, color='black', s = 1.5)
        ax3.set_ylabel(r'$\Delta t_{tot}$ ($\mu s$)', fontsize = 20)
        ax3.set_xlabel(r'$\nu$ (GHz)', fontsize=20)
        ax3.axhline(y = 0, ls='dashed', color='black')
        ax3.set_ylim(np.min(toapert) - 1, np.max(toapert) + 1)
        ax3.tick_params(labelsize = 16)
        ax2.tick_params(labelsize = 16)
        ax1.tick_params(labelsize = 16)
        
        plt.tight_layout()
        plt.show()
        
    return

def fslicepertBulk(upvec, fcaus, fmin, fmax, leqinv, ax, ay, rx, ry, uvec, rF2p, lcp, alpp, tdm0p, tg0, coeff, taxis, df, dt, cdist, npoints, template, period, plot, noise, ntoas, spacing, tsize = 2048):
    
    ucross, fcross = fcaus
    ncross = len(ucross)
    print(upvec)
    
    # Calculate sign of second derivative at caustics
    sigs = np.zeros(ncross)
    for i in range(ncross):
        rF2 = rF2p/fcross[i]
        lc = lcp/fcross[i]
        sigs[i] = np.sign(ax**2/rF2 + lc*lensh(ucross[i][0], ucross[i][1])[0])
        
    cdist = 1e5 # set minimum caustic distance

    # Set up boundaries
    bound = np.insert(fcross, 0, fmin)
    bound = np.append(bound, fmax)
    midpoints = [(bound[i] + bound[i+1])/2. for i in range(len(bound) - 1)] # find middle point between boundaries
    nzones = len(midpoints)
    nreal = np.zeros(nzones, dtype = int)
    for i in range(nzones):
        mpoint = midpoints[i]
        leqinvtemp = leqinv/mpoint**2
        evleq = np.array([uvec[0] + leqinvtemp[0] - upvec[0], uvec[1] + leqinvtemp[1] - upvec[1]])
        nreal[i] = len(findRootsBulk(evleq, rx, ry))
    segs = np.array([np.arange(bound[i-1] + cdist, bound[i] - cdist, spacing) for i in range(1, ncross + 2)])
    
    # Calculate coefficient for each frequency
    alp = alpp/segs**2
    rF2 = rF2p/segs
    lc = lcp/segs
    tdm0 = tdm0p/segs**2
    
    ncomplex = np.zeros(nzones) # only real rays
    
    # Solve lens equation at each coordinate
    allroots = rootFinderFreqBulk(segs, nreal, ncomplex, ucross, upvec, uvec, leqinv, rx, ry, coeff)
    
    singleim = np.argwhere(nreal == 1).flatten()
    multiim = np.argwhere(nreal > 1).flatten()
    
    allgains = list(np.zeros(nzones))
    alltoas = list(np.zeros(nzones))
    toapert = np.array([np.zeros(len(segs[i])) for i in range(nzones)])
    
    # Calculate gain and TOA perturbation for regions with a single image
    for i in singleim:
        fvec = segs[i]
        npoints = len(fvec)
        gain = np.zeros(npoints)
        toas = np.zeros(npoints)
        roots = allroots[i]
        for j in range(npoints):
            gain[j] = GOAmp(roots[j].real[0], rF2[i][j], lc[i][j], ax, ay)**2
            toas[j] = deltat(roots[j].real[0], tg0, tdm0[i][j], alp[i][j], ax, ay)
            noisytemp = (template + noise*np.random.randn(tsize))**2
            orpulse = pp.SinglePulse(noisytemp)
            pulse = gain[j]*orpulse.shiftit(toas[j]/dt)
            toapert[i][j] = tempmatch(pulse, template**2, dt)
        allgains[i] = [gain]
        alltoas[i] = [toas]
    
    # Calculate field components, TOAs, gains for regions with more than one image
    allfields = list(np.zeros(nzones))
    for i in multiim:
        npoints = len(segs[i])
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
    orpulse = pp.SinglePulse(template)
    
    for i in multiim:
        
        npoints = len(segs[i])
        
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
                    noisytemp = (template + noise*np.random.randn(tsize))**2
                    orpulse = pp.SinglePulse(noisytemp)
                    pulse = regfields[l][k]*orpulse.shiftit(regtoas[l][k]/dt)
                    tpulse = tpulse + pulse
                regpert[k] = tempmatch(np.abs(tpulse)**2, template**2, dt)
            zonepert = np.append(zonepert, regpert)
        toapert[i] = zonepert
    
    toapert = np.hstack(toapert)
    fvec = np.hstack(segs)
    
    wind = int(3e6/spacing)
    end = len(toapert) - len(toapert)%wind
    toapert = np.mean(toapert[:end].reshape(-1, wind), 1)
    fvec = np.mean(fvec[:end].reshape(-1, wind), 1)
    
    band1 = (fvec > 0.73*GHz) * (fvec < 0.91*GHz)
    band2 = (fvec > 1.15*GHz) * (fvec < 1.88*GHz)
    band3 = (fvec > 2.05*GHz) * (fvec < 2.4*GHz)
    av1 = np.average(toapert[np.asarray(np.where(band1)).flatten()])
    av2 = np.average(toapert[np.asarray(np.where(band2)).flatten()])
    av3 = np.average(toapert[np.asarray(np.where(band3)).flatten()])
    
    f_handle = file("epochs" + str(ntoas) + ".dat", 'a')
    np.savetxt(f_handle, np.array([fvec, toapert]))
    f_handle.close()
    
    f_handle2 = file('upx' + str(ntoas) + '.dat', 'a')
    np.savetxt(f_handle2, np.array([upvec[0]]))
    f_handle2.close()
    
    if plot:
        fig = plt.figure(figsize = (15, 10))
        grid = gs.GridSpec(3, 2, width_ratios=[4, 1])
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0], sharex = ax1)
        ax3 = fig.add_subplot(grid[2, 0], sharex = ax1)
        plt.setp(ax1.get_xticklabels(), visible = False)
        plt.setp(ax2.get_xticklabels(), visible = False)
        tableax2 = plt.subplot(grid[1:, 1])
        tableax1 = plt.subplot(grid[0, 1])
        colors = assignColor(allroots, nreal)
        for i in range(len(segs)):
            gains = allgains[i]
            toas = alltoas[i]
            for j in range(len(gains)):
                ax1.plot(segs[i]/GHz, gains[j], color = colors[i][j])
                ax2.plot(segs[i]/GHz, toas[j], color = colors[i][j])
        ax1.plot([-1, 10], [1, 1], ls='dashed', color='black')
        ax1.set_xlim([fmin/GHz, fmax/GHz])
        ax1.set_yscale('log')
        ax1.set_ylabel('G')
        ax2.set_ylabel(r'$\Delta t_{ind}$ ($\mu s$)')
        ax2.plot([-1, 10], [0, 0], ls='dashed', color='black')
        ax2.set_yscale('symlog')
        axes = [ax1, ax2, ax3]
        for i in range(len(fcross)):
            for j in range(3):
                axes[j].plot([fcross[i]/GHz, fcross[i]/GHz], [-1000, 1000], color = 'black', ls = ':', scaley = False, scalex = False)
        ax3.plot(fvec/GHz, toapert, color='black')
        ax3.set_ylabel(r'$\Delta t_{comb}$ ($\mu s$)')
        ax3.set_xlabel(r'$\nu$ (GHz)')
        ax3.plot([-1, 10], [0, 0], ls='dashed', color='black')
        ax3.set_ylim(np.min(toapert) - 1, np.max(toapert) + 1)
        
        # Tables
        clabels = ['Band', r'$\overline{\Delta t}_{comb}$ ($\mu s$)']
        tvals = [['820 MHz', np.around(av1, 4)], ['1400 MHz', np.around(av2, 4)], ['2300 MHz', np.around(av3, 4)]]
        tableax1.axis('tight')
        tableax1.axis('off')
        table1 = tableax1.table(cellText = tvals, colWidths = [0.25, 0.25], colLabels = clabels, loc = 'center')
        table1.auto_set_font_size(False)
        table1.set_fontsize(12)
        table1.scale(3., 3.)
        
        
        # col_labels = ['Parameter', 'Value']
        # if np.abs(dm/pctocm) < 1:
        #     dmlabel = "{:.2E}".format(Decimal(dm/pctocm))
        # else:
        #     dmlabel = str(dm/pctocm)
        # tablevals = [[r'$d_{so} \: (kpc)$', np.around(dso/pctocm/kpc, 2)], [r'$d_{sl} \: (kpc)$', np.around(dsl/pctocm/kpc, 2)], [r'$a_x \: (AU)$', np.around(ax/autocm, 2)], [r'$a_y \: (AU)$', np.around(ay/autocm, 2)], [r'$DM_l \: (pc \, cm^{-3})$', dmlabel], [r"$\vec{u}'$", np.around(upvec, 2)]]
        # tableax2.axis('tight')
        # tableax2.axis('off')
        # table = tableax2.table(cellText = tablevals, colWidths = [0.25, 0.25], colLabels = col_labels, loc = 'center')
        # table.auto_set_font_size(False)
        # table.set_fontsize(12)
        # table.scale(3., 3.)
        
        plt.show()
        
    return np.array([av1, av2, av3])
        
