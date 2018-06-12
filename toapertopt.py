from observables import *
from solvers import *
import time
        
def fslicepert(upvec, fmin, fmax, dso, dsl, dm, ax, ay, period, template = None, spacing = 1e5, tsize = 2048, plot = True, noise = 0.2, chw = 1.5e6):
    
    start = time.time()
    taxis = np.linspace(-period/2., period/2., tsize)
    if template == None:
        template = gaussian(taxis, period*0.1, 1., 0.)
    
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
    print(fcross/GHz)
    for i in range(ncross):
        print(causAmp(ucrossb[i], rF2p/fcross[i], lcp/fcross[i], ax, ay))
    
    
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
    segs = np.array([np.arange(bound[i-1] + cdist, bound[i] - cdist, spacing) for i in range(1, ncross + 2)])
    
    # Calculate coefficient for each frequency
    alp = alpp/segs**2
    rF2 = rF2p/segs
    lc = lcp/segs
    tdm0 = tdm0p/segs**2
    
    ncomplex = np.zeros(nzones) # only real rays
    
    # df = (fmax - fmin - 2*cdist)/npoints # frequency grid spacing
    dt = period/tsize # time axis spacing
    
    print(nreal)
    
    # Solve lens equation at each coordinate
    allroots = rootFinderFreq(segs, nreal, ncomplex, ucrossb, upvec, coeff)
    # print(allroots)
    
    singleim = np.argwhere(nreal == 1).flatten()
    multiim = np.argwhere(nreal > 1).flatten()
    
    nfpts = [len(seg) for seg in segs]
    totfpts = np.sum(nfpts)
    dspec = np.zeros([totfpts, tsize], dtype = complex)
        
    allgains = list(np.zeros(nzones))
    alltoas = list(np.zeros(nzones))
    orpulse = pp.SinglePulse(template)
    
    count = 0
    for i in range(nzones):
        fvec = segs[i]
        nf = nfpts[i]
        nroots = nreal[i]
        roots = allroots[i]
        if nroots == 1: # single image
            gain = np.zeros(nf)
            toas = np.zeros(nf)
            for j in range(nf):
                fieldcomp = GOfield(roots[j].real[0], rF2[i][j], lc[i][j], ax, ay)
                gain[j] = np.abs(fieldcomp[0])**2
                toa = deltat(roots[j].real[0], tg0, tdm0[i][j], alp[i][j], ax, ay)
                toas[j] = toa
                shiftp = orpulse.shiftit(toa/dt)
                dspec[count] = fieldcomp[0]*shiftp*np.exp(1j*(fieldcomp[1] + fieldcomp[2]))
                count = count + 1
            allgains[i] = [gain]
            alltoas[i] = [toas]
        else: # multiple images
            fields = np.zeros([nroots, 3, nf], dtype = complex)
            gain = np.zeros([nroots, nf])
            toas = np.zeros([nroots, nf])
            for j in range(nf):
                for k in range(nroots):
                    field = GOfield(roots[j][k], rF2[i][j], lc[i][j], ax, ay)
                    gain[k][j] = np.abs(field[0])**2
                    toas[k][j] = deltat(roots[j][k].real, tg0, tdm0[i][j], alp[i][j], ax, ay)
                    for l in range(3):
                        fields[k][l][j] = field[l]
            allgains[i] = gain
            alltoas[i] = toas
            if i == 0:
                sig = [sigs[0], sigs[0]]
            elif  i == nzones - 1:
                sig = [sigs[-1], sigs[-1]]
            else:
                sig = [sigs[i - 1], sigs[i]]
            fields, toas = uniAsympTOA(roots, fields, toas, nroots, nf, sig)
            for j in range(len(fields)):
                regfields = fields[j]
                # print(regfields)
                nimreg = len(regfields)
                regtoas = toas[j]
                nptsreg = len(regfields[0])
                for k in range(nptsreg):
                    tpulse = np.zeros(tsize, dtype = complex)
                    for l in range(nimreg):
                        pulse = regfields[l][k]*orpulse.shiftit(regtoas[l][k]/dt)
                        tpulse = tpulse + pulse
                    dspec[count] = tpulse
                    count = count + 1
    
    fvec = np.hstack(segs)
    wind = int(chw/spacing)
    # dspecav = np.abs(groupedAvg(dspec, N=wind))**2
    fvecav = groupedAvg(fvec, N=wind)/GHz
    nfptsav = len(fvecav)
    noise = np.random.randn([nfpts, tsize])
    dspecav = np.abs(groupedAvg(dspec, N=wind) + noise)**2
    
    toapert = np.zeros(nfptsav)
    sqrtemp = template**2
    for i in range(nfptsav):
        toapert[i] = tempmatch(dspecav[i], sqrtemp, dt)
    
    plt.close()
    
    # np.savetxt('test.dat', np.array([fvec, toapert]))

    print 'It took', time.time()-start, 'seconds.'
    band1 = (fvecav > 0.73) * (fvecav < 0.91)
    band2 = (fvecav > 1.15) * (fvecav < 1.88)
    band3 = (fvecav > 2.05) * (fvecav < 2.4)
    av1 = np.average(toapert[np.asarray(np.where(band1)).flatten()])
    av2 = np.average(toapert[np.asarray(np.where(band2)).flatten()])
    av3 = np.average(toapert[np.asarray(np.where(band3)).flatten()])
    print([av1, av2, av3])
    
    if plot:
        fig = plt.figure(figsize = (12, 8), dpi = 100)
        # grid = gs.GridSpec(3, 2, width_ratios=[4, 1])
        grid = gs.GridSpec(3, 1)
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0], sharex = ax1)
        ax3 = fig.add_subplot(grid[2, 0], sharex = ax1)
        plt.setp(ax1.get_xticklabels(), visible = False)
        plt.setp(ax2.get_xticklabels(), visible = False)
        # tableax2 = plt.subplot(grid[1:, 1])
        # tableax1 = plt.subplot(grid[0, 1])
        colors = assignColor(allroots, nreal)
        for i in range(len(segs)):
            gains = allgains[i]
            toas = alltoas[i]
            for j in range(len(gains)):
                ax1.plot(segs[i]/GHz, gains[j], color = colors[i][j])
                ax2.plot(segs[i]/GHz, toas[j], color = colors[i][j])
        ax1.set_yscale('log')
        ax1.plot([-1, 10], [1, 1], ls='dashed', color='black', scalex = False)
        ax1.set_xlim([fmin/GHz, fmax/GHz])
        ax1.set_ylabel(r'$G_j$', fontsize = 20)
        ax2.set_ylabel(r'$\Delta t_{j}$ ($\mu s$)', fontsize = 20)
        ax2.plot([-1, 10], [0, 0], ls='dashed', color='black')
        ax2.set_yscale('symlog')
        axes = [ax1, ax2, ax3]
        for i in range(len(fcross)):
            ax1.plot([fcross[i]/GHz, fcross[i]/GHz], [1e-9, 1e4], color = 'black', ls = ':', scaley = False, scalex = False)
            for j in range(1, 3):
                axes[j].plot([fcross[i]/GHz, fcross[i]/GHz], [-1000, 1000], color = 'black', ls = ':', scaley = False, scalex = False)
        ax3.scatter(fvecav, toapert, color='black', s = 1.5)
        ax3.set_ylabel(r'$\Delta t_{tot}$ ($\mu s$)', fontsize = 20)
        ax3.set_xlabel(r'$\nu$ (GHz)', fontsize = 20)
        ax3.plot([-1, 10], [0, 0], ls='dashed', color='black', marker = '.')
        ax3.set_ylim(np.min(toapert) - 1, np.max(toapert) + 1)
        ax3.tick_params(labelsize = 16)
        ax2.tick_params(labelsize = 16)
        ax1.tick_params(labelsize = 16)
        
        # Tables
        # clabels = ['Band', r'$\overline{\Delta t}_{comb}$ ($\mu s$)']
        # tvals = [['820 MHz', np.around(av1, 4)], ['1400 MHz', np.around(av2, 4)], ['2300 MHz', np.around(av3, 4)]]
        # tableax1.axis('tight')
        # tableax1.axis('off')
        # table1 = tableax1.table(cellText = tvals, colWidths = [0.25, 0.25], colLabels = clabels, loc = 'center')
        # table1.auto_set_font_size(False)
        # table1.set_fontsize(12)
        # table1.scale(3., 3.)
        
        
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
        
        plt.tight_layout()
        plt.show()
        
    return np.array([fvec, toapert])

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
        
