from fundfunctions import *
from observables import *
from solvers import *
from upslice import *

def fsliceG(upvec, fmin, fmax, dso, dsl, dm, ax, ay, spacing = 1e5, chw = 1e5, comp = True, plot = False):
    
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
    print(ucrossb)
    ncross = len(fcross)
    print(fcross/GHz)
    
    # Calculate sign of second derivative at caustics
    sigs = np.zeros(ncross)
    for i in range(ncross):
        rF2 = rFsqr(dso, dsl, fcross[i])
        lc = lensc(dm, fcross[i])
        sigs[i] = np.sign(ax**2/rF2 + lc*lensh(ucrossb[i][0], ucrossb[i][1])[0])
    print(sigs)
        
    cdist = 1e4

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
    segs = np.array([np.arange(bound[i-1] + cdist, bound[i] - cdist, spacing) for i in range(1, ncross + 2)])
    if comp == True:
        diff = difference(nreal)
        print(diff)
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
    print(nreal)
    print(ncomplex)
    
    # Solve lens equation at each coordinate
    allroots = rootFinderFreq(segs, nreal, ncomplex, ucrossb, upvec, coeff)
    
    # Calculate fields
    allfields = []
    for l in range(nzones):
        nroots = len(allroots[l][0])
        fvec = segs[l]
        roots = allroots[l]
        npoints = len(fvec)
        fields = np.zeros([nroots, 3, npoints], dtype = complex)
        for i in range(npoints):
            freq = fvec[i]
            rF2 = rF2p/freq
            lc = lcp/freq
            for j in range(nroots):
                ans = GOfield(roots[i][j], rF2, lc, ax, ay)
                for k in range(3):
                    fields[j][k][i] = ans[k]
        # print(fields.shape)
        allfields.append(fields)
    
    # Calculate gain at caustics
    causgains = np.zeros(ncross)
    for i in range(ncross):
        freq = fcross[i]
        rF2 = rF2p/freq
        lc = lcp/freq
        causgains[i] = causAmp(ucrossb[i], rF2, lc, ax, ay)
    
    print(causgains)
    
    # Calculate first order gains
    allgains = []
    for l in range(nzones):
        fvec = segs[l]
        roots = allroots[l]
        nroots = int(nreal[l])
        npoints = len(fvec)
        gains = np.zeros(npoints)
        for i in range(npoints):
            freq = fvec[i]
            rF2 = rF2p/freq
            lc = lcp/freq
            tgain = 0
            for j in range(nroots):
                amp = GOAmp(roots[i][j], rF2, lc, ax, ay)
                tgain = tgain + amp**2
            gains[i] = tgain
        allgains.append(gains)
    
    # Construct uniform asymptotics
    asymp = uniAsymp(allroots, allfields, nreal, ncomplex, nzones, sigs)
    wind = int(chw/spacing)
    asympav = groupedAvg(asymp, N = wind)
    asymGav = np.abs(asympav)**2
    interp = UnivariateSpline(np.hstack(segs), np.abs(asymp)**2, s = 0)
    finf = np.linspace(fmin + cdist, fmax, len(asymp))
    finfav = groupedAvg(finf, N=wind)
    asymG = interp(finf)
    
    if plot:
        # Plots
        fig = plt.figure(figsize=(15, 10))
        grid = gs.GridSpec(1, 2, width_ratios=[5, 1])
        ax0 = plt.subplot(grid[0, 0])
        ax1 = plt.subplot(grid[0, 1])
        
        ax0.plot(finf/GHz, asymG, color = 'black')
        ax0.set_ylabel(r'$G$')
        ax0.set_xlabel(r"$\nu$ (GHz)")
        ax0.set_title('Lens shape: ' + '$%s$' % sym.latex(lensf))
        for freq in fcross/GHz:
            ax0.plot([freq, freq], [-10, 1000], ls='dashed', color='black')
        ax0.set_ylim(-0.1, np.max(asymG) + 1.)
        ax0.set_xlim(fmin/GHz, fmax/GHz)
        # ax0.set_yscale('log')
        ax0.grid()
        
        # Create table
        col_labels = ['Parameter', 'Value']
        if np.abs(dm/pctocm) < 1:
            dmlabel = "{:.2E}".format(Decimal(dm/pctocm))
        else:
            dmlabel = str(dm/pctocm)
        tablevals = [[r'$d_{so} \: (kpc)$', np.around(dso/pctocm/kpc, 2)], [r'$d_{sl} \: (kpc)$', np.around(dsl/pctocm/kpc, 2)], [r'$a_x \: (AU)$', np.around(ax/autocm, 2)], [r'$a_y \: (AU)$', np.around(ay/autocm, 2)], [r'$DM_l \: (pc \, cm^{-3})$', dmlabel], [r"$\vec{u}'$", np.around(upvec, 2)]]
        ax1.axis('tight')
        ax1.axis('off')
        ax1.set_anchor('N')
        table = ax1.table(cellText = tablevals, colWidths = [0.25, 0.25], colLabels = col_labels, loc = 'center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(3., 3.)
        
        plt.show()
    
    return np.array([finf/GHz, finfav/GHz, asymG, asymGav, fcross, np.hstack(allgains), np.hstack(segs)])
    
def fsliceGBulk(upvec, fcaus, fmin, fmax, leqinv, ax, ay, rx, ry, uvec, rF2p, lcp, coeff, cdist, npoints, comp, num, plot = False):
    
    print(upvec)
    ucross, fcross = fcaus
    # print(fcross/GHz)
    ncross = len(ucross)
    
    # Calculate sign of second derivative at caustics
    sigs = np.zeros(ncross)
    for i in range(ncross):
        rF2 = rF2p/fcross[i]
        lc = lcp/fcross[i]
        sigs[i] = np.sign(ax**2/rF2 + lc*lensh(ucross[i][0], ucross[i][1])[0])

    # Set up boundaries
    bound = np.insert(fcross, 0, fmin)
    bound = np.append(bound, fmax)
    midpoints = [(bound[i] + bound[i+1])/2. for i in range(len(bound) - 1)] # find middle point between boundaries
    nzones = len(midpoints)
    nreal = np.zeros(nzones)
    for i in range(nzones):
        mpoint = midpoints[i]
        leqinvtemp = leqinv/mpoint**2
        evleq = np.array([uvec[0] + leqinvtemp[0] - upvec[0], uvec[1] + leqinvtemp[1] - upvec[1]])
        nreal[i] = len(findRootsBulk(evleq, rx, ry))
    # print(nreal)
    segs = np.array([np.linspace(bound[i-1] + cdist, bound[i] - cdist, npoints) for i in range(1, ncross + 2)])
    if comp:
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
    
    # Solve lens equation at each coordinate
    allroots = rootFinderFreqBulk(segs, nreal, ncomplex, npoints, ucross, upvec, uvec, leqinv, rx, ry, coeff)
    
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
    
    # Construct uniform asymptotics
    asymp = uniAsymp(allroots, allfields, nreal, ncomplex, npoints, nzones, sigs)
    interp = interp1d(np.sort(segs.flatten()), asymp)
    finf = np.linspace(fmin + cdist, fmax - cdist, npoints)
    asymG = interp(finf)
    f_handle = file("dspectra" + str(num) + ".dat", 'a')
    np.savetxt(f_handle, np.array([asymG]))
    f_handle.close()
    f_handle2 = file("dspectra" + str(num + 1) + ".dat", "a")
    np.savetxt(f_handle2, np.array([upvec[0]]))
    f_handle2.close()
    
    if plot:
        # Plots
        fig = plt.figure(figsize=(15, 10))
        plt.plot(finf/GHz, asymG, color = 'black')
        plt.ylabel(r'$G$')
        plt.xlabel(r"$\nu$ (GHz)")
        plt.title('Lens shape: ' + '$%s$' % sym.latex(lensf))
        for freq in fcross/GHz:
            plt.plot([freq, freq], [-10, 1000], ls='dashed', color='black')
        plt.ylim(-0.1, np.max(asymG) + 1.)
        plt.xlim(fmin/GHz, fmax/GHz)
        plt.grid()
        plt.show()
        
    return
    
def fsliceGfull(upvec, uxmax, uymax, fmin, fmax, dso, dsl, dm, ax, ay, m, n, N=200, spacing = 1e5, chw = 1.5e6, comp = True):
        
    freqcaus = causCurveFreq(uxmax, uymax, ax, ay, dso, dsl, dm, m, n, plot = False, N = N)
    finf, finfav, asymG, asymGav, fcross, fogain, fvec = fsliceG(upvec, fmin, fmax, dso, dsl, dm, ax, ay, plot = False, spacing = spacing, comp = comp, chw = chw)
    
    plt.close()
    
    fig = plt.figure(figsize=(14, 8), dpi = 100)
    grid = gs.GridSpec(2, 2, width_ratios=[4, 1])
    ax2 = plt.subplot(grid[1, 1])
    ax1 = plt.subplot(grid[0, 1])
    ax0 = plt.subplot(grid[0, 0])
    ax3 = plt.subplot(grid[1, 0])
    
    ax0.plot(finf, asymG, color = 'black')
    ax0.set_ylabel(r'$G$')
    ax0.set_title('Lens shape: ' + '$%s$' % sym.latex(lensf))
    for freq in fcross/GHz:
        ax0.plot([freq, freq], [-10, 1000], ls='dashed', color='black')
    ax0.plot(fvec/GHz, fogain, color = 'red')
    ax0.set_ylim(-0.1, np.max(asymG) + 1.)
    ax0.set_xlim(min(finfav), max(finfav))
    ax0.grid()
    
    ax3.plot(finfav, asymGav, color = 'black')
    ax3.set_ylabel(r'$G$')
    ax3.set_xlabel(r"$\nu$ (GHz)")
    ax0.set_title('Lens shape: ' + '$%s$' % sym.latex(lensf))
    for freq in fcross/GHz:
        ax3.plot([freq, freq], [-10, 1000], ls='dashed', color='black')
    ax3.set_ylim(-0.1, np.max(asymGav) + 1.)
    ax3.set_xlim(min(finfav), max(finfav))
    ax3.grid()
    
    ax1.scatter(freqcaus[0], freqcaus[1], marker= '.', color = 'red')
    ax1.set_xlim(min(freqcaus[0]), max(freqcaus[0]))
    ax1.set_xlabel(r"$u'_x$")
    ax1.set_ylabel(r'$\nu$ (GHz)')
    ax1.grid()
    ax1.set_title("Caustic curves")
    ax1.set_aspect('auto', anchor='C')
    ax1.plot([upvec[0], upvec[0]], [-1, 10], color = 'black', lw = 2)
    ax1.set_ylim(fmin/GHz, fmax/GHz)

    # Create table
    col_labels = ['Parameter', 'Value']
    if np.abs(dm/pctocm) < 1:
        dmlabel = "{:.2E}".format(Decimal(dm/pctocm))
    else:
        dmlabel = str(dm/pctocm)
    tablevals = [[r'$d_{so} \: (kpc)$', np.around(dso/pctocm/kpc, 2)], [r'$d_{sl} \: (kpc)$', np.around(dsl/pctocm/kpc, 2)], [r'$a_x \: (AU)$', np.around(ax/autocm, 2)], [r'$a_y \: (AU)$', np.around(ay/autocm, 2)], [r'$DM_l \: (pc \, cm^{-3})$', dmlabel], [r"$\vec{u}'$", np.around(upvec, 2)], [r'Sampling (MHz)', np.around(spacing/1e6, 3)], ['Ch. W. (MHz)', np.around(chw/1e6, 3)]]
    ax2.axis('tight')
    ax2.axis('off')
    # ax2.set_anchor('N')
    table = ax2.table(cellText = tablevals, colWidths = [0.35, 0.25], colLabels = col_labels, loc = 'center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(2., 2.)
    
    grid.tight_layout(fig, h_pad = 2., pad = 2.)
    
    plt.show()
    return
        
