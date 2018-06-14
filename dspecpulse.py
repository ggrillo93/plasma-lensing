from fundfunctions import *
from solvers import *
from observables import *
from scipy import ndimage as ndim
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

def pulsedynspec(dso, dsl, fmin, fmax, dm, upvec, period, ax, ay, template = None, spacing = 1e5, noise = 0.2, tsize = 2048, chw = 1.5e6):
    
    taxis = np.linspace(-period/2., period/2., tsize)
    if template ==  None:
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
        
    cdist = 1e4 # set minimum caustic distance

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
    
    singleim = np.argwhere(nreal == 1).flatten()
    multiim = np.argwhere(nreal > 1).flatten()
    
    nfpts = [len(seg) for seg in segs]
    totfpts = np.sum(nfpts)
    dspec = np.zeros([totfpts, tsize], dtype = complex)
    
    count = 0
    for i in range(nzones):
        fvec = segs[i]
        nf = nfpts[i]
        nroots = nreal[i]
        roots = allroots[i]
        if nroots == 1: # single image
            for j in range(nf):
                fieldcomp = GOfield(roots[j].real[0], rF2[i][j], lc[i][j], ax, ay)
                toa = deltat(roots[j].real[0], tg0, tdm0[i][j], alp[i][j], ax, ay)*1e-3
                noisytemp = template + noise*np.random.uniform(-1., 1., tsize)
                orpulse = pp.SinglePulse(noisytemp)
                shiftp = orpulse.shiftit(toa/dt)
                dspec[count] = fieldcomp[0]*shiftp*np.exp(1j*(fieldcomp[1] + fieldcomp[2]))
                count = count + 1
        else: # multiple images
            fields = np.zeros([nroots, 3, nf], dtype = complex)
            toas = np.zeros([nroots, nf])
            for j in range(nf):
                for k in range(nroots):
                    field = GOfield(roots[j][k], rF2[i][j], lc[i][j], ax, ay)
                    toas[k][j] = deltat(roots[j][k].real, tg0, tdm0[i][j], alp[i][j], ax, ay)*1e-3
                    for l in range(3):
                        fields[k][l][j] = field[l]
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
                        noisytemp = template + noise*np.random.uniform(-1., 1., tsize)
                        orpulse = pp.SinglePulse(noisytemp)
                        pulse = regfields[l][k]*orpulse.shiftit(regtoas[l][k]/dt)
                        tpulse = tpulse + pulse
                    dspec[count] = tpulse
                    count = count + 1
    
    fvec = np.hstack(segs)
    wind = int(chw/spacing)
    dspecav = np.abs(groupedAvg(dspec, N = wind))**2
    fvec = groupedAvg(fvec, N = wind)/GHz
    
    plt.close()
    
    band1 = (fvec > 0.73) * (fvec < 0.91)
    band2 = (fvec > 1.15) * (fvec < 1.88)
    band3 = (fvec > 2.05) * (fvec < 2.4)
    
    fig = plt.figure(figsize = (10, 8), dpi = 100)
    grid = gs.GridSpec(3, 3)
    ax0 = plt.subplot(grid[:, 0])
    ax1 = fig.add_subplot(grid[2, 1])
    ax2 = fig.add_subplot(grid[1, 1], sharex = ax1)
    ax3 = fig.add_subplot(grid[0, 1], sharex = ax1)
    ax4 = fig.add_subplot(grid[0, 2])
    ax5 = fig.add_subplot(grid[1:, 2])
    plt.setp(ax2.get_xticklabels(), visible = False)
    plt.setp(ax3.get_xticklabels(), visible = False)
    xlim = period/2.
    # spec = np.mean(template**2)
    spec = 1
    
    # All
    avpulse = np.mean(dspecav, 0)
    im0 = ax0.imshow(dspecav, origin = 'lower', aspect = 'auto', extent = (-xlim, xlim, min(fvec), max(fvec)), cmap = 'Greys')
    ax0.set_ylabel(r'$\nu$ (GHz)', fontsize = 16)
    ax0.set_xlabel(r'$\Delta t$ (ms)', fontsize = 16)
    ax0.tick_params(labelsize = 12)
    div0 = make_axes_locatable(ax0)
    ax01 = div0.append_axes("top", size = "20%", sharex = ax0)
    ax01.plot(taxis, avpulse, color = 'black')
    ax01.plot([0, 0], [-1, 10], ls = 'dashed', color = 'blue', scalex = False, scaley = False)
    ax01.tick_params(labelsize = 12)
    ax02 = div0.append_axes("right", size = "40%", sharey = ax0)
    ax02.plot(np.mean(dspecav, 1)/spec, fvec, color = 'black')
    ax02.xaxis.tick_top()
    ax02.xaxis.set_label_position('top')
    ax02.set_ylim([min(fvec), max(fvec)])
    ax01.set_ylabel(r'$\overline{G(t)}$', fontsize = 12)
    ax02.set_xlabel(r'$\overline{G(\nu)}$', fontsize = 12)
    plt.setp(ax01.get_xticklabels(), visible=False)
    plt.setp(ax02.get_yticklabels(), visible=False)
    ax0.yaxis.set_major_locator(MaxNLocator(nbins=len(ax0.get_yticklabels()), prune='upper'))
    ax02.xaxis.set_major_locator(MaxNLocator(nbins=len(ax02.get_yticklabels()), prune='lower'))
    ax02.set_xlim(left = 0)
    # plt.setp(ax02.get_xticklabels()[:-2], visible=False)
    
    # Band 1: 0.73 - 0.91 GHz
    dspec1 = dspecav[:][band1]
    fvecb1 = fvec[band1]
    avpulse1 = np.mean(dspec1, 0)
    im1 = ax1.imshow(dspec1, origin = 'lower', aspect = 'auto', extent = (-xlim, xlim, 0.73, 0.91), cmap = 'Greys')
    ax1.set_ylabel(r'$\nu$ (GHz)', fontsize = 16)
    ax1.set_xlabel(r'$\Delta t$ (ms)', fontsize=16)
    ax1.tick_params(labelsize = 12)
    div1 = make_axes_locatable(ax1)
    ax11 = div1.append_axes("top", size = "20%", sharex = ax1)
    ax11.set_ylabel(r'$\overline{G(t)}$', fontsize = 12)
    ax11.plot(taxis, avpulse1, color = 'black')
    ax11.plot([0, 0], [-1, 10], ls = 'dashed', color = 'blue', scalex = False, scaley = False)
    ax11.tick_params(labelsize = 12)
    ax12 = div1.append_axes("right", size="40%", sharey = ax1)
    ax12.plot(np.mean(dspec1, 1)/spec, fvecb1, color = 'black')
    ax12.xaxis.tick_top()
    ax12.xaxis.set_label_position('top')
    ax12.set_xlabel(r'$\overline{G(\nu)}$', fontsize = 12)
    ax12.set_ylim([0.73, 0.91])
    plt.setp(ax11.get_xticklabels(), visible=False)
    plt.setp(ax12.get_yticklabels(), visible=False)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins = len(ax1.get_yticklabels()), prune = 'upper'))
    ax12.xaxis.set_major_locator(MaxNLocator(nbins=len(ax12.get_yticklabels()), prune='lower'))
    ax12.set_xlim(left = 0)
    # plt.setp(ax12.get_xticklabels()[:-2], visible=False)
    
    # Band 2: 1.15 - 1.88 GHz
    dspec2 = dspecav[:][band2]
    fvecb2 = fvec[band2]
    avpulse2 = np.mean(dspec2, 0)
    im2 = ax2.imshow(dspec2, origin = 'lower', aspect = 'auto', extent = (-xlim, xlim, 1.15, 1.88), cmap = 'Greys')
    ax2.set_ylabel(r'$\nu$ (GHz)', fontsize = 16)
    ax2.tick_params(labelsize = 12)
    div2 = make_axes_locatable(ax2)
    ax21 = div2.append_axes("top", size = "20%", sharex = ax2)
    ax21.plot(taxis, avpulse2, color = 'black')
    ax21.plot([0, 0], [-1, 10], ls = 'dashed', color = 'blue', scalex = False, scaley = False)
    ax21.tick_params(labelsize = 12)
    ax22 = div2.append_axes("right", size="40%", sharey = ax2)
    ax22.plot(np.mean(dspec2, 1)/spec, fvecb2, color = 'black')
    ax22.xaxis.tick_top()
    ax22.xaxis.set_label_position('top')
    ax22.set_ylim([1.15, 1.88])
    ax21.set_ylabel(r'$\overline{G(t)}$', fontsize=12)
    ax22.set_xlabel(r'$\overline{G(\nu)}$', fontsize=12)
    plt.setp(ax21.get_xticklabels(), visible=False)
    plt.setp(ax22.get_yticklabels(), visible=False)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins = len(ax2.get_yticklabels()), prune = 'upper'))
    ax22.xaxis.set_major_locator(MaxNLocator(nbins=len(ax22.get_yticklabels()), prune='lower'))
    ax22.set_xlim(left = 0)
    # plt.setp(ax22.get_xticklabels()[:-2], visible=False)
    
    # Band 3: 2.05 - 2.40 GHz
    dspec3 = dspecav[:][band3]
    fvecb3 = fvec[band3]
    avpulse3 = np.mean(dspec3, 0)
    im3 = ax3.imshow(dspec3, origin = 'lower', aspect = 'auto', extent = (-xlim, xlim, 2.05, 2.4), cmap = 'Greys')
    ax3.set_ylabel(r'$\nu$ (GHz)', fontsize = 16)
    ax3.tick_params(labelsize = 12)
    div3 = make_axes_locatable(ax3)
    ax31 = div3.append_axes("top", size = "20%", sharex = ax3)
    ax31.plot(taxis, avpulse3, color = 'black')
    ax31.plot([0, 0], [-1, 10], ls = 'dashed', color = 'blue', scalex = False, scaley = False)
    ax31.tick_params(labelsize = 12)
    ax32 = div3.append_axes("right", size="40%", sharey = ax3)
    ax32.plot(np.mean(dspec3, 1)/spec, fvecb3, color = 'black')
    ax32.xaxis.tick_top()
    ax32.xaxis.set_label_position('top')
    ax32.set_ylim([2.05, 2.4])
    ax31.set_ylabel(r'$\overline{G(t)}$', fontsize=12)
    ax32.set_xlabel(r'$\overline{G(\nu)}$', fontsize=12)
    plt.setp(ax31.get_xticklabels(), visible=False)
    plt.setp(ax32.get_yticklabels(), visible=False)
    ax3.yaxis.set_major_locator(MaxNLocator(nbins = len(ax3.get_yticklabels()), prune = 'upper'))
    ax32.xaxis.set_major_locator(MaxNLocator(nbins=len(ax32.get_yticklabels()), prune='lower'))
    ax32.set_xlim(left = 0)
    # plt.setp(ax32.get_xticklabels()[:-2], visible=False)
    
    fig.canvas.draw()
    plt.setp(sideticks(ax02), visible = False)
    plt.setp(sideticks(ax12), visible = False)
    plt.setp(sideticks(ax22), visible = False)
    plt.setp(sideticks(ax32), visible = False)
    
    toapert0 = tempmatch(avpulse, template**2, dt)[0]*1e3
    toapert1 = tempmatch(avpulse1, template**2, dt)[0]*1e3
    toapert2 = tempmatch(avpulse2, template**2, dt)[0]*1e3
    toapert3 = tempmatch(avpulse3, template**2, dt)[0]*1e3
    
    ax4.axis('off')
    ax4.axis('tight')
    col_labels1 = (['Band', r'$\Delta t$ ($\mu$s)'])
    table1vals = [['All', np.around(toapert0, 3)], ['0.82 GHz', np.around(toapert1, 3)], ['1.4 GHz', np.around(toapert2, 3)], ['2.3 GHz', np.around(toapert3, 3)]]
    table1 = ax4.table(cellText = table1vals, colWidths = [0.35, 0.25], colLabels = col_labels1, loc = 'center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(2., 2.)
    
    col_labels2 = ['Parameter', 'Value']
    if np.abs(dm/pctocm) < 1:
        dmlabel = "{:.2E}".format(Decimal(dm/pctocm))
    else:
        dmlabel = str(dm/pctocm)
    table2vals = [[r'$d_{so} \: (kpc)$', np.around(dso/pctocm/kpc, 2)], [r'$d_{sl} \: (kpc)$', np.around(dsl/pctocm/kpc, 2)], [r'$a_x \: (AU)$', np.around(ax/autocm, 2)], [r'$a_y \: (AU)$', np.around(ay/autocm, 2)], [r'$DM_l \: (pc \, cm^{-3})$', dmlabel], [r"$\vec{u}'$", np.around(upvec, 2)], [r'Sampling (MHz)', np.around(spacing/1e6, 3)], ['Ch. W. (MHz)', np.around(chw/1e6, 3)]]
    ax5.axis('tight')
    ax5.axis('off')
    # ax2.set_anchor('N')
    table2 = ax5.table(cellText = table2vals, colWidths = [0.35, 0.25], colLabels = col_labels2, loc = 'center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(2., 2.)
    
    plt.tight_layout(pad = 1.75)
    plt.show()
    return

def sideticks(ax):
    ticks = ax.get_xticklabels()
    if ticks[-1].get_text() == '':
        return ticks[:-2]
    else:
        return ticks[:-1]
