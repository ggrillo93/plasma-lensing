from fundfunctions import *
from upslice import *
from joblib import Parallel, delayed

def dmcalc(f, lc):
    return -f*lc/(c*re)

def planeSliceGFig2(uxmax, uymax, rF2, lc, ax, ay, m, n, npoints = 3000, gsizex = 2048, gsizey = 2048, comp = True):
    """ Plots gain for slice across the u'-plane for given lens parameters, observation frequency, uxmax, slope m and offset n. Compares it to the gain given by solving the Kirchhoff diffraction integral using convolution. Plots the slice gain and the entire u' plane gain. """

    # Calculate coefficients
    alp  = rF2*lc
    coeff = alp*np.array([1./ax**2, 1./ay**2])
    uF2x, uF2y = rF2*np.array([1./ax**2, 1./ay**2])

    # Calculate caustic intersections
    ucross = polishedRoots(causticEqSlice, uxmax, uymax, args = (alp, m, n, ax, ay))
    ncross = len(ucross)
    upcross = mapToUp(ucross.T, alp, ax, ay)
    p = np.argsort(upcross[0])
    upcross = upcross.T[p]
    ucross = ucross[p]
    print(upcross)
    print(ucross)

    # Calculate sign of second derivative at caustics
    sigs = np.zeros(ncross)
    for i in range(ncross):
        sigs[i] = np.sign(ax**2/rF2 + lc*(lensh(*[ucross[i][0], ucross[i][1]])[0]))
    print(sigs)

    # Set up quantities for proper u' plane slicing
    ymin = -m*uxmax + n
    ymax = m*uxmax + n
    if ymin < -uymax:
        xmin = (-uymax - n)/m
        ymin = m*xmin + n
    else:
        xmin = -uxmax
    if ymax > uymax:
        xmax = (uymax - n)/m
        ymax = m*xmax + n
    else:
        xmax = uxmax
    xx = np.linspace(gridToPixel(xmin, uxmax, gsizex/2), gridToPixel(xmax, uxmax, gsizex/2) - 1, gsizex)
    yy = np.linspace(gridToPixel(ymin, uymax, gsizey/2), gridToPixel(ymax, uymax, gsizey/2) - 1, gsizey)

    cdist = uxmax/(np.abs(100*lc))
    print(cdist)

    bound = np.insert(upcross, 0, np.array([[xmin, ymin]]), axis = 0) # set up boundaries
    bound = np.append(bound, np.array([[xmax, ymax]]), axis = 0)
    midpoints = [(bound[i] + bound[i+1])/2. for i in range(len(bound) - 1)] # find middle point between boundaries
    nzones = len(midpoints)
    nreal = np.zeros(nzones, dtype = int)
    print(nzones)
    for i in range(nzones): # find number of roots at each midpoint
        mpoint = midpoints[i]
        nreal[i] = int(len(findRoots(lensEq, 2*uxmax, 2*uymax, args = (mpoint, coeff), N = 1000)))
    upxvecs = np.array([np.linspace(bound[i-1][0] + cdist, bound[i][0] - cdist, npoints) for i in range(1, ncross + 2)]) # generate upx vector
    segs = np.asarray([lineVert(upx, m, n) for upx in upxvecs]) # generate slice across plane
    if comp == True:
        diff = difference(nreal)  # determine number of complex solutions
        ncomplex = np.ones(nzones)*100
        for i in range(nzones):
            if diff[i] == 0 or diff[i] == -2:
                ncomplex[i] = 1
            elif diff[i] == -4:
                ncomplex[i] = 2
            elif diff[i] == 4:
                ncomplex[i] = 0
    else:
        ncomplex = np.zeros(nzones)
        
    print(nreal)
    print(ncomplex)

    # Solve lens equation at each coordinate
    allroots = rootFinder(segs, nreal, ncomplex, npoints, ucross, uxmax, uymax, coeff)
    
    # Calculate fields
    allfields = []
    for i in range(nzones):
        fields = obsCalc(GOfield, allroots[i], len(allroots[i][0]), npoints, 1, args=(rF2, lc, ax, ay))
        allfields.append(fields)
    
    fogain = np.zeros([nzones, npoints])
    zogain = np.zeros([nzones, npoints])
    for i in range(nzones):
        nroots = nreal[i]
        if nroots == 1:
            fogain[i] = np.abs(allfields[i])**2
            zogain[i] = np.abs(allfields[i])**2
        else:
            fogain[i] = np.abs(np.sum(allfields[i], axis = 0))**2
            zog = 0
            for j in range(nroots):
                zog = zog + np.abs(allfields[i][j])**2
            zogain[i] = zog
            
    fogain = fogain.flatten()
    zogain = zogain.flatten()

    # Construct uniform asymptotics
    # asymp = uniAsymp(allroots, allfields, nreal, ncomplex, npoints, nzones, sigs)
    # interp = UnivariateSpline(upxvecs.flatten(), asymp, s = 0)
    # finx = np.linspace(xmin, xmax, 4*npoints)
    # asymG = interp(finx)

    # KDI
    rx = np.linspace(-2*uxmax, 2*uxmax, gsizex)
    ry = np.linspace(-2*uymax, 2*uymax, gsizey)
    dux = 4*uxmax/gsizex
    duy = 4*uymax/gsizey
    extent = (-uxmax, uxmax, -uymax, uymax)
    ux, uy = np.meshgrid(rx, ry)
    lens = lensPhase(ux, uy, lc)
    lensfft = fft2(lens)
    geo = geoPhase(ux, uy, uF2x, uF2y)
    geofft = fft2(geo)
    fieldfft = lensfft*geofft
    field = fftshift(ifft2(fieldfft))
    soln = np.abs((dux*duy*field)**2/(4*pi**2*uF2x*uF2y))
    soln = soln[int(0.25*gsizex):int(0.75*gsizex), int(0.25*gsizey):int(0.75*gsizey)]

    # Plots
    fig = plt.figure(figsize = (15, 7.5))
    grid = gs.GridSpec(2, 2)
    # tableax = plt.subplot(grid[1, :])
    # tableax2 = plt.subplot(grid[2, :])
    ax0, ax1 = plt.subplot(grid[:, 0]), plt.subplot(grid[0, 1])
    ax2 = plt.subplot(grid[1, 1], sharex=ax1)

    rx = np.linspace(-uxmax, uxmax, gsizex)
    ry = np.linspace(-uymax, uymax, gsizey)
    ux, uy = np.meshgrid(rx, ry)

    rx2 = np.linspace(xmin, xmax, gsizex)
    im0 = ax0.imshow(soln, origin = 'lower', extent = extent, aspect = 'auto') # Plot entire screen
    cbar = fig.colorbar(im0, ax = ax0)
    cbar.set_label('G', fontsize = 16)
    ucaus = causCurve([ux, uy], lc*np.array([uF2x, uF2y]))
    cs = plt.contour(np.linspace(-uxmax, uxmax, gsizex), ry, ucaus, levels = [0, np.inf], linewidths = 0)
    paths = cs.collections[0].get_paths()
    uppaths = []
    for p in paths:
        cuvert = np.array(p.vertices).T
        upx, upy = mapToUp(cuvert, alp, ax, ay)
        ax0.plot(upx, upy, color = 'white') # Plot caustic curves
    ax0.scatter(upcross.T[0], upcross.T[1], color = 'white')
    ax0.plot(rx2, rx2*m + n, color = 'white') # Plot observer motion
    ax0.set_xlabel(r"$u'_x$", fontsize = 16)
    ax0.set_ylim([-uymax, uymax])
    ax0.set_xlim([-uxmax, uxmax])
    ax0.set_ylabel(r"$u'_y$", fontsize = 16)
    # ax0.set_title("Gain in the u' plane")

    G = map_coordinates(soln.T, np.vstack((xx, yy))) # Plot gain along observer motion
    G = G - G[-1] + 1
    ax1.plot(rx2, G, color = 'blue', label = "Gain from FFT", linewidth = 1.)
    for caus in upcross.T[0]:
        ax1.plot([caus, caus], [-10, 1000], ls = 'dashed', color = 'black')
    xaxis = upxvecs.flatten()
    ax1.plot(xaxis, zogain, color = 'red', label = "Gain from zeroth order GO")
    ax1.set_ylim(-cdist, np.max(G) + 1.)
    ax1.set_xlim(np.min(rx2), np.max(rx2))
    # ax1.set_xlabel(r"$u'_x$")
    ax1.set_ylabel('G', fontsize = 16)
    ax1.legend(loc = 1)
    # ax1.set_title("Slice Gain")
    ax1.grid()
    
    # Plot gain along observer motion
    ax2.plot(rx2, G, color='blue', label="Gain from FFT", linewidth=1.)
    for caus in upcross.T[0]:
        ax2.plot([caus, caus], [-10, 1000], ls='dashed', color='black')
    ax2.plot(xaxis, fogain, color='orange', label="Gain from first order GO")
    ax2.set_ylim(-cdist, np.max(G) + 1.)
    ax2.set_xlim(np.min(rx2), np.max(rx2))
    ax2.set_xlabel(r"$u'_x$", fontsize = 16)
    ax2.set_ylabel('G', fontsize = 16)
    ax2.legend(loc = 1)
    # ax1.set_title("Slice Gain")
    ax2.grid()
    grid.tight_layout(fig)

    # col_labels = ['Parameter', 'Value'] # Create table with parameter values
    # if np.abs(dm/pctocm) < 1:
    #     dmlabel = "{:.2E}".format(Decimal(dm/pctocm))
    # else:
    #     dmlabel = str(dm/pctocm)
    # tablevals = [[r'$d_{so} \: (kpc)$', np.around(dso/pctocm/kpc, 2)], [r'$d_{sl} \: (kpc)$', np.around(dsl/pctocm/kpc, 3)], [r'$a_x \: (AU)$', np.around(ax/autocm, 3)], [r'$a_y \: (AU)$', np.around(ay/autocm, 3)], [r'$DM_l \: (pc \, cm^{-3})$', dmlabel], [r"$\nu$ (GHz)", f/GHz], ['Slope', np.around(m, 2)], ['Offset', n]]
    # tableax.axis('tight')
    # tableax.axis('off')
    # table = tableax.table(cellText = np.asarray(tablevals).T, colWidths = np.ones(8)*0.045, rowLabels = col_labels, loc = 'center')
    # table.auto_set_font_size(False)
    # table.set_fontsize(11)
    # table.scale(2.5, 2.5)
    
    # row_label =  ['Lens shape']
    # val = [['$%s$' % sym.latex(lensf)]]
    # tableax2.axis('tight')
    # tableax2.axis('off')
    # table2 = tableax2.table(cellText=val, colWidths=[0.0015*len(sym.latex(lensf))], rowLabels=row_label, loc='top')
    # table2.auto_set_font_size(False)
    # table2.set_fontsize(12)
    # table2.scale(2.5, 2.5)

    plt.show()
    return

def planeSliceGFig3(uxmax, uymax, rF2, lc, ax, ay, m, n, npoints = 3000, gsizex = 2048, gsizey = 2048, comp = True):
    """ Plots gain for slice across the u'-plane for given lens parameters, observation frequency, uxmax, slope m and offset n. Compares it to the gain given by solving the Kirchhoff diffraction integral using convolution. Plots the slice gain and the entire u' plane gain. """

    # Calculate coefficients
    alp  = rF2*lc
    coeff = alp*np.array([1./ax**2, 1./ay**2])
    uF2x, uF2y = rF2*np.array([1./ax**2, 1./ay**2])

    # Calculate caustic intersections
    ucross = polishedRoots(causticEqSlice, uxmax, uymax, args = (alp, m, n, ax, ay))
    ncross = len(ucross)
    upcross = mapToUp(ucross.T, alp, ax, ay)
    p = np.argsort(upcross[0])
    upcross = upcross.T[p]
    ucross = ucross[p]
    print(upcross)
    print(ucross)

    # Calculate sign of second derivative at caustics
    sigs = np.zeros(ncross)
    for i in range(ncross):
        sigs[i] = np.sign(ax**2/rF2 + lc*(lensh(*[ucross[i][0], ucross[i][1]])[0]))
    print(sigs)

    # Set up quantities for proper u' plane slicing
    ymin = -m*uxmax + n
    ymax = m*uxmax + n
    if ymin < -uymax:
        xmin = (-uymax - n)/m
        ymin = m*xmin + n
    else:
        xmin = -uxmax
    if ymax > uymax:
        xmax = (uymax - n)/m
        ymax = m*xmax + n
    else:
        xmax = uxmax
    xx = np.linspace(gridToPixel(xmin, uxmax, gsizex/2), gridToPixel(xmax, uxmax, gsizex/2) - 1, gsizex)
    yy = np.linspace(gridToPixel(ymin, uymax, gsizey/2), gridToPixel(ymax, uymax, gsizey/2) - 1, gsizey)

    cdist = uxmax/(np.abs(50*lc))
    print(cdist)

    bound = np.insert(upcross, 0, np.array([[xmin, ymin]]), axis = 0) # set up boundaries
    bound = np.append(bound, np.array([[xmax, ymax]]), axis = 0)
    midpoints = [(bound[i] + bound[i+1])/2. for i in range(len(bound) - 1)] # find middle point between boundaries
    nzones = len(midpoints)
    nreal = np.zeros(nzones)
    print(nzones)
    for i in range(nzones): # find number of roots at each midpoint
        mpoint = midpoints[i]
        nreal[i] = len(findRoots(lensEq, 2*uxmax, 2*uymax, args = (mpoint, coeff), N = 1000))
    upxvecs = np.array([np.linspace(bound[i-1][0] + cdist, bound[i][0] - cdist, npoints) for i in range(1, ncross + 2)]) # generate upx vector
    segs = np.asarray([lineVert(upx, m, n) for upx in upxvecs]) # generate slice across plane
    diff = difference(nreal) # determine number of complex solutions
    if comp == True:
        ncomplex = np.ones(nzones)*100
        for i in range(nzones):
            if diff[i] == 0 or diff[i] == -2:
                ncomplex[i] = 1
            elif diff[i] == -4:
                ncomplex[i] = 2
            elif diff[i] == 4:
                ncomplex[i] = 0
    else:
        ncomplex = np.zeros(nzones)
        
    print(nreal)
    print(ncomplex)

    # Solve lens equation at each coordinate
    allroots = rootFinder(segs, nreal, ncomplex, npoints, ucross, uxmax, uymax, coeff)
    
    # Calculate fields
    allfields = []
    for i in range(nzones):
        fields = obsCalc(GOfield, allroots[i], len(allroots[i][0]), npoints, 3, args=(rF2, lc, ax, ay))
        allfields.append(fields)

    # Construct uniform asymptotics
    asymp = uniAsymp(allroots, allfields, nreal, ncomplex, npoints, nzones, sigs)
    interp = UnivariateSpline(upxvecs.flatten(), asymp, s = 0)
    finx = np.linspace(xmin, xmax, 4*npoints)
    asymG = interp(finx)

    # KDI
    rx = np.linspace(-2*uxmax, 2*uxmax, gsizex)
    ry = np.linspace(-2*uymax, 2*uymax, gsizey)
    dux = 4*uxmax/gsizex
    duy = 4*uymax/gsizey
    extent = (-uxmax, uxmax, -uymax, uymax)
    ux, uy = np.meshgrid(rx, ry)
    lens = lensPhase(ux, uy, lc)
    lensfft = fft2(lens)
    geo = geoPhase(ux, uy, uF2x, uF2y)
    geofft = fft2(geo)
    fieldfft = lensfft*geofft
    field = fftshift(ifft2(fieldfft))
    soln = np.abs((dux*duy*field)**2/(4*pi**2*uF2x*uF2y))
    soln = soln[int(0.25*gsizex):int(0.75*gsizex), int(0.25*gsizey):int(0.75*gsizey)]

    # Plots
    fig = plt.figure(figsize = (20, 7.5))
    grid = gs.GridSpec(1, 2)
    # tableax = plt.subplot(grid[1, :])
    # tableax2 = plt.subplot(grid[2, :])
    ax0, ax1 = plt.subplot(grid[0, 0]), plt.subplot(grid[0, 1])

    rx = np.linspace(-uxmax, uxmax, gsizex)
    ry = np.linspace(-uymax, uymax, gsizey)
    ux, uy = np.meshgrid(rx, ry)

    rx2 = np.linspace(xmin, xmax, gsizex)
    im0 = ax0.imshow(soln, origin = 'lower', extent = extent, aspect = 'auto') # Plot entire screen
    cbar = fig.colorbar(im0, ax = ax0)
    # cbar.set_label(r'$\log{G}$', fontsize = 16)
    cbar.set_label('G', fontsize=16)
    ucaus = causCurve([ux, uy], lc*np.array([uF2x, uF2y]))
    cs = plt.contour(np.linspace(-uxmax, uxmax, gsizex), ry, ucaus, levels = [0, np.inf], linewidths = 0)
    paths = cs.collections[0].get_paths()
    uppaths = []
    for p in paths:
        cuvert = np.array(p.vertices).T
        upx, upy = mapToUp(cuvert, alp, ax, ay)
        ax0.plot(upx, upy, color = 'white') # Plot caustic curves
    ax0.scatter(upcross.T[0], upcross.T[1], color = 'white')
    ax0.plot(rx2, rx2*m + n, color = 'white') # Plot observer motion
    ax0.set_xlabel(r"$u'_x$", fontsize = 16)
    ax0.set_ylim([-uymax, uymax])
    ax0.set_xlim([-uxmax, uxmax])
    ax0.set_ylabel(r"$u'_y$", fontsize = 16)
    # ax0.set_title("Gain in the u' plane")

    G = map_coordinates(soln.T, np.vstack((xx, yy))) # Plot gain along observer motion
    G = G - G[-1] + 1
    ax1.plot(rx2, G, color = 'blue', label = "Gain from FFT")
    for caus in upcross.T[0]:
        ax1.plot([caus, caus], [-10, 1000], ls = 'dashed', color = 'black')
    ax1.plot(finx, asymG, color = 'red', label = "Gain from second order GO")
    ax1.set_ylim(-cdist, np.max(asymG) + 1.)
    ax1.set_xlim(np.min(rx2), np.max(rx2))
    ax1.set_xlabel(r"$u'_x$", fontsize = 16)
    ax1.set_ylabel('G', fontsize = 16)
    # ax1.set_title("Slice Gain")
    ax1.grid()
    ax1.legend(loc = 1)


    # col_labels = ['Parameter', 'Value'] # Create table with parameter values
    # if np.abs(dm/pctocm) < 1:
    #     dmlabel = "{:.2E}".format(Decimal(dm/pctocm))
    # else:
    #     dmlabel = str(dm/pctocm)
    # tablevals = [[r'$d_{so} \: (kpc)$', np.around(dso/pctocm/kpc, 2)], [r'$d_{sl} \: (kpc)$', np.around(dsl/pctocm/kpc, 3)], [r'$a_x \: (AU)$', np.around(ax/autocm, 3)], [r'$a_y \: (AU)$', np.around(ay/autocm, 3)], [r'$DM_l \: (pc \, cm^{-3})$', dmlabel], [r"$\nu$ (GHz)", f/GHz], ['Slope', np.around(m, 2)], ['Offset', n]]
    # tableax.axis('tight')
    # tableax.axis('off')
    # table = tableax.table(cellText = np.asarray(tablevals).T, colWidths = np.ones(8)*0.045, rowLabels = col_labels, loc = 'center')
    # table.auto_set_font_size(False)
    # table.set_fontsize(11)
    # table.scale(2.5, 2.5)
    
    # row_label =  ['Lens shape']
    # val = [['$%s$' % sym.latex(lensf)]]
    # tableax2.axis('tight')
    # tableax2.axis('off')
    # table2 = tableax2.table(cellText=val, colWidths=[0.0015*len(sym.latex(lensf))], rowLabels=row_label, loc='top')
    # table2.auto_set_font_size(False)
    # table2.set_fontsize(12)
    # table2.scale(2.5, 2.5)

    grid.tight_layout(fig, pad = 1.5)
    plt.show()
    return

def causDspectra(uxmax, uymax, ax, ay, dso, dsl, dm, m, n, N = 1000):
    """ Constructs caustic curves for path along the u' plane as a function of frequency. """
    
    ymin = -m*uxmax + n
    ymax = m*uxmax + n
    if ymin < -uymax:
        xmin = (-uymax - n)/m
        ymin = m*xmin + n
    else:
        xmin = -uxmax
    if ymax > uymax:
        xmax = (uymax - n)/m
        ymax = m*xmax + n
    else:
        xmax = uxmax
    
    dlo = dso - dsl
    coeff = dsl*dlo*re*dm/(2*pi*dso)
    
    rx = np.linspace(xmin - 5., xmax + 5., 500)
    ry = np.linspace(ymin - 5., ymax + 5., 500)
    uvec = np.meshgrid(rx, ry)
    A, B, C, D, E = causticFreqHelp(uvec, ax, ay, m, n)
    upxvec = np.linspace(xmin, xmax, N)
    freqcaus = []
    for upx in upxvec:
        eq1 = A*upx**2 + B*upx + C
        eq2 = D*upx + E
        evcaus = np.array([eq1, eq2])
        roots = polishedRootsBulk(evcaus, causEqFreq, rx, ry, args = (upx, ax, ay, m, n))
        for root in roots:
            ux, uy = root
            arg = coeff*lensg(ux, uy)[0]/(ux - upx)
            # print(arg)
            if arg > 0:
                freq = c*np.sqrt(arg)/(ax*GHz)
                if freq > 0.01:
                    freqcaus.append([upx, freq])
    # print(freqcaus)
    freqcaus = np.asarray(freqcaus).T
    plt.scatter(freqcaus[0], freqcaus[1], marker = '.', color = 'black', s = 3.)
    plt.xlim(xmin, xmax)
    plt.ylim(0., max(freqcaus[1]) + 0.5)
    plt.xlabel(r"$u'_x$", fontsize = 16)
    plt.ylabel(r'$\nu$ (GHz)', fontsize = 16)
    plt.grid()
    plt.show()
    return freqcaus
    
# # figure 2a
# dso, dsl, f, lc, ax, ay = 1.*kpc*pctocm, 0.5*kpc*pctocm, 0.8*GHz, -50., 0.015*autocm, 0.015*autocm
# dm = dmcalc(f, lc)
# print(dm/pctocm)
# rF2 = rFsqr(dso, dsl, f)
# alp = lc*rF2
# print(alp/autocm**2)
# print([alp/ax**2, alp/ay**2])
# lc = lensc(dm, f)
# print(lc)
# # causPlotter(5., 5., alp, ax, ay)
# planeSliceGEx(5.5, 5.5, rF2, lc, ax, ay, 1., 0., npoints=3000, gsizex=2048, gsizey=2048, comp=False)

# # figure 2b
# dso, dsl, f, lc, ax, ay = 1.*kpc*pctocm, 0.5*kpc*pctocm, 0.8*GHz, -250., 0.015*autocm*5**0.5, 0.015*autocm*5**0.5
# dm = dmcalc(f, lc)
# print(dm/pctocm)
# rF2 = rFsqr(dso, dsl, f)
# alp = lc*rF2
# print(alp/autocm**2)
# print([alp/ax**2, alp/ay**2])
# lc = lensc(dm, f)
# print(lc)
# planeSliceGEx(5.5, 5.5, rF2, lc, ax, ay, 1., 0., npoints=3000, gsizex=2*2048, gsizey=2*2048, comp=False)

# second order examples
# dso, dsl, f, lc, ax, ay = 1.*kpc*pctocm, 0.5*kpc*pctocm, 1.4*GHz, -30., 0.015*autocm, 0.025*autocm
# dm = dmcalc(f, lc)
# print(dm/pctocm)
# rF2 = rFsqr(dso, dsl, f)
# alp = lc*rF2
# print(alp/autocm**2)
# print([alp/ax**2, alp/ay**2])
# lc = lensc(dm, f)
# print(lc)
# # causPlotter(5., 5., alp, ax, ay, m= 0.2, n = 0)
# planeSliceGFig3(5., 5., rF2, lc, ax, ay, 0.3, 0., npoints=2000, gsizex=2*2048, gsizey=2*2048, comp=True)

# caustics in dynamic spectra
# dso, dsl, dm, ax, ay = 1.*kpc*pctocm, 0.5*kpc*pctocm, 1e-3*pctocm, 0.5*autocm, 1.*autocm
# lc = lensc(dm, 0.8*GHz)
# print(lc)
# # causDspectra(5., 5., ax, ay, dso, dsl, dm, 3., -1., N=750)
# off = [0., 1., 1.5, 2.]
# slopes = [1., 0.5, 0., 0.3]
# freqcaus = Parallel(n_jobs = 4)(delayed(causDspectra)(*[5., 5., ax, ay, dso, dsl, dm, slopes[i], off[i]]) for i in range(4))
# colors = ['blue', 'red', 'green', 'grey']
# plt.figure(figsize=(10, 10))
# for i in range(4):
#     plt.scatter(freqcaus[i][0], freqcaus[i][1], marker = '.', color = colors[i], s = 4.)
# plt.xlim(-5., 5.)
# # plt.ylim(0., max(freqcaus[1]) + 0.5)
# plt.xlabel(r"$u'_x$", fontsize = 16)
# plt.ylim(0., 2.)
# plt.ylabel(r'$\nu$ (GHz)', fontsize = 16)
# plt.grid()
# plt.tight_layout()
# plt.show()

# dynamic spectrum of overdense rectangular gaussian
dso, dsl, dm, ax, ay, f = 1.*kpc*pctocm, 0.5*kpc*pctocm, 1e-4*pctocm, 0.1*autocm, 0.2*autocm, 0.8*GHz
lc = lensc(dm, f)
rF2 = rFsqr(dso, dsl, f)
alp = lc*rF2
print([alp/ax**2, alp/ay**2])
# causPlotter(5., 5., alp, ax, ay, m=0.5, n=2.)
causDspectra(5., 5., ax, ay, dso, dsl, dm, 0.5, 2.5, N = 250)