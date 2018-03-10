from fundfunctions import *
from observables import *
from solvers import *
from scipy.special import airy
from scipy.spatial import distance
from scipy.interpolate import *
from kdi import *

def fieldCalc(roots, npoints, rF2, lc, ax, ay):
    """ Calculates field for a list of roots of arbitrary dimensionality. """
    nroots = roots.shape[1]
    fields = np.zeros([nroots, npoints], dtype=complex)
    for i in range(npoints):
        for j in range(nroots):
            fields[j][i] = GOfieldA(roots[i][j], rF2, lc, ax, ay)
    return fields

def phaseCalc(roots, npoints, rF2, lc, ax, ay):
    """ Calculates phase for a list of roots of arbitrary dimensionality. """
    nroots = roots.shape[1]
    phis = np.zeros([nroots, npoints], dtype=complex)
    for i in range(npoints):
        for j in range(nroots):
            phis[j][i] = phi(roots[i][j], rF2, lc, ax, ay)
    return phis

def lineVert(upxvec, m, n):
    """ Returns list of line vertices. """
    return np.array([upxvec, m*upxvec + n]).T

def litAsymp(roots, phases, fields, sigs, rF2, lc, ax, ay):
    """ Constructs uniform asymptotic for bright region. """

    def helpAsympA(U1, U2, phi1, phi2, u2, sig):
        if phi1[0] > phi2[0]:
            pdiff = phi1 - phi2
            g1 = U2 - U1
        else:
            pdiff = phi2 - phi1
            g1 = U1 - U2
        chi = 0.5*(phi1 + phi2)
        xi = -(0.75*pdiff)**(2./3.)
        air = airy(xi)
        u1 = pi**0.5 *((U1 + U2)*(-xi)**0.25*air[0] - 1j*g1*(-xi)** -0.25*air[1]) * exp(1j*(chi + sig*0.25*pi))
        return np.abs(u1 + u2)**2

    def helpAsympB(sig, pos):
        npoints = len(roots)
        cphases = phases[pos]
        amp = np.zeros(npoints)
        for i in range(npoints):
            amp[i] = GOAmplitude(roots[i][pos], rF2, lc, ax, ay)
        xi = (1.5 * np.abs(cphases.imag))**(2. / 3.)
        u1 = 2*pi**0.5*amp * \
            (xi)**0.25 * airy(xi)[0] * exp(1j*(cphases.real - sig*0.25*pi))
        return u1

    nroots = roots.shape[1]
    if nroots == 3:  # just three real roots
        cond1 = np.abs(roots[0][0][0] - roots[0][1][0]) < np.abs(roots[0][1][0] - roots[0][2][0])
        cond2 = np.abs(roots[-1][0][0] - roots[-1][1][0]) < np.abs(roots[-1][1][0] - roots[-1][2][0])
        if cond1 and cond2:  # first root merges with second root at both ends
            litG = helpAsympA(np.abs(fields[0]), np.abs(fields[1]), phases[0], phases[1], fields[2], sigs[0])
        elif not cond1 and not cond2:  # second root merges with third root at both ends
            litG = helpAsympA(np.abs(fields[1]), np.abs(fields[2]), phases[1], phases[2], fields[0], sigs[0])
        else:  # need to split in two
            u1, u2, u3 = fields
            u11, u12 = np.split(u1, 2)
            U21, U22 = np.split(np.abs(u2), 2)
            u31, u32 = np.split(u3, 2)
            phi1, phi2, phi3 = phases
            phi11, phi12 = np.split(phi1, 2)
            phi21, phi22 = np.split(phi2, 2)
            phi31, phi32 = np.split(phi3, 2)
            if cond1 and not cond2:  # first root merges with second root at first end and second root merges with third root at other end
                litG1 = helpAsympA(np.abs(u11), U21, phi11, phi21, u31, sigs[0])
                litG2 = helpAsympA(U22, np.abs(u32), phi22, phi32, u12, sigs[0])
                litG = np.concatenate((litG1, litG2))
            else:  # second root merges with third root at first end and first root merges with second root at other end
                litG1 = helpAsympA(U21, np.abs( u31), phi21, phi31, u11, sigs[0])
                litG2 = helpAsympA(np.abs(u12), U22, phi12, phi22, u32, sigs[0])
                litG = np.concatenate((litG1, litG2))
    elif nroots == 4:  # three real roots and one complex root
        # second root merges with third root
        if np.abs(roots[0][1][0] - roots[0][2][0]) < np.abs(roots[0][0][0] - roots[0][1][0]):
            ucomp = helpAsympB(sigs[0], 3)
            u2 = np.sum([fields[0], ucomp], axis=0)
            litG = helpAsympA(np.abs(fields[1]), np.abs(fields[2]), phases[1], phases[2], u2, sigs[0])
        else:  # first root merges with second root
            ucomp = helpAsympB(sigs[3], 3)
            u2 = np.sum([fields[2], ucomp], axis=0)
            litG = helpAsympA(np.abs(fields[0]), np.abs(fields[1]), phases[0], phases[1], u2, sigs[3])
    else:  # five real roots
        u1, u2, u3, u4, u5 = fields  # super ugly, maybe there's a better way to do this
        u11, u12 = np.split(u1, 2)
        u21, u22 = np.split(u2, 2)
        u31, u32 = np.split(u3, 2)
        u41, u42 = np.split(u4, 2)
        u51, u52 = np.split(u5, 2)
        phi1, phi2, phi3, phi4, phi5 = phases
        phi11, phi12 = np.split(phi1, 2)
        phi31, phi32 = np.split(phi3, 2)
        phi41, phi42 = np.split(phi4, 2)
        phi51, phi52 = np.split(phi5, 2)
        litG1 = helpAsympA(np.abs(u41), np.abs(u51), phi41, phi51, np.sum([u11, u21, u31], axis=0), sigs[1])
        litG2 = helpAsympA(np.abs(u12), np.abs(u32), phi12, phi32, np.sum([u22, u42, u52], axis=0), sigs[1])
        litG = np.concatenate((litG1, litG2))
    return litG

def darkAsymp(roots, phases, fields, sigs, rF2, lc, ax, ay):

    def helpAsymp(sig, pos):
        npoints = len(roots)
        cphases = phases[pos]
        amp = np.zeros(npoints)
        for i in range(npoints):
            amp[i] = GOAmplitude(roots[i][pos], rF2, lc, ax, ay)
        xi = (1.5*np.abs(cphases.imag))**(2./3.)
        u1 = 2*pi**0.5*amp *(xi)**0.25 * airy(xi)[0] * exp(1j*(cphases.real + sig*0.25*pi))
        return u1

    nroots = roots.shape[1]

    if nroots == 2:
        u1 = helpAsymp(sigs[0], 1)
        return np.abs(u1 + fields[0])**2
    else:
        u1 = helpAsymp(sigs[1], 1)
        if ax == ay:
            u2 = helpAsymp(sigs[1], 2)
        else:
            u2 = helpAsymp(sigs[2], 2)
        return np.abs(u1 + u2 + fields[0])**2

def planeSliceG(uxmax, uymax, dso, dsl, f, dm, m, n, ax, ay, npoints = 100, gsizex = 2048, gsizey = 2048):
    """ Plots gain for slice across the u'-plane for given lens parameters, observation frequency, uxmax, slope m and offset n. Compares it to the gain given by solving the Kirchhoff diffraction integral using convolution. Plots the slice gain and the entire u' plane gain. """

    # Calculate coefficients
    rF2 = rFsqr(dso, dsl, f)
    uF2x, uF2y = rF2*np.array([1./ax**2, 1./ay**2])
    lc = lensc(dm, f)
    alp  = rF2*lc
    coeff = alp*np.array([1./ax**2, 1./ay**2])

    # Calculate caustic intersections
    ucross = polishedRoots(causticEqSlice, uxmax, uymax, args = (alp, m, n, ax, ay))
    ncross = len(ucross)
    upcross = mapToUp(ucross.T, alp, ax, ay)
    p = np.argsort(upcross[0])
    upcross = upcross.T[p]
    ucross = ucross[p]
    print(ucross)

    # Calculate sign of second derivative at caustics
    sigs = np.zeros(ncross)
    for i in range(ncross):
        sigs[i] = np.sign(ax**2/rF2 + lc*gauss20(ucross[i][0], ucross[i][1]))
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

    cdist = xmax*2e-3

    if ncross == 2: # 2 dark regions with 1 image, 1 bright region with 3 images
        # Create slice by segments
        d1upx = np.linspace(xmin, upcross[0][0] - cdist, npoints) # d = dark, b = bright
        bupx = np.linspace(upcross[0][0] + cdist, upcross[1][0] - cdist, npoints)
        d2upx = np.linspace(upcross[1][0] + cdist, xmax, npoints)
        upxvecs = np.array([d1upx, bupx, d2upx])
        segs = np.asarray([lineVert(upx, m, n) for upx in upxvecs])
        nsolns = np.array([1, 3, 1]) # Number of expected real solutions at each segment

    elif ncross == 4:
        if dm > 0: # Positive DM. 3 dark regions with 1 image, 2 bright regions with 3 images.
            # Create slice by segments
            d1upx = np.linspace(xmin, upcross[0][0] - cdist, npoints) # d = dark, b = bright
            b1upx = np.linspace(upcross[0][0] + cdist, upcross[1][0] - cdist, npoints)
            d2upx = np.linspace(upcross[1][0] + cdist, upcross[2][0] - cdist, npoints)
            b2upx = np.linspace(upcross[2][0] + cdist, upcross[3][0] - cdist, npoints)
            d3upx = np.linspace(upcross[3][0] + cdist, xmax, npoints)
            upxvecs = np.array([d1upx, b1upx, d2upx, b2upx, d3upx])
            segs = np.asarray([lineVert(upx, m, n) for upx in upxvecs])
            nsolns = np.array([1, 3, 1, 3, 1])

        if dm < 0: # Negative DM. 2 dark regions with 1 image, 2 bright regions with 3 images, 1 bright region with 5 images.
            d1upx = np.linspace(xmin, upcross[0][0] - cdist, npoints) # d = dark, b = bright
            b1upx = np.linspace(upcross[0][0] + cdist, upcross[1][0] - cdist, npoints)
            b2upx = np.linspace(upcross[1][0] + cdist, upcross[2][0] - cdist, npoints)
            b3upx = np.linspace(upcross[2][0] + cdist, upcross[3][0] - cdist, npoints)
            d2upx = np.linspace(upcross[3][0] + cdist, xmax, npoints)
            upxvecs = np.array([d1upx, b1upx, b2upx, b3upx, d2upx])
            segs = np.asarray([lineVert(upx, m, n) for upx in upxvecs])
            nsolns = np.array([1, 3, 5, 3, 1])


    # Solve lens equation at each coordinate
    allroots = []
    for i in range(len(nsolns)):
        roots = rootFinder(segs[i], nsolns[i], dm, cdist, ucross, upcross, ncross, uxmax, uymax, coeff)
        allroots.append(roots)
    # print(allroots)

    # Calculate fields
    allfields = []
    for roots in allroots:
        fields = fieldCalc(roots, npoints, rF2, lc, ax, ay)
        allfields.append(fields)

    # Calculate phases
    allphases = []
    for roots in allroots:
        phis = phaseCalc(roots, npoints, rF2, lc, ax, ay)
        allphases.append(phis)

    # Construct uniform asymptotics
    if ncross == 2:
        litG = litAsymp(allroots[1], allphases[1], allfields[1], sigs, rF2, lc, ax, ay)
        darkG1 = darkAsymp(allroots[0], allphases[0], allfields[0], sigs, rF2, lc, ax, ay)
        darkG2 = darkAsymp(allroots[2], allphases[2], allfields[2], sigs, rF2, lc, ax, ay)
        interp = interp1d(upxvecs.flatten(), np.concatenate((darkG1, litG, darkG2)), kind = 'cubic', fill_value = 'extrapolate')
    else:
        if dm > 0:
            litG1 = litAsymp(allroots[1], allphases[1], allfields[1], sigs, rF2, lc, ax, ay)
            litG2 = litAsymp(allroots[3], allphases[3], allfields[3], sigs, rF2, lc, ax, ay)
            darkG1 = darkAsymp(allroots[0], allphases[0], allfields[0], sigs, rF2, lc, ax, ay)
            darkG2 = darkAsymp(allroots[2], allphases[2], allfields[2], sigs, rF2, lc, ax, ay)
            darkG3 = darkAsymp(allroots[4], allphases[4], allfields[4], sigs, rF2, lc, ax, ay)
            interp = interp1d(upxvecs.flatten(), np.concatenate((darkG1, litG1, darkG2, litG2, darkG3)), kind = 'cubic', fill_value = 'extrapolate')
        else:
            litG1 = litAsymp(allroots[1], allphases[1], allfields[1], sigs, rF2, lc, ax, ay)
            litG2 = litAsymp(allroots[2], allphases[2], allfields[2], sigs, rF2, lc, ax, ay)
            litG3 = litAsymp(allroots[3], allphases[3], allfields[3], sigs, rF2, lc, ax, ay)
            darkG1 = darkAsymp(allroots[0], allphases[0], allfields[0], sigs, rF2, lc, ax, ay)
            darkG2 = darkAsymp(allroots[4], allphases[4], allfields[4], sigs, rF2, lc, ax, ay)
            interp = interp1d(upxvecs.flatten(), np.concatenate((darkG1, litG1, litG2, litG3, darkG2)), kind = 'cubic', fill_value = 'extrapolate')

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
    fig = plt.figure(figsize = (15, 10))
    grid = gs.GridSpec(2, 2, height_ratios = [3, 1])
    tableax = plt.subplot(grid[1, :])
    ax0, ax1 = plt.subplot(grid[0, 0]), plt.subplot(grid[0, 1])

    rx = np.linspace(-uxmax, uxmax, gsizex)
    ry = np.linspace(-uymax, uymax, gsizey)
    ux, uy = np.meshgrid(rx, ry)

    rx2 = np.linspace(xmin, xmax, gsizex)
    im0 = ax0.imshow(soln, origin = 'lower', extent = extent, aspect = 'auto', cmap = 'jet') # Plot entire screen
    fig.colorbar(im0, ax = ax0)
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
    ax0.set_xlabel(r"$u'_x$")
    ax0.set_ylabel(r"$u'_y$")
    ax0.set_title("Gain in the u' plane")

    G = map_coordinates(soln.T, np.vstack((xx, yy))) # Plot gain along observer motion
    ax1.plot(rx2, G, color = 'blue')
    for caus in upcross.T[0]:
        ax1.plot([caus, caus], [-10, 1000], ls = 'dashed', color = 'black')
    ax1.plot(finx, asymG, color = 'red')
    ax1.set_ylim(-cdist, np.max(asymG) + 1.)
    ax1.set_xlim(np.min(rx2), np.max(rx2))
    ax1.set_xlabel(r"$u'_x$")
    ax1.set_ylabel('G')
    ax1.set_title("Slice Gain")
    ax1.grid()


    col_labels = ['Parameter', 'Value'] # Create table with parameter values
    if np.abs(dm/pctocm) < 1:
        dmlabel = "{:.2E}".format(Decimal(dm/pctocm))
    else:
        dmlabel = str(dm/pctocm)
    tablevals = [[r'$d_{so} \: (kpc)$', np.around(dso/pctocm/kpc, 2)], [r'$d_{sl} \: (kpc)$', np.around(dsl/pctocm/kpc, 3)], [r'$a_x \: (AU)$', np.around(ax/autocm, 3)], [r'$a_y \: (AU)$', np.around(ay/autocm, 3)], [r'$DM_l \: (pc \, cm^{-3})$', dmlabel], [r"$\nu$ (GHz)", f/GHz], ['Slope', np.around(m, 2)], ['Offset', n]]
    tableax.axis('tight')
    tableax.axis('off')
    table = tableax.table(cellText = np.asarray(tablevals).T, colWidths = np.ones(8)*0.05, rowLabels = col_labels, loc = 'center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(2.5, 2.5)

    plt.show()
    return
