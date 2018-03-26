from fundfunctions import *
from observables import *
from solvers import *
from scipy.special import airy
from scipy.spatial.distance import *
from scipy.interpolate import *
from kdi import *

def constructField(amp, phases, pshift):
    return amp*np.exp(1j*(phases + pshift))
    
def difference(arr):
    diff = np.ones(len(arr))
    diff[0] = arr[0] - arr[1]
    diff[-1] = arr[-1] - arr[-2]
    for i in range(1, len(arr) - 1):
        diff[i] = 2*arr[i] - arr[i-1] - arr[i+1]
    return diff

def findClosest(roots):
    """ Returns index of the roots closest to each other and their distance. """
    dist = pdist(roots)
    mdist = np.min(dist)
    dist = squareform(dist)
    ij_min = np.where(dist == mdist)
    return [ij_min[0], mdist]

def obsCalc(func, roots, nroots, npoints, ansdim, args = ()):
    """ Calculates observable using observable function func for a list of roots of arbitrary dimensionality. Returns multidimensional array with shape [nroots, ansdim, npoints]. """
    if ansdim == 1:
        obs = np.zeros([nroots, npoints], dtype = complex)
        for i in range(npoints):
            for j in range(nroots):
                obs[j][i] = func(roots[i][j], *args)
    else:
        obs = np.zeros([nroots, ansdim, npoints], dtype = complex)
        for i in range(npoints):
            for j in range(nroots):
                ans = func(roots[i][j], *args)
                for k in range(ansdim):
                    obs[j][k][i] = ans[k]
    return obs

def lineVert(upxvec, m, n):
    """ Returns list of line vertices. """
    return np.array([upxvec, m*upxvec + n]).T
    
def uniAsymp(allroots, allfields, nreal, ncomplex, npoints, nzones, sigs):
    """ Constructs the uniform asympotics for a segmented array of roots and their respective fields. """
    def bright(A1, A2, phi1, phi2, sig):
        if phi1[0] > phi2[0]:
            pdiff = phi1 - phi2
            g1 = A2 - A1
        else:
            pdiff = phi2 - phi1
            g1 = A1 - A2
        chi = 0.5*(phi1 + phi2)
        xi = -(0.75*pdiff)**(2./3.)
        air = airy(xi)
        a1 = pi**0.5 *((A1 + A2)*(-xi)**0.25*air[0] - 1j*g1*(-xi)**-0.25*air[1]) * np.exp(1j*(chi + sig*0.25*pi))
        return a1
        
    def dark(A, phi, sig):
        xi = (1.5*np.abs(phi.imag))**(2./3.)
        a1 = 2*pi**0.5*A*(xi)**0.25 * airy(xi)[0] * np.exp(1j*(phi.real + sig*0.25*pi))
        return a1
    
    asymp = np.zeros([nzones, npoints])
    for i in range(nzones):
        p = i - 1 # caustic index
        if i < nzones/2:
            p = i
        roots, fields = allroots[i], allfields[i]
        nroots = (ncomplex + nreal)[i]
        realn = int(nreal[i])
        if nreal[i] == 1: # no real roots merge
            areal = constructField(*fields[0])
        else: # deal with merging real roots
            merge = [findClosest(roots[0][:realn].real), findClosest(roots[-1][:realn].real)] # find closest real roots at each end
            mroot1, mroot2 = merge[0][0], merge[1][0] # set indices of merging roots
            print([mroot1, mroot2])
            # print([merge[0][1], merge[1][1]])
            nmroots1 = list(set(range(realn)) - set(mroot1)) # indices of non merging roots at one end
            nmroots2 = list(set(range(realn)) - set(mroot2)) # indices of non merging roots at other end
            if merge[0][1] < 0.4 and merge[1][1] < 0.4: # case 1: real root merging at both ends
                print('Double root merging')
                if np.all(mroot1 == mroot2): # same root merges at both ends
                    A1, phi1 = fields[mroot1[0]][:2]
                    A2, phi2 = fields[mroot1[1]][:2]
                    amerge = bright(A1, A2, phi1, phi2, sigs[p-1])
                    anonm = np.zeros(npoints, dtype = complex)
                    for index in nmroots1:
                        anonm = anonm + constructField(*fields[index]) # sum of fields not involved in merging
                    areal = amerge + anonm
                else: # different roots merge at each end
                    A11, A21 = np.split(fields[mroot1[0]][0], 2)[0], np.split(fields[mroot1[1]][0], 2)[0]
                    A32, A42 = np.split(fields[mroot2[0]][0], 2)[1], np.split(fields[mroot2[1]][0], 2)[1]
                    phi11, phi21 = np.split(fields[mroot1[0]][1], 2)[0], np.split(fields[mroot1[1]][1], 2)[0]
                    phi32, phi42 = np.split(fields[mroot2[0]][1], 2)[1], np.split(fields[mroot2[1]][1], 2)[1]
                    if i < nzones/2.:
                        amerge1 = bright(A11, A21, phi11, phi21, sigs[p])
                        amerge2 = bright(A32, A42, phi32, phi42, sigs[p])
                    else:
                        amerge1 = bright(A11, A21, phi11, phi21, sigs[p+1])
                        amerge2 = bright(A32, A42, phi32, phi42, sigs[p+1])
                    nmfields1 = [constructField(*np.split(fields[nmroot], 2, axis = 1)[0]) for nmroot in nmroots1]
                    nmfields2 = [constructField(*np.split(fields[nmroot], 2, axis = 1)[1]) for nmroot in nmroots2]
                    anonm1 = np.zeros(npoints/2, dtype = complex)
                    for j in range(len(nmroots1)):
                        anonm1 = anonm1 + nmfields1[j]
                    anonm2 = np.zeros(npoints/2, dtype = complex)
                    for j in range(len(nmroots2)):
                        anonm2 = anonm2 + nmfields2[j]
                    areal = np.concatenate((amerge1 + anonm1, amerge2 + anonm2))
            elif merge[0][1] < 0.4 and merge[1][1] > 0.4: # case 2: real root merging at first end only
                print('Root merging at first end')
                A1, phi1 = fields[mroot1[0]][:2]
                A2, phi2 = fields[mroot1[1]][:2]
                if i < nzones/2.:
                    amerge = bright(A1, A2, phi1, phi2, sigs[p-1])
                else:
                    amerge = bright(A1, A2, phi1, phi2, sigs[p])
                anonm = np.zeros(npoints, dtype = complex)
                for index in nmroots1:
                    anonm = anonm + constructField(*fields[index]) # sum of fields not involved in merging
                areal = amerge + anonm
            elif merge[0][1] > 0.4 and merge[1][1] < 0.4: # case 3: real root merging at second end only
                print('Root merging at second end')
                A1, phi1 = fields[mroot2[0]][:2]
                A2, phi2 = fields[mroot2[1]][:2]
                if i < nzones/2:
                    amerge = bright(A1, A2, phi1, phi2, sigs[p])
                else:
                    amerge = bright(A1, A2, phi1, phi2, sigs[p+1])
                anonm = np.zeros(npoints, dtype = complex)
                for index in nmroots2:
                    anonm = anonm + constructField(*fields[index]) # sum of fields not involved in merging
                areal = amerge + anonm
        if ncomplex[i] != 0: # deal with merging complex roots. its a mess, there should a better way of doing it
            A, phi = fields[realn][:2]
            cond = np.abs(roots[0][realn][0].imag/roots[0][realn][0].real) < np.abs(roots[-1][realn][0].imag/roots[-1][realn][0].real)
            if cond: # complex root merges at first end
                if i < nzones/2.:
                    acomp = dark(A, phi, sigs[p - 1])
                else:
                    acomp = dark(A, phi, sigs[p])
            else: # complex root merges at second end
                if i < nzones/2. or i == nzones - 1:
                    acomp = dark(A, phi, sigs[p])
                else:
                    acomp = dark(A, phi, sigs[p + 1])
            if ncomplex[i] == 2 and np.around(fields[realn][0][0], 3) != np.around(fields[realn + 1][0][0], 3):
                A2, phi2 = fields[realn + 1][:2]
                if cond:
                    if i < nzones/2.:
                        acomp = acomp + dark(A2, phi2, sigs[p])
                    else:
                        acomp = acomp + dark(A2, phi2, sigs[p + 1])
                else:
                    if i < nzones/2.:
                        acomp = acomp + dark(A2, phi2, sigs[p - 1])
                    else:
                        acomp = acomp + dark(A2, phi2, sigs[p])
        else:
            acomp = np.zeros(npoints)
        asymp[i] = np.abs(areal + acomp)**2
    return asymp.flatten()

def planeSliceTOA(uxmax, uymax, dso, dsl, f, dm, m, n, ax, ay, npoints):
    """ Plots TOA perturbation for slice across the u'-plane for given lens parameters, observation frequency, uxmax, slope m and offset n. Also shows path across the plane with respect with the caustic curves. """
    
    # Calculate coefficients
    rF2 = rFsqr(dso, dsl, f)
    uF2x, uF2y = rF2*np.array([1./ax**2, 1./ay**2])
    lc = lensc(dm, f)
    alp = rF2*lc
    coeff = alp*np.array([1./ax**2, 1./ay**2])
    tg0 = tg0coeff(dso, dsl)
    tdm0 = tdm0coeff(dm, f)

    # Calculate caustic intersections
    ucross = polishedRoots(causticEqSlice, uxmax, uymax, args=(alp, m, n, ax, ay))
    ncross = len(ucross)
    upcross = mapToUp(ucross.T, alp, ax, ay)
    p = np.argsort(upcross[0])
    upcross = upcross.T[p]
    ucross = ucross[p]
    # print(upcross)
    
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
        
    cdist = uxmax/(np.abs(50*lc))
    
    bound = np.insert(upcross, 0, np.array([[xmin, ymin]]), axis = 0) # set up boundaries
    bound = np.append(bound, np.array([[xmax, ymax]]), axis = 0)
    midpoints = [(bound[i] + bound[i+1])/2. for i in range(len(bound) - 1)] # find middle point between boundaries
    nzones = len(midpoints)
    nreal = np.zeros(nzones)
    for i in range(nzones): # find number of roots at each midpoint
        mpoint = midpoints[i]
        nreal[i] = len(findRoots(lensEq, 2*uxmax, 2*uymax, args = (mpoint, coeff)))
    upxvecs = np.array([np.linspace(bound[i-1][0] + cdist, bound[i][0] - cdist, npoints) for i in range(1, ncross + 2)]) # generate upx vector
    segs = np.asarray([lineVert(upx, m, n) for upx in upxvecs]) # generate slice across plane
    ncomplex = np.zeros(nzones) # don't care about complex solutions in this case
    print(nreal)
    
    # Find roots
    allroots = rootFinder(segs, nreal, ncomplex, npoints, ucross, uxmax, uymax, coeff)
    
    # Calculate TOAs
    alltoas = []
    for i in range(nzones):
        toas = obsCalc(deltat, allroots[i], int(nreal[i]), npoints, 1, args = (tg0, tdm0, alp, ax, ay)).real
        alltoas.append(toas)
    
    # Plots
    fig = plt.figure(figsize=(15, 10))
    grid = gs.GridSpec(2, 2, width_ratios=[4, 1])
    ax0 = plt.subplot(grid[1:, 1])
    ax1 = plt.subplot(grid[0, 1])
    
    rx = np.linspace(-uxmax, uxmax, 1000) # Plot caustic surfaces
    ry = np.linspace(-uxmax, uxmax, 1000)
    uvec = np.meshgrid(rx, ry)
    ucaus = causCurve(uvec, coeff)
    cs = ax1.contour(rx, ry, ucaus, levels = [0, np.inf], linewidths = 0)
    paths = cs.collections[0].get_paths()
    uppaths = []
    for p in paths:
        cuvert = np.array(p.vertices).T
        upx, upy = mapToUp(cuvert, alp, ax, ay)
        ax1.plot(upx, upy, color = 'blue')
    ax1.plot(np.linspace(xmin, xmax, 10), np.linspace(ymin, ymax, 10), color = 'green')
    ax1.scatter(upcross.T[0], upcross.T[1], color = 'green')
    ax1.set_xlabel(r"$u'_x$")
    ax1.set_ylabel(r"$u'_y$")
    ax1.set_xlim(-uxmax, uxmax)
    ax1.set_title("Caustic curves")
    ax1.set_aspect('equal', anchor = 'N')
    ax1.grid()
    
    ax2 = plt.subplot(grid[:, 0]) # Plot results
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i in range(len(upxvecs)):
        zone = alltoas[i]
        for j in range(len(zone)):
            ax2.plot(upxvecs[i], zone[j], color = 'black')
    ax2.set_ylabel(r'$\Delta t \: (\mu s)$')
    ax2.set_xlabel(r"$u'_x$")
    ax2.set_title('Lens shape: ' + '$%s$' % sym.latex(lensf))
    ax2.grid()
    
    
    # Create table
    col_labels = ['Parameter', 'Value']
    if np.abs(dm/pctocm) < 1:
        dmlabel = "{:.2E}".format(Decimal(dm/pctocm))
    else:
        dmlabel = str(dm/pctocm)
    tablevals = [[r'$d_{so} \: (kpc)$', np.around(dso/pctocm/kpc, 2)], [r'$d_{sl} \: (kpc)$', np.around(dsl/pctocm/kpc, 2)], [r'$a_x \: (AU)$', np.around(ax/autocm, 2)], [r'$a_y \: (AU)$', np.around(ay/autocm, 2)], [r'$DM_l \: (pc \, cm^{-3})$', dmlabel], [r"$\nu$ (GHz)", np.around(f/GHz, 2)], ['Slope', m], ['Offset', n]]
    ax0.axis('tight')
    ax0.axis('off')
    ax0.set_anchor('N')
    table = ax0.table(cellText = tablevals, colWidths = [0.25, 0.25], colLabels = col_labels, loc = 'center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(3., 3.)
    
    plt.show()
    return
    
# @profile    
def planeSliceG(uxmax, uymax, dso, dsl, f, dm, m, n, ax, ay, npoints = 3000, gsizex = 2048, gsizey = 2048):
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

    cdist = uxmax/(np.abs(5*lc))
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
    ncomplex = np.ones(nzones)*100
    for i in range(nzones):
        if diff[i] == 0 or diff[i] == -2:
            ncomplex[i] = 1
        elif diff[i] == -4:
            ncomplex[i] = 2
        elif diff[i] == 4:
            ncomplex[i] = 0
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
    fig = plt.figure(figsize = (15, 10))
    grid = gs.GridSpec(3, 2, height_ratios = [4, 1, 0.2])
    tableax = plt.subplot(grid[1, :])
    tableax2 = plt.subplot(grid[2, :])
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
    ax0.set_ylim([-uymax, uymax])
    ax0.set_xlim([-uxmax, uxmax])
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
    table = tableax.table(cellText = np.asarray(tablevals).T, colWidths = np.ones(8)*0.045, rowLabels = col_labels, loc = 'center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(2.5, 2.5)
    
    row_label =  ['Lens shape']
    val = [['$%s$' % sym.latex(lensf)]]
    tableax2.axis('tight')
    tableax2.axis('off')
    table2 = tableax2.table(cellText=val, colWidths=[0.0015*len(sym.latex(lensf))], rowLabels=row_label, loc='top')
    table2.auto_set_font_size(False)
    table2.set_fontsize(12)
    table2.scale(2.5, 2.5)

    plt.show()
    return
