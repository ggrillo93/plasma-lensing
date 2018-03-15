from fundfunctions import *
from observables import *
from solvers import *
from scipy.special import airy
from scipy.spatial import distance
from scipy.interpolate import *
from kdi import *

def difference(arr):
    diff = np.ones(len(arr))
    diff[0] = arr[0] - arr[1]
    diff[-1] = arr[-1] - arr[-2]
    for i in range(1, len(arr) - 1):
        diff[i] = 2*arr[i] - arr[i-1] - arr[i+1]
    return diff

def findClosest(roots):
    dist = pdist(roots)
    mdist = np.min(dist)
    dist = squareform(dist)
    ij_min = np.where(dist == mdist)
    return [ij_min[0], mdist]

def obsCalc(func, roots, nroots, npoints, args = ()):
    """ Calculates observable using observable function func for a list of roots of arbitrary dimensionality. Returns multidimensional array with shape [nroots, npoints]. """
    obs = np.zeros([nroots, npoints], dtype = complex)
    for i in range(npoints):
        for j in range(nroots):
            obs[j][i] = func(roots[i][j], *args)
    return obs

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
        u1 = 2*pi**0.5*amp * (xi)**0.25 * airy(xi)[0] * exp(1j*(cphases.real - sig*0.25*pi))
        return u1
    
    
    # merge = [findClosest(roots[0]), findClosest(roots[-1])] # find closest roots at each end
    # if merge[0][1] < 0.3 and merge[1][1] < 0.3: # roots merge at both ends
    #     mroot1, mroot2 = merge[0][0], merge[1][0] # set indices of merging roots
    #     nmroots1 = list(set(range(nroots)) - set(mroot1)) # indices of non merging roots at one end
    #     nmroots2 = list(set(range(nroots)) - set(mroot2)) # indices of non merging roots at other end
    #     if mroot1 == mroot2: # same root merges at both ends
    #         amp1, amp2 = np.abs(fields[mroot1[0]], np.abs(fields[mroot1[1]]))
    #         phi1, phi2 = phases[mroot1[0]], phases[mroot1[1]]
    #         # check for complex roots
    #         others = np.zeros(len(amp1))
    #         for index in nmroots1:
    #             others = others + fields[index] # sum of fields not involved in merging
    #         litG = helpAsympA(amp1, amp2, phi1, phi2, )
    
    nroots = roots.shape[1]
    if nroots == 3: # just three real roots
                cond1 = np.abs(roots[0][0][0] - roots[0][1][0]) < np.abs(roots[0][1][0] - roots[0][2][0])
                cond2 = np.abs(roots[-1][0][0] - roots[-1][1][0]) < np.abs(roots[-1][1][0] - roots[-1][2][0])
                if cond1 and cond2: # first root merges with second root at both ends
                    litG = helpAsympA(np.abs(fields[0]), np.abs(fields[1]), phases[0], phases[1], fields[2], sigs[0])
                elif not cond1 and not cond2: # second root merges with third root at both ends
                    litG = helpAsympA(np.abs(fields[1]), np.abs(fields[2]), phases[1], phases[2], fields[0], sigs[0])
                else: # need to split in two
                    u1, u2, u3 = fields
                    u11, u12 = np.split(u1, 2)
                    U21, U22 = np.split(np.abs(u2), 2)
                    u31, u32 = np.split(u3, 2)
                    phi1, phi2, phi3 = phases
                    phi11, phi12 = np.split(phi1, 2)
                    phi21, phi22 = np.split(phi2, 2)
                    phi31, phi32 = np.split(phi3, 2)
                    if cond1 and not cond2: # first root merges with second root at first end and second root merges with third root at other end
                        litG1 = helpAsympA(np.abs(u11), U21, phi11, phi21, u31, sigs[0])
                        litG2 = helpAsympA(U22, np.abs(u32), phi22, phi32, u12, sigs[0])
                        litG = np.concatenate((litG1, litG2))
                    else: # second root merges with third root at first end and first root merges with second root at other end
                        litG1 = helpAsympA(U21, np.abs(u31), phi21, phi31, u11, sigs[0])
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
        phi21, phi22 = np.split(phi2, 2)
        phi31, phi32 = np.split(phi3, 2)
        phi41, phi42 = np.split(phi4, 2)
        phi51, phi52 = np.split(phi5, 2)
        litG1 = helpAsympA(np.abs(u31), np.abs(u51), phi31, phi51, np.sum([u11, u21, u41], axis=0), sigs[1])
        litG2 = helpAsympA(np.abs(u12), np.abs(u22), phi12, phi22, np.sum([u32, u42, u52], axis=0), sigs[1])
        litG = np.concatenate((litG1, litG2))
    return litG

def darkAsymp(roots, phases, fields, sigs, rF2, lc, ax, ay):

    def helpAsymp(sig, pos):
        cphases = phases[pos]
        amp = np.zeros(npoints)
        for i in range(npoints):
            amp[i] = GOAmplitude(roots[i][pos], rF2, lc, ax, ay)
        xi = (1.5*np.abs(cphases.imag))**(2./3.)
        u1 = 2*pi**0.5*amp *(xi)**0.25 * airy(xi)[0] * exp(1j*(cphases.real + sig*0.25*pi))
        return u1

    nroots = roots.shape[1]
    npoints = len(roots)

    if nroots == 2:
        u1 = helpAsymp(sigs[0], 1)
        return np.abs(u1 + fields[0])**2
    else:
            u1 = helpAsymp(sigs[1], 1)
            if ax == ay:
                if np.around(roots[0][1][0].real, 3) != np.around(roots[0][2][0].real, 3):
                    # print([roots[0][1][0].real, roots[0][2][0].real])
                    u2 = helpAsymp(sigs[1], 2)
                else:
                    u2 = np.zeros(npoints)
            else:
                u2 = helpAsymp(sigs[2], 2)
            return np.abs(u1 + u2 + fields[0])**2

def planeSliceTOA(uxmax, uymax, dso, dsl, f, dm, m, n, ax, ay, npoints):
    """ Plots TOA perturbation for slice across the u'-plane for given lens parameters, observation frequency, uxmax, slope m and offset n. Also shows path across the plane with respect with the caustic curves. """
    
    def findRealRoots(segs):
        """ Finds all real roots of the lens equation as a function of u' coordinates given in segs[i]. Unlike rootFinder, finds the initial set of roots at the middle of segs[i] and iterates forward and backwards. rootFinder, on the other hand, starts from the beginning and moves forward. This function is able to find roots that are much closer to the caustic than rootFinder."""
        allroots = []
        for i in range(len(segs)):
            temp0 = polishedRoots(lensEq, 2*uxmax, 2*uymax, args = (segs[i][npoints/2], coeff))
            nsolns = len(temp0)
            roots = np.zeros([npoints, nsolns, 2])
            roots[npoints/2] = temp0
            for j in np.flipud(range(npoints/2 + 1)):
                for k in range(nsolns):
                    temp = op.root(lensEq, roots[j][k], args = (segs[i][j-1], coeff))
                    if temp.success:
                        roots[j-1][k] = temp.x
                    else:
                        print('Error')
                        print(segs[i][j])
                        print(temp)
            for j in range(npoints/2 + 1, npoints):
                for k in range(nsolns):
                    temp = op.root(lensEq, roots[j-1][k], args = (segs[i][j], coeff))
                    if temp.success:
                        roots[j][k] = temp.x
                    else:
                        print('Error')
                        print(segs[i][j])
                        print(temp)
            allroots.append(roots)
        return allroots
    
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
        
    cdist = xmax*1e-6
    
    bound = np.insert(upcross, 0, np.array([[xmin, ymin]]), axis = 0)
    bound = np.append(bound, np.array([[xmax, ymax]]), axis = 0)
    upxvecs = np.asarray([np.linspace(bound[i-1][0] + cdist, bound[i][0] - cdist, npoints) for i in range(1, ncross + 2)])
    segs = np.asarray([lineVert(upx, m, n) for upx in upxvecs])
    allroots = findRealRoots(segs)
    nsolns = [len(roots[0]) for roots in allroots]
    print(nsolns)
    # print(allroots)
    
    # Calculate TOAs
    alltoas = []
    for i in range(len(allroots)):
        toas = obsCalc(deltatA, allroots[i], nsolns[i], npoints, args = (tg0, tdm0, alp, ax, ay)).real
        alltoas.append(toas)
    
    # Plots
    fig = plt.figure(figsize=(15, 10))
    grid = gs.GridSpec(2, 2, width_ratios=[4, 1])
    # grid.update(hspace=mincdist)
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
    # ax1.set_ylim(np.min(upy)*1.5, np.max(upy)*1.5)
    ax1.set_title("Caustic curves")
    ax1.set_aspect('equal', anchor = 'C')
    ax1.grid()
    
    ax2 = plt.subplot(grid[:, 0]) # Plot results
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i in range(len(upxvecs)):
        zone = alltoas[i]
        for j in range(len(zone)):
            ax2.plot(upxvecs[i], zone[j], color = 'black')
    ax2.set_ylabel(r'$\Delta t \: (\mu s)$')
    ax2.set_xlabel(r"$u'_x$")
    ax2.grid()
    
    # Create table
    col_labels = ['Parameter', 'Value']
    if np.abs(dm/pctocm) < 1:
        dmlabel = "{:.2E}".format(Decimal(dm/pctocm))
    else:
        dmlabel = str(dm/pctocm)
    tablevals = [[r'$d_{so} \: (kpc)$', np.around(dso/pctocm/kpc, 2)], [r'$d_{sl} \: (kpc)$', np.around(dsl/pctocm/kpc, 2)], [r'$a_x \: (AU)$', np.around(ax/autocm, 2)], [r'$a_y \: (AU)$', np.around(ay/autocm, 2)], [r'$DM_l \: (pc \, cm^{-3})$', dmlabel], [r"$\nu$ (GHz)", f/GHz], ['Slope', m], ['Offset', n]]
    ax0.axis('tight')
    ax0.axis('off')
    table = ax0.table(cellText = tablevals, colWidths = [0.25, 0.25], colLabels = col_labels, loc = 'center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(3.2, 3.2)
    
    plt.show()
    return
    
@profile    
def planeSliceG(uxmax, uymax, dso, dsl, f, dm, m, n, ax, ay, npoints = 4000, gsizex = 2048, gsizey = 2048):
    """ Plots gain for slice across the u'-plane for given lens parameters, observation frequency, uxmax, slope m and offset n. Compares it to the gain given by solving the Kirchhoff diffraction integral using convolution. Plots the slice gain and the entire u' plane gain. """

    # Calculate coefficients
    rF2 = rFsqr(dso, dsl, f)
    uF2x, uF2y = rF2*np.array([1./ax**2, 1./ay**2])
    lc = lensc(dm, f)
    alp  = rF2*lc
    coeff = alp*np.array([1./ax**2, 1./ay**2])

    # Calculate caustic intersections
    ucross = polishedRoots(causticEqSlice, uxmax, uymax, args = (alp, m, n, ax, ay))
    plt.close()
    ncross = len(ucross)
    upcross = mapToUp(ucross.T, alp, ax, ay)
    p = np.argsort(upcross[0])
    upcross = upcross.T[p]
    ucross = ucross[p]
    print(upcross)

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

    cdist = uxmax/(np.abs(5*lc))
    print(cdist)

    bound = np.insert(upcross, 0, np.array([[xmin, ymin]]), axis = 0) # set up boundaries
    bound = np.append(bound, np.array([[xmax, ymax]]), axis = 0)
    midpoints = [(bound[i] + bound[i+1])/2. for i in range(len(bound) - 1)] # find middle point between boundaries
    nzones = len(midpoints)
    nreal = np.zeros(nzones)
    for i in range(nzones): # find number of roots at each midpoint
        mpoint = midpoints[i]
        nreal[i] = len(findRoots(lensEq, 2*uxmax, 2*uymax, args = (mpoint, coeff)))
    upxvecs = np.array([np.linspace(bound[i-1][0] + cdist, bound[i][0] - cdist, npoints) for i in range(1, ncross + 2)]) # generate upx vector
    # print(upxvecs)
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
    # print(allroots[2])
    
    # Calculate fields
    allfields = []
    for i in range(nzones):
        fields = obsCalc(GOfieldA, allroots[i], len(allroots[i][0]), npoints, args=(rF2, lc, ax, ay))
        allfields.append(fields)

    # Calculate phases
    allphases = []
    for i in range(nzones):
        phis = obsCalc(phi, allroots[i], len(allroots[i][0]), npoints, args=(rF2, lc, ax, ay))
        allphases.append(phis)

    # Construct uniform asymptotics
    if ncross == 2:
        litG = litAsymp(allroots[1], allphases[1], allfields[1], sigs, rF2, lc, ax, ay)
        darkG1 = darkAsymp(allroots[0], allphases[0], allfields[0], sigs, rF2, lc, ax, ay)
        darkG2 = darkAsymp(allroots[2], allphases[2], allfields[2], sigs, rF2, lc, ax, ay)
        interp = UnivariateSpline(upxvecs.flatten(), np.concatenate((darkG1, litG, darkG2)), s = 0)
    else:
        if dm > 0:
            litG1 = litAsymp(allroots[1], allphases[1], allfields[1], sigs, rF2, lc, ax, ay)
            litG2 = litAsymp(allroots[3], allphases[3], allfields[3], sigs, rF2, lc, ax, ay)
            darkG1 = darkAsymp(allroots[0], allphases[0], allfields[0], sigs, rF2, lc, ax, ay)
            darkG2 = darkAsymp(allroots[2], allphases[2], allfields[2], sigs, rF2, lc, ax, ay)
            darkG3 = darkAsymp(allroots[4], allphases[4], allfields[4], sigs, rF2, lc, ax, ay)
            interp = UnivariateSpline(upxvecs.flatten(), np.concatenate((darkG1, litG1, darkG2, litG2, darkG3)), s = 0)
        else:
            litG1 = litAsymp(allroots[1], allphases[1], allfields[1], sigs, rF2, lc, ax, ay)
            litG2 = litAsymp(allroots[2], allphases[2], allfields[2], sigs, rF2, lc, ax, ay)
            litG3 = litAsymp(allroots[3], allphases[3], allfields[3], sigs, rF2, lc, ax, ay)
            darkG1 = darkAsymp(allroots[0], allphases[0], allfields[0], sigs, rF2, lc, ax, ay)
            darkG2 = darkAsymp(allroots[4], allphases[4], allfields[4], sigs, rF2, lc, ax, ay)
            interp = UnivariateSpline(upxvecs.flatten(), np.concatenate((darkG1, litG1, litG2, litG3, darkG2)), s = 0)

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
