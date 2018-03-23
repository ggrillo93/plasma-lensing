from fundfunctions import *
import matplotlib.pyplot as plt
import scipy.optimize as op
from matplotlib import gridspec as gs
from shapely import geometry

def compLensEq(uvec, upvec, coeff):
    """ Evaluates the 2D lens equation with u a complex vector. """
    uxr, uxi, uyr, uyi = uvec
    upx, upy = upvec
    grad = lensg(*[uxr + 1j*uxi, uyr + 1j*uyi])
    eq = np.array([uxr + 1j*uxi + coeff[0]*grad[0] - upx, uyr + 1j*uyi + coeff[1]*grad[1] - upy])
    return [eq[0].real, eq[0].imag, eq[1].real, eq[1].imag]

def lensEq(uvec, upvec, coeff):
    """ Evaluates the 2D lens equation. coeff = alp*[1/ax**2, 1/ay**2]. """
    ux, uy = uvec
    upx, upy = upvec
    grad = lensg(ux, uy)
    return np.array([ux + coeff[0]*grad[0] - upx, uy + coeff[1]*grad[1] - upy])

def close(myarr, list_arrays):
    """ Determines whether array is inside another list of arrays. """
    return next((True for elem in list_arrays if elem.size == myarr.size and np.allclose(elem, myarr, atol = 1e-3)), False)

# Caustic finders

def causCurve(uvec, coeff):
    ux, uy = uvec
    psi20, psi02, psi11 = lensh(ux, uy)
    return 1 + coeff[0]*psi20 + coeff[1]*psi02 - coeff[0]*coeff[1]*(psi11**2 - psi20*psi02)

def causCurveFreq(uvec, ax, ay, m, n):
    ux, uy = uvec
    psi20, psi02, psi11 = lensh(ux, uy)
    psi10, psi01 = lensg(ux, uy)
    gamma = (uy - m*ux - n)/(m*psi10/ax**2 - psi01/ay**2)
    return 1 + gamma*(psi20/ax**2 + psi02/ay**2) - gamma**2*(psi11**2 - psi02*psi20)/(ax*ay)**2

def causticEqSlice(uvec, alp, m, n, ax, ay):
    """ Evaluates the caustic equations for a slice across the u'-plane for given ux, uy, slope m and offset n, and lens parameters. Input in cgs units. """
    ux, uy = uvec
    grad = lensg(ux, uy)
    eq1 = uy - m*ux - n + alp/ay**2*grad[1] - m*alp/ax**2*grad[0] 
    psi20, psi02, psi11 = lensh(ux, uy)
    eq2 = 1 + alp*(ay**2*psi20 + ax**2*psi02)/(ay*ax)**2 - alp**2*(psi11**2 - psi20*psi02)/(ay*ax)**2
    return np.array([eq1, eq2])

def causticEqFreq(uvec, upvec, ax, ay):
    """ Evaluates the caustic equations for the frequency line, given ux, uy, upx, upy, ax, and ay. """
    ux, uy = uvec
    upx, upy = upvec
    eq1 = 2*uy**3*upx - 2*uy**2*upx*upy + upx*upy - 2*ux**2*upx*upy + 2*ux**3*upy
    eq2 = ax**2*uy*(upx - ux)/ay**2 - ux*(upy - uy)
    return np.array([eq1, eq2])

def causticFreq(roots, upx, ax, dsl, dso, dm):
    """ Locates caustic in the frequency line for given root, upx and parameters. coeff = (c/ax)*(dsl*dlo*re*dm/(2*pi*dso))**0.5 """
    freqcaustics = []
    dlo = dso - dsl
    coeff = dsl*dlo*re*dm/(2*pi*dso)
    for root in roots:
        ux, uy = root
        arg = coeff*lensg(*root)[0]/(ux - upx)
        if arg > 0:
            freqcaustics.append(c*np.sqrt(arg)/(ax*GHz))
    return np.sort(freqcaustics)

def causPlotter(uxmax, uymax, alp, ax, ay, m = 1000, n = 1000):
    """ Plots caustic surfaces in u-plane and u'-plane. """
    rx = np.linspace(-uxmax, uxmax, 500)
    ry = np.linspace(-uymax, uymax, 500)
    uvec = np.meshgrid(rx, ry)
    coeff = np.array([alp/ax**2, alp/ay**2])
    # if -1.1215 < alp < 0.5 and -1.1215 < beta < 0.5:
    #     print("No caustics")
    #     return
    ucaus = causCurve(uvec, coeff)
    upmax = mapToUp(np.array([uxmax, uymax]), alp, ax, ay)
    fig = plt.figure(figsize=(15, 10))
    grid = gs.GridSpec(2, 2, height_ratios=[3, 1])
    tableax = plt.subplot(grid[1, :])
    ax0, ax1 = plt.subplot(grid[0, 0]), plt.subplot(grid[0, 1])
    cs = ax0.contour(rx, ry, ucaus, levels = [0, np.inf], colors = 'red')
    ax0.set_xlabel(r'$u_x$')
    ax0.set_ylabel(r'$u_y$')
    ax0.set_title('Caustic surfaces in the u-plane')
    ax0.grid()
    paths = cs.collections[0].get_paths()
    uppaths = []
    for p in paths:
        cuvert = np.array(p.vertices).T
        upx, upy = mapToUp(cuvert, alp, ax, ay)
        ax1.plot(upx, upy, color = 'blue')
    if m != 1000 and n != 1000:
        roots = polishedRoots(causticEqSlice, 2*uxmax, 2*uymax, args = (alp, m, n, ax, ay)).T
        ax1.plot(rx, rx*m + n, color = 'green')
        rootupx, rootupy = mapToUp(roots, alp, ax, ay)
        ax1.scatter(rootupx, rootupy, color = 'green')
    ax1.set_xlabel(r"$u'_x$")
    ax1.set_ylabel(r"$u'_y$")
    ax1.set_xlim(-upmax[0], upmax[0])
    ax1.set_ylim(-upmax[1], upmax[1])
    ax1.set_title("Caustic surfaces in the u'-plane")
    ax1.grid()
    
    col_labels = ['Parameter', 'Value'] # Create table with parameter values
    tablevals = [[r'$\alpha \: (AU^2)$', np.around(alp/autocm**2,3)], [r'$a_x \: (AU)$', np.around(ax/autocm, 3)], [r'$a_y \: (AU)$', np.around(ay/autocm, 3)], ['Slope', np.around(m, 2)], ['Offset', n], ['Lens shape', '$%s$' %sym.latex(lensf)]]
    tableax.axis('tight')
    tableax.axis('off')
    table = tableax.table(cellText = np.asarray(tablevals).T, colWidths = [0.045, 0.045, 0.045, 0.045, 0.045, 0.12], rowLabels = col_labels, loc = 'center', cellLoc = 'center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(3., 3.)
    
    plt.show()
    return

# General 2D root finding

def findRoots(func, uxmax, uymax, args = (), N = 500, plot = False):
    """ Finds all real roots of a 2D function in a window of width 2*umax centered at the origin by locating contour plot intersections. func must be a 2D vector function that returns a 2D vector. Do not use to solve lens equation for upvec = [0, 0], negative DM and symmetric lens. Appears to work well, but accuracy is not as good as SciPy's "root" function. """

    def findIntersection(p1, p2):
        v1 = p1.vertices
        v2 = p2.vertices
        poly1 = geometry.LineString(v1)
        poly2 = geometry.LineString(v2)
        intersection = poly1.intersection(poly2)
        # print(intersection)
        try:
            coo = np.ones([100, 2])*1000
            for a in range(len(intersection)):
                coo[a] = np.asarray(list(intersection[a].coords))
        except:
            try:
                coo = np.asarray(list(intersection.coords))
            except:
                pass
        coo = coo[np.nonzero(coo - 1000)]
        return coo

    rx = np.linspace(-uxmax, uxmax, N)
    ry = np.linspace(-uymax, uymax, N)
    uvec = np.meshgrid(rx, ry)
    efunc = func(uvec, *args)
    if not plot:
        cs0 = plt.contour(rx, ry, efunc[0], levels = [0, np.inf], colors = 'red', linewidths = 0)
        cs1 = plt.contour(rx, ry, efunc[1], levels = [0, np.inf], colors = 'blue', linewidths = 0)
    else:
        cs0 = plt.contour(rx, ry, efunc[0], levels = [0, np.inf], colors = 'red')
        cs1 = plt.contour(rx, ry, efunc[1], levels = [0, np.inf], colors = 'blue')
    c0 = cs0.collections[0]
    c1 = cs1.collections[0]
    paths0 = c0.get_paths()
    paths1 = c1.get_paths()
    if plot:
        print('# of paths for contour 1 = ' + str(len(paths0)))
        print('# of paths for contour 2 = ' + str(len(paths1)))
    roots = np.array([])
    for p0 in paths0:
        for p1 in paths1:
            root = findIntersection(p0, p1)
            if len(root) != 0:
                roots = np.append(roots, root)
    roots = np.asarray(roots).flatten().reshape(-1, 2)
    if plot:
        if roots.size != 0:
            plt.scatter(roots.T[0], roots.T[1], color = 'black')
        plt.xlabel(r"$u_x$")
        plt.ylabel(r"$u_y$")
        plt.grid(True)
        plt.show()
    if len(roots) > 1:
        p = np.argsort(roots.T[0])
        roots = roots[p]
    # print(roots)
    return roots

def polishedRoots(func, uxmax, uymax, args = (), plot = False, N = 1000):
    """ Finds roots of func using SciPy's "root" and the roots found by findRoots as initial guesses. """
    raw = findRoots(func, uxmax, uymax, args = args, plot = plot, N = N)
    count = 0
    polished = np.zeros(raw.shape)
    while count < len(raw):
        x0 = raw[count]
        rmess = op.root(func, x0, args = args)
        if rmess.success:
            root = rmess.x
            if not close(root, polished):
                polished[count] = root
        count = count + 1
    return polished

def checkRoot(func, soln, args = ()):
    """ Returns values of func at soln. """
    rem = func(soln, *args)
    return rem

# Root finding along a set of coordinates

def rootFinder(segs, nreal, ncomplex, npoints, ucross, uxmax, uymax, coeff):
    """ Solves the lens equation for every pair of points in the u'-plane contained in upvec, given expected number of real and complex solutions. """
    
    def findFirstComp(ucross, uppoint):
        imguess = np.linspace(-1, 1, 200)
        for guess in imguess:
            croot = op.root(compLensEq, [ucross[0], guess, ucross[1], guess], args=(uppoint, coeff))
            # print(croot)
            # check that the root finder finds the correct complex ray
            if croot.success and np.abs(croot.x[1]) > 1e-6*np.abs(croot.x[0]) and np.abs(croot.x[0] - ucross[0]) < 0.1 and np.abs(croot.x[1]/croot.x[0]) < 1.:
                print([ucross, croot.x])
                croot1 = [croot.x[0] + 1j*croot.x[1], croot.x[2] + 1j*croot.x[3]]
                return croot1
            elif croot.success:  # for debugging purposes
                pass
                # print([ucross, croot.x])
        print('No complex ray found')
        return 0
    
    def findAllComp(roots, seg, pos):
        for j in range(1, npoints):
            prevcomp = roots[j-1][pos]
            tempcomp = op.root(compLensEq, [prevcomp[0].real, prevcomp[0].imag, prevcomp[1].real, prevcomp[1].imag], args=(seg[j], coeff))
            if tempcomp.success:
                roots[j][pos] = np.array([tempcomp.x[0] + 1j*tempcomp.x[1], tempcomp.x[2] + 1j*tempcomp.x[3]])
            else:
                print('Error finding complex root')
                print(seg[j])
                roots[j][pos] = roots[j-1][pos]
        return roots
    
    allroots = []
    for i in range(len(segs)):
        seg = segs[i]
        sreal = polishedRoots(lensEq, 2*uxmax, 2*uymax, args = (seg[npoints/2], coeff)) # starting real roots
        # print(sreal)
        roots = np.zeros([npoints, int(nreal[i] + ncomplex[i]), 2], dtype = complex)
        for j in range(int(nreal[i])):
            roots[npoints/2][j] = sreal[j]
        for j in np.flipud(range(1, npoints/2 + 1)): # find real roots from middle point towards starting point
            for k in range(int(nreal[i])):
                temp = op.root(lensEq, roots[j][k].real, args = (seg[j-1], coeff))
                if temp.success:
                    roots[j-1][k] = temp.x
                else:
                    print('Error 1')
                    print(seg[j])
                    print(temp)
                    print(roots[j].real)
                    roots[j-1][k] = roots[j][k]
        for j in range(npoints/2 + 1, npoints): # find real roots from middle point towards end point
            for k in range(int(nreal[i])):
                temp = op.root(lensEq, roots[j-1][k].real, args = (seg[j], coeff))
                if temp.success:
                    roots[j][k] = temp.x
                else:
                    print('Error 2')
                    print(seg[j])
                    print(temp)
                    print(roots[j-1].real)
                    roots[j][k] = roots[j-1][k]
        if ncomplex[i] > 0:
            p = i - 1
            if i < len(segs)/2: # need to flip
                seg = np.flipud(seg)
                roots = np.flipud(roots)
                p = i
            scomp = findFirstComp(ucross[p], seg[0])
            roots[0][int(nreal[i])] = scomp
            roots = findAllComp(roots, seg, int(nreal[i]))
            if i < len(segs)/2:
                seg = np.flipud(seg) # flip back
                roots = np.flipud(roots)
            if ncomplex[i] == 2:
                if i > len(segs)/2:
                    seg = np.flipud(seg)
                    roots = np.flipud(roots)
                    scomp = findFirstComp(ucross[p + 1], seg[0])
                else:
                    scomp = findFirstComp(ucross[p - 1], seg[0])
                roots[0][int(nreal[i]) + 1] = scomp
                roots = findAllComp(roots, seg, int(nreal[i]) + 1)
                if i > len(segs)/2:
                    roots = np.flipud(roots)
        allroots.append(roots)
    return allroots
