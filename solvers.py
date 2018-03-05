from fundfunctions import *
import matplotlib.pyplot as plt
import scipy.optimize as op
from shapely import geometry

def compLensEq(uvec, upvec, coeff):
    """ Evaluates the 2D lens equation with u a complex vector. """
    uxr, uxi, uyr, uyi = uvec
    upx, upy = upvec
    funcg = np.array([gauss10(uxr + 1j*uxi, uyr + 1j*uyi), gauss01(uxr + 1j*uxi, uyr + 1j*uyi)])
    eq = np.array([uxr + 1j*uxi + coeff[0]*funcg[0] - upx, uyr + 1j*uyi + coeff[1]*funcg[1] - upy])
    return [eq[0].real, eq[0].imag, eq[1].real, eq[1].imag]

def lensEq(uvec, upvec, coeff):
    """ Evaluates the 2D lens equation. coeff = alp*[1/ax**2, 1/ay**2]. """
    ux, uy = uvec
    upx, upy = upvec
    funcg = np.array([gauss10(ux, uy), gauss01(ux, uy)])
    return np.array([ux + coeff[0]*funcg[0] - upx, uy + coeff[1]*funcg[1] - upy])

def close(myarr, list_arrays):
    """ Determines whether array is inside another list of arrays. """
    return next((True for elem in list_arrays if elem.size == myarr.size and np.allclose(elem, myarr, atol = 1e-3)), False)

# Caustic finders

def causCurve(uvec, coeff):
    g20, g02, g11 = gauss20(*uvec), gauss02(*uvec), gauss11(*uvec)
    return 1 + coeff[0]*g20 + coeff[1]*g02 - coeff[0]*coeff[1]*(g11**2 - g20*g02)

def causticEqSlice(uvec, alp, m, n, ax, ay):
    """ Evaluates the caustic equations for a slice across the u'-plane for given ux, uy, slope m and offset n, and lens parameters. Input in cgs units. """
    ux, uy = uvec
    eq1 = uy - m*ux - n + 2*alp*gauss(ux, uy)*(m*ux/ax**2 - uy/ay**2)
    g20 = gauss20(ux, uy)
    g02 = gauss02(ux, uy)
    g11 = gauss11(ux, uy)
    eq2 = 1 + alp*(ay**2*g20 + ax**2*g02)/(ay*ax)**2 - alp**2*(g11**2 - g20*g02)/(ay*ax)**2
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
        arg = coeff*gauss10(ux, uy)/(ux - upx)
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
    f, axarr = plt.subplots(1, 2, figsize = (16, 8))
    cs = axarr[0].contour(rx, ry, ucaus, levels = [0, np.inf], colors = 'red')
    axarr[0].set_xlabel(r'$u_x$')
    axarr[0].set_ylabel(r'$u_y$')
    axarr[0].set_title('Caustic surfaces in the u-plane')
    axarr[0].grid()
    paths = cs.collections[0].get_paths()
    uppaths = []
    for p in paths:
        cuvert = np.array(p.vertices).T
        upx, upy = mapToUp(cuvert, alp, ax, ay)
        axarr[1].plot(upx, upy, color = 'blue')
    if m != 1000 and n != 1000:
        roots = polishedRoots(causticEqSlice, uxmax, uymax, args = (alp, m, n, ax, ay)).T
        axarr[1].plot(rx, rx*m + n, color = 'green')
        rootupx, rootupy = mapToUp(roots, alp, ax, ay)
        axarr[1].scatter(rootupx, rootupy, color = 'green')
    axarr[1].set_xlabel(r"$u'_x$")
    axarr[1].set_ylabel(r"$u'_y$")
    axarr[1].set_xlim(-upmax[0], upmax[0])
    axarr[1].set_ylim(-upmax[1], upmax[1])
    axarr[1].set_title("Caustic surfaces in the u'-plane")
    axarr[1].grid()
    plt.show()
    return

# General 2D root finding

def findRoots(func, uxmax, uymax, args = (), N = 500, plot = False):
    """ Finds all roots of a 2D function in a window of width 2*umax centered at the origin by locating contour plot intersections. func must be a 2D vector function that returns a 2D vector. Do not use to solve lens equation for upvec = [0, 0], negative DM and symmetric lens. Appears to work well, but accuracy is not as good as SciPy's "root" function. """

    def findIntersection(p1, p2):
        v1 = p1.vertices
        v2 = p2.vertices
        poly1 = geometry.LineString(v1)
        poly2 = geometry.LineString(v2)
        intersection = poly1.intersection(poly2)
        try:
            coo = np.ones([5, 2])*1000
            for a in range(len(intersection)):
                coo[a] = np.asarray(list(intersection[a].coords))
        except:
            coo = np.asarray(list(intersection.coords))
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
    if plot and roots.size != 0:
        if roots.size != 0:
            plt.scatter(roots.T[0], roots.T[1], color = 'black')
        plt.xlabel(r"$u_x$")
        plt.ylabel(r"$u_y$")
        # plt.xlim(-uxmax, uxmax)
        # plt.ylim(-uymax, uymax)
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

# Numerical derivatives

@jit(nopython=True)
def der(func, uvec, order, args = ()):
    """ Returns numerical derivative of order = [o1, o2] of function func at point uvec. Max o1 + o2 = 3. """
    o1, o2 = order
    ux, uy = uvec
    hx, hy = 1e-6, 1e-6
    if o1 == 1 and o2 == 0:
        ans = (func(ux + hx, uy, *args) - func(ux, uy, *args))/hx
    elif o1 == 0 and o2 == 1:
        ans = (func(ux, uy + hy, *args) - func(ux, uy, *args))/hy
    elif o1 == 1 and o2 == 1:
        ans = (func(ux + hx, uy + hy, *args) - func(ux + hx, uy - hy, *args) - func(ux - hx, uy + hy, *args) + func(ux - hx, uy - hy, *args))/(4.*hx*hy)
    elif o1 == 2 and o2 == 0:
        ans = (func(ux + 2*hx, uy, *args) - 2*func(ux, uy, *args) + func(ux - 2*hx, uy, *args))/(4*hx**2)
    elif o1 == 0 and o2 == 2:
        ans = (func(ux, uy + 2*hy, *args) - 2*func(ux, uy, *args) + func(ux, uy - 2*hy, *args))/(4*hy**2)
    return ans
