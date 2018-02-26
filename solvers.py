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
    """ Evaluates the 2D lens equation. coeff = gamma*[1/ax**2, 1/ay**2]. """
    ux, uy = uvec
    upx, upy = upvec
    funcg = np.array([gauss10(ux, uy), gauss01(ux, uy)])
    return np.array([ux + coeff[0]*funcg[0] - upx, uy + coeff[1]*funcg[1] - upy])

def close(myarr, list_arrays):
    """ Determines whether array is inside another list of arrays. """
    return next((True for elem in list_arrays if elem.size == myarr.size and np.allclose(elem, myarr, atol = 1e-3)), False)

# Lens equation solvers

def solveLensA(upvec, coeff, numguess = 100):
    """ Attempts to solve lens equation using SciPy's "root" function blindly, without knowing the number of solutions there should be. coeff = gamma*[1/ax**2, 1/ay**2]. """
    l = np.linspace(-2, 2, numguess)
    guesses = np.zeros([numguess,2])
    for n in range(len(l)):
        guesses[n] = l[n]*upvec
    count = 0
    usol = np.zeros([10, 2])
    for x0 in guesses:
        rmess = op.root(lensEq, x0, args = (upvec, coeff))
        if rmess.success:
            root = rmess.x
            if (count == 0 or not close(root, usol)):
                usol[count] = root
                count = count + 1
    return np.trim_zeros(usol.flatten()).reshape(-1,2)

def solveLensB(upvec, coeff, nsoln, numguess = 100):
    """ Attempts to solve lens equation using Scipy's "root" function based on the number of expected solutions. coeff = gamma*[1/ax**2, 1/ay**2].  Might need more work for highly asymmetric lenses. The case of nsoln = 5 needs more work. """
    if nsoln == 1:
        count = 0
        guesses = np.array([np.zeros(2), 0.5*upvec, upvec, 2*upvec, 2.5*upvec, 3*upvec, 3.5*upvec, 4*upvec, 4.5*upvec, 5*upvec, 5.5*upvec, 6*upvec, 6.5*upvec, 7*upvec, 7.5*upvec])
        for x0 in guesses:
            rmess = op.root(lensEq, x0, args = (upvec, coeff))
            if rmess.success:
                return [rmess.x]
            else:
                print("Problem! No solution found")
                return [[]]
    elif nsoln == 3:
        usol = np.zeros([3, 2])
        if np.sum(coeff) < 0: # positive DM
            guesses = np.array([upvec, np.zeros(2), 4*upvec, 3.5*upvec, 3*upvec, 2.5*upvec, 2*upvec, 1.5*upvec, 0.5*upvec, 0.8*upvec, 0.3*upvec])
            count = 0
            n = 0
            while count < 3 and n < len(guesses):
                x0 = guesses[n]
                rmess = op.root(lensEq, x0, args = (upvec, coeff))
                if rmess.success:
                    root = rmess.x
                    if (count == 0 or not close(root, usol)):
                        usol[count] = root
                        count = count + 1
                n = n + 1
        else:  # negative DM
            guesses1 = np.array([-2*upvec, -upvec, -1.5*upvec, -0.5*upvec, [0., 0.], [-1., -1.], [-2., -2.], -0.1*upvec, [-0.1, -0.1], -3*upvec, -2.5*upvec, -0.1*upvec, -0.2*upvec, -0.01*upvec])
            count = 0
            n = 0
            while count < 2 and n < len(guesses1):
                x0 = guesses1[n]
                rmess = op.root(lensEq, x0, args = (upvec, coeff))
                if rmess.success:
                    root = rmess.x
                    if (count == 0 or not close(root, usol)):
                        usol[count] = root
                        count = count + 1
                n = n + 1
            n = 0
            guesses2 = np.array([2*upvec, upvec, 1.5*upvec, 0.5*upvec, [0., 0.], [1., 1.], [2., 2.], 3*upvec, 2.5*upvec, 0.1*upvec, 0.2*upvec, 0.01*upvec])
            while count < 3 and n < len(guesses2):
                x0 = guesses2[n]
                rmess = op.root(lensEq, x0, args = (upvec, coeff))
                if rmess.success:
                    root = rmess.x
                    if count == 0 or not close(root, usol):
                        usol[count] = root
                        count = count + 1
                n = n + 1
    elif nsoln == 5:
        l = np.linspace(-2, 2, numguess)
        guesses = np.zeros([numguess, 2])
        for n in range(len(l)):
            guesses[n] = l[n]*upvec
        count = 0
        n = 0
        usol = np.zeros([5, 2])
        while count < 5 and n < numguess:
            x0 = guesses[n]
            rmess = op.root(lensEq, x0, args = (upvec, coeff))
            if rmess.success:
                root = rmess.x
                if (count == 0 or not close(root, usol)):
                    usol[count] = root
                    count = count + 1
            n = n + 1
    if count == 3 or count == 5:
        usum = np.sum(np.abs(usol), axis = 1)
        p = np.argsort(usum)
        usol = usol[p]
    return usol

# Caustic finders

def causCurve(uvec, alp, beta):
    gxx, gyy, gxy = gauss20(*uvec), gauss02(*uvec), gauss11(*uvec)
    return 1 + alp*gxx + beta*gyy - alp*beta*(gxy**2 - gxx*gyy)

def causticEqSlice(uvec, gamma, m, n, ax, ay):
    """ Evaluates the caustic equations for a slice across the u'-plane for given ux, uy, slope m and offset n, and lens parameters. Input in cgs units. """
    ux, uy = uvec
    eq1 = uy - m*ux - n + 2*gamma*gauss(ux, uy)*(m*ux/ax**2 - uy/ay**2)
    gxx = gauss20(ux, uy)
    gyy = gauss02(ux, uy)
    gxy = gauss11(ux, uy)
    eq2 = 1 + gamma*(ay**2*gxx + ax**2*gyy)/(ay*ax)**2 - gamma**2*(gxy**2 - gxx*gyy)/(ay*ax)**2
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

def causPlotter(uxmax, uymax, gam, ax, ay, m = 1000, n = 1000):
    """ Plots caustic surfaces in u-plane and u'-plane. """
    rx = np.linspace(-uxmax, uxmax, 1000)
    ry = np.linspace(-uymax, uymax, 1000)
    uvec = np.meshgrid(rx, ry)
    alp, beta = gam/ax**2, gam/ay**2
    # if -1.1215 < alp < 0.5 and -1.1215 < beta < 0.5:
    #     print("No caustics")
    #     return
    ucaus = causCurve(uvec, alp, beta)
    upmax = mapToUp(np.array([uxmax, uymax]), gam, ax, ay)
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
        upx, upy = mapToUp(cuvert, gam, ax, ay)
        axarr[1].plot(upx, upy, color = 'blue')
    if m != 1000 and n != 1000:
        roots = polishedRoots(causticEqSlice, uxmax, uymax, args = (gam, m, n, ax, ay)).T
        axarr[1].plot(rx, rx*m + n, color = 'green')
        rootupx, rootupy = mapToUp(roots, gam, ax, ay)
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

def polishedRoots(func, uxmax, uymax, args = (), plot = False, N = 500):
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
    check = np.zeros(soln.shape)
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
