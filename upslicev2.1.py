from fundfunctions import *
from observables import *
from solvers import *
from scipy.special import airy
from scipy.spatial import distance
from kdi import *

# TO DO:

def planeSlice(uxmax, dso, dsl, f, dm, m, n, ax, ay, npoints = 100):
    """ Plots gain, TOA perturbation, and DM perturbation for slice across the u'-plane for given lens parameters, observation frequency, uxmax, slope m and offset n. """

    def lineVert(upxvec, m, n):
        """ Returns list of line vertices. """
        return np.array([upxvec, m*upxvec + n]).T

    def rootFinder(upvec, nsoln):
        """ Solves the lens equation for every pair of points in the u'-plane contained in upvec, given expected number of solutions. """
        roots = np.zeros([len(upvec), nsoln, 2])
        print(nsoln)
        for i in range(len(upvec)):
            count = 0
            root = np.ones([nsoln, 2])*100
            # print(upvec[i])
            temp = polishedRoots(lensEq, 5, 5, args = (upvec[i], coeff))
            #print(temp)
            for j in range(len(temp)):
                root[count] = temp[j]
                count = count + 1
            n = 0.5
            while count != nsoln and n < 1.51: # if not able to find all roots, try different limits and increase N
                print(upvec[i])
                temp = polishedRoots(lensEq, n*uxmax, n*uymax, args = (upvec[i], coeff), N = 1000)
                for j in range(len(temp)):
                    if not close(temp[j], root): # make sure root is not already in the list
                        root[count] = temp[j]
                        count = count + 1
                print(root)
                n = n + mincdist5
            if nsoln > 1:
                p = np.argsort(root.T[0])
                root = root[p]
            roots[i] = root
        return roots

    def obsCalc(roots):
        """ Calculates observables for a list of roots of arbitrary dimensionality. """
        nroots = roots.shape[1]
        print(nroots)
        if nroots == 1: # only one root per coordinate
            obs = np.zeros([3, len(roots)])
            for i in range(len(roots)):
                sgnG = GOgain(roots[i][0], gam, ax, ay, absolute = False)
                obs[0][i] = np.abs(sgnG)
                obs[1][i] = deltatA(roots[i][0], tg0, tdm0, gam, ax, ay)
                obs[2][i] = deltaDMA(roots[i][0], tg0, tdm0, gam, ax, ay, f, sgnG)
        else: # multiple roots per coordinate
            obs = np.zeros([3, nroots, len(roots)])
            for i in range(len(roots)):
                for j in range(nroots):
                    sgnG = GOgain(roots[i][j], gam, ax, ay, absolute = False)
                    obs[0][j][i] = np.abs(sgnG)
                    obs[1][j][i] = deltatA(roots[i][j], tg0, tdm0, gam, ax, ay)
                    obs[2][j][i] = deltaDMA(roots[i][j], tg0, tdm0, gam, ax, ay, f, sgnG)
        return obs

    def realGain(roots, gains):

        def chi(f1, f2):
            return 0.5*(f1 + f2)

        def xi(f1, f2):
            return -lc**(2./3.)*(0.75*(f2 - f1))**(2./3.)

        def abcoeff(f1, f2, U1, U2):
            xi1 = xi(f1, f2)
            air = airy(xi1)
            # print(air)
            a = (-xi1)**mincdist5 * (U1 + 1j*U2) * air[0]
            b = -1j*(-xi1)**-mincdist5 * (U1 - 1j*U2) * air[1]
            return np.array([a, b])

        phis = np.zeros([3, len(roots)])
        fields = np.zeros([3, len(roots)], dtype=complex)
        for j in range(3):
            phi = np.zeros(len(roots))
            field = np.zeros(len(roots), dtype = complex)
            for i in range(len(roots)):
                phi[i] = phiC(roots[i][j], gam, ax, ay)
                field[i] = GOfield(roots[i][j], rF2, lc, ax, ay)
            phis[j] = phi
            fields[j] = field
        # print(phis)
        field1, field2, field3 = fields
        # phi1, phi2, phi3 = phis
        # tp = np.argwhere(np.diff(np.sign(gains[2] - gains[0])) != 0).reshape(-1)[0]
        # # print(tp)
        # phi11, phi12 = np.split(phi1, [tp])
        # phi21, phi22 = np.split(phi2, [tp])
        # phi31, phi32 = np.split(phi3, [tp])
        # U11, U12 = np.split(field1, [tp])
        # U21, U22 = np.split(field2, [tp])
        # U31, U32 = np.split(field3, [tp])
        # # # First caustic
        # a, b = abcoeff(phi12, phi22, U12, U22)
        # # # print([a, b])
        # I1 = pi**0.5*exp(1j*(lc*chi(phi12, phi22) - mincdist5*pi))*(a + b)
        # G1 = np.abs((I1 + U32)**2)
        # # # Second caustic
        # c, d = abcoeff(phi31, phi21, U31, U21)
        # # # print([c, d])
        # I2 = pi**0.5*exp(1j*(lc*chi(phi31, phi21) - mincdist5*pi))*(c + d)
        # G2 = np.abs((I2 + U11)**2)
        # G = pi * lc**(2./3.) * (a**2 + b**2 + c**2 + d**2 + 2*((a*c + b*d)*np.cos(tau1 - tau2) + (b*c - a*d) * np.sin(tau1 - tau2)))
        return np.abs(field1 + field2 + field3)**2

    # Calculate coefficients
    uymax = np.max([uxmax + np.abs(n), m*uxmax + np.abs(n)])
    gam = gamma(dso, dsl, f, dm)
    rF2 = rFsqr(dso, dsl, f)
    lc = lensc(dm, f)
    tg0 = tg0coeff(dso, dsl)
    tdm0 = tdm0coeff(dm, f)
    coeff = gam*np.array([1./ax**2, 1./ay**2])

    # Find caustic intersections
    ucross = polishedRoots(causticEqSlice, 2., 1., args = (gam, m, n, ax, ay))
    ncross = len(ucross)
    # print(ncross)
    upcross = mapToUp(ucross.T, gam, ax, ay)
    p = np.argsort(upcross[0])
    upcross = upcross.T[p]
    ucross = ucross[p]
    print(ucross)
    # print(upcross)
    # print(ucross)

    # Calculate gain at caustics
    causgains = np.zeros(ncross)
    for i in range(ncross):
        causpt = ucross[i]
        causgains[i] = physGainA(causpt, rF2, lc, ax, ay)

    if ncross == 2: # 2 dark regions with 1 image, 1 bright region with 3 images
        # Create slice by segments
        d1upx = np.linspace(-uxmax, upcross[0][0] - mincdist, npoints/2) # d = dark, b = bright
        bupx = np.linspace(upcross[0][0] + mincdist, upcross[1][0] - mincdist, npoints*2)
        d2upx = np.linspace(upcross[1][0] + mincdist, uxmax, npoints/2)
        upxvecs = np.array([d1upx, bupx, d2upx])
        segs = np.array([lineVert(d1upx, m, n), lineVert(bupx, m, n), lineVert(d2upx, m, n)])
        nsolns = np.array([1, 3, 1]) # Number of expected solutions at each segment

        # Solve lens equation at each coordinate
        allroots = []
        for i in range(3):
            roots = rootFinder(segs[i], nsolns[i])
            allroots.append(roots)

        # Calculate gain, TOAs, DMs
        allobs = []
        for roots in allroots:
            obs = obsCalc(roots)
            allobs.append(obs)

    elif ncross == 4:
        if dm > 0: # Positive DM. 3 dark regions with 1 image, 2 bright regions with 3 images.
            # Create slice by segments
            d1upx = np.linspace(-uxmax, upcross[0][0] - 1e-4, npoints) # d = dark, b = bright
            b1upx = np.linspace(upcross[0][0] + 1e-4, upcross[1][0] - 1e-4, npoints)
            d2upx = np.linspace(upcross[1][0] + 1e-4, upcross[2][0] - 1e-4, npoints)
            b2upx = np.linspace(upcross[2][0] + 1e-4, upcross[3][0] - 1e-4, npoints)
            d3upx = np.linspace(upcross[3][0] + 1e-4, uxmax, npoints)
            upxvecs = np.array([d1upx, b1upx, d2upx, b2upx, d3upx])
            segs = np.asarray([lineVert(upxvec, m, n) for upxvec in upxvecs])
            nsolns = np.array([1, 3, 1, 3, 1])

            # Solve lens equation at each coordinate
            allroots = []
            for i in range(5):
                roots = rootFinder(segs[i], nsolns[i])
                allroots.append(roots)

            # Calculate gain, TOAs, DMs
            allobs = []
            for roots in allroots:
                obs = obsCalc(roots)
                allobs.append(obs)

        if dm < 0: # Negative DM. 2 dark regions with 1 image, 2 bright regions with 3 images, 1 bright region with 5 images.
            d1upx = np.linspace(-uxmax, upcross[0][0] - 1e-4, npoints) # d = dark, b = bright
            b1upx = np.linspace(upcross[0][0] + 1e-4, upcross[1][0] - 1e-4, npoints)
            b2upx = np.linspace(upcross[1][0] + 1e-4, upcross[2][0] - 1e-4, npoints)
            b3upx = np.linspace(upcross[2][0] + 1e-4, upcross[3][0] - 1e-4, npoints)
            d2upx = np.linspace(upcross[3][0] + 1e-4, uxmax, npoints)
            upxvecs = np.array([d1upx, b1upx, b2upx, b3upx, d2upx])
            segs = np.asarray([lineVert(upxvec, m, n) for upxvec in upxvecs])
            nsolns = np.array([1, 3, 5, 3, 1])

            # Solve lens equation at each coordinate
            allroots = []
            for i in range(5):
                roots = rootFinder(segs[i], nsolns[i])
                allroots.append(roots)

            # Calculate gain, TOAs, DMs
            allobs = []
            for roots in allroots:
                obs = obsCalc(roots)
                allobs.append(obs)

    # Plots
    fig = plt.figure(figsize = (15, 10))
    grid = gs.GridSpec(3, 2, width_ratios=[4, 1])
    grid.update(hspace = mincdist)
    ax0 = plt.subplot(grid[1:,1])
    ax1 = plt.subplot(grid[0, 1])

    rx = np.linspace(-4, 4, 1000) # Plot caustic surfaces
    ry = np.linspace(-4, 4, 1000)
    uvec = np.meshgrid(rx, ry)
    ucaus = causCurve(uvec, *coeff)
    cs = ax1.contour(rx, ry, ucaus, levels = [0, np.inf], linewidths = 0)
    paths = cs.collections[0].get_paths()
    uppaths = []
    for p in paths:
        cuvert = np.array(p.vertices).T
        upx, upy = mapToUp(cuvert, gam, ax, ay)
        ax1.plot(upx, upy, color = 'blue')
    ax1.plot(rx, rx*m + n, color = 'green')
    ax1.scatter(upcross.T[0], upcross.T[1], color = 'green')
    ax1.set_xlabel(r"$u'_x$")
    ax1.set_ylabel(r"$u'_y$")
    ax1.set_xlim(-uxmax, uxmax)
    # ax1.set_ylim(np.min(upy)*1.5, np.max(upy)*1.5)
    ax1.set_title("Caustic curves")
    ax1.set_aspect('equal', anchor = 'C')
    ax1.grid()

    axes = [plt.subplot(grid[0, 0]), plt.subplot(grid[1, 0]), plt.subplot(grid[2, 0])] # Plot results
    labels = ['G', r'$\Delta t \: (\mu s)$', r'$\Delta DM \: (pc\,cm^{-3})$']
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    if ncross == 2 and dm < 0:
        realG = realGain(allroots[1], allobs[1][0])
        # print(allroots[1])

        axes[0].plot(upxvecs[0], allobs[0][0], color = 'black')
        axes[1].plot(upxvecs[0], allobs[0][1], color = colors[0])
        axes[2].plot(upxvecs[0], allobs[0][2], color = colors[0])

        axes[0].plot(upxvecs[1], allobs[1][0][0], color = colors[0])
        axes[1].plot(upxvecs[1], allobs[1][1][0], color = colors[0])
        axes[2].plot(upxvecs[1], allobs[1][2][0], color = colors[0])
        axes[0].plot(upxvecs[1], allobs[1][0][1], color = colors[1])
        axes[1].plot(upxvecs[1], allobs[1][1][1], color = colors[1])
        axes[2].plot(upxvecs[1], allobs[1][2][1], color = colors[1])
        axes[0].plot(upxvecs[1], allobs[1][0][2], color = colors[2])
        axes[1].plot(upxvecs[1], allobs[1][1][2], color = colors[2])
        axes[2].plot(upxvecs[1], allobs[1][2][2], color = colors[2])

        axes[0].plot(upxvecs[2], allobs[2][0], color = 'black')
        axes[1].plot(upxvecs[2], allobs[2][1], color =  colors[2])
        axes[2].plot(upxvecs[2], allobs[2][2], color = colors[2])

        axes[0].plot(upxvecs[1], realG, color = 'purple')
    else:
        for i in range(len(allobs)):
            obs = allobs[i]
            if len(obs.shape) == 2:
                for j in range(3):
                    axes[j].scatter(upxvecs[i], obs[j], color = colors[j], marker = '.')
            else:
                for j in range(3):
                    for k in range(obs.shape[1]):
                        axes[j].scatter(upxvecs[i], obs[j][k], color = colors[j], marker = '.')
    for j in range(3):
        axes[j].set_ylabel(labels[j])
    axes[0].scatter(upcross.T[0], causgains, color = 'black', marker = '.')
    # axes[0].set_ylim(1e-2, 1e2)
    # axes[0].set_yscale('log')
    # axes[1].set_ylim(1e-9, 5*1e2)
    # axes[1].set_yscale('log')
    # axes[2].set_yscale('symlog')
    axes[2].set_xlabel(r"$u'_x$")

    # Create table
    col_labels = ['Parameter', 'Value']
    if np.abs(dm/pctocm) < 1:
        dmlabel = "{:.2E}".format(Decimal(dm/pctocm))
    else:
        dmlabel = str(dm/pctocm)
    tablevals = [[r'$d_{so} \: (kpc)$', np.around(dso/pctocm/kpc, 2)], [r'$d_{sl} \: (kpc)$', np.around(dsl/pctocm/kpc, 2)], [r'$a_x \: (AU)$', np.around(ax/autocm, 2)], [r'$a_y \: (AU)$', np.around(ay/autocm, 2)], [r'$DM_l \: (pc \, cm^{-3})$', dmlabel], [r"$\nu$ (GHz)", f/GHz], ['Slope', m], ['Offset', n]]
    ax0.axis('tight')
    ax0.axis('off')
    table = ax0.table(cellText = tablevals, colWidths = [mincdist5, mincdist5], colLabels = col_labels, loc = 'center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(3.2, 3.2)

    # plt.figure()
    # cs = plt.contour(rx, ry, ucaus, levels = [0, np.inf])
    # # paths = cs.collections[0].get_paths()
    # # uppaths = []
    # # for p in paths:
    # #     cuvert = np.array(p.vertices).T
    # #     upx, upy = mapToUp(cuvert, gam, ax, ay)
    # #     plt.plot(upx, upy, color = 'blue')
    # r1x = np.concatenate((allroots[0].T[0][0], allroots[1].T[0][0]))
    # r1y = np.concatenate((allroots[0].T[1][0], allroots[1].T[1][0]))
    # r2x = allroots[1].T[0][1]
    # r2y = allroots[1].T[1][1]
    # r3x = np.concatenate((allroots[1].T[0][2], allroots[2].T[0][0]))
    # r3y = np.concatenate((allroots[1].T[1][2], allroots[2].T[1][0]))
    # # r1upx, r1upy = mapToUp(np.array([r1x, r1y]), gam, ax, ay)
    # # r2upx, r2upy = mapToUp(np.array([r2x, r2y]), gam, ax, ay)
    # # r3upx, r3upy = mapToUp(np.array([r3x, r3y]), gam, ax, ay)
    # plt.plot(r1x, r1y, color = 'blue')
    # plt.plot(r2x, r2y, color = 'red')
    # plt.plot(r3x, r3y, color = 'green')
    plt.show()
    return

def planeSliceG(uxmax, uymax, dso, dsl, f, dm, m, n, ax, ay, npoints = 100, gsizex = 2048, gsizey = 2048):
    """ Plots gain for slice across the u'-plane for given lens parameters, observation frequency, uxmax, slope m and offset n. """

    def findComp(upvec):
        """ Finds appropriate (ie. non-exponential) complex ray right next to caustic boundary. """

        def chooseRay(ucross):
            imguess = np.linspace(0, 1, 50)
            for guess in imguess:
                croot = op.root(compLensEq, [ucross[0], guess, ucross[1], guess], args = (upvec, coeff))
                if croot.success and np.abs(croot.x[1]) > 0.1*croot.x[0] and croot.x[0] - ucross[0] < 0.3: # check that the root finder finds the correct complex ray
                    print([ucross, croot.x])
                    croot1 = [croot.x[0] + 1j*croot.x[1], croot.x[2] + 1j*croot.x[3]]
                    field1 = GOfieldA(croot1, rF2, lc, ax, ay)
                    croot2 = np.conj(croot1)
                    field2 = GOfieldA(croot2, rF2, lc, ax, ay)
                    if np.abs(field1)**2 < np.abs(field2)**2:
                        return croot1
                    else:
                        return croot2
            print('No complex ray found')
            return 0

        if len(upcross) == 2:
            if upvec[0] < upcross[0][0]:
                return chooseRay(ucross[0])
            else:
                return chooseRay(ucross[1])
        
        elif len(upcross) == 4:
            if dm < 0:
                if upvec[0] < upcross[0][0]:
                    return chooseRay(ucross[0])
                else:
                    return chooseRay(ucross[3])
            else:
                if upvec[0] < upcross[0][0]:
                    return chooseRay(ucross[0])
                elif upvec[0] > upcross[3][0]:
                    return chooseRay(ucross[3])
                else:
                    cond = (upvec[0] - upcross[1][0] > 2*mincdist) # need to distinguish between point right after second caustic and point right before third caustic
                    if cond:
                        return chooseRay(ucross[2])
                    else:
                        return chooseRay(ucross[1])


    def lineVert(upxvec, m, n):
        """ Returns list of line vertices. """
        return np.array([upxvec, m*upxvec + n]).T

    def rootFinder(upvec, nsoln):
        """ Solves the lens equation for every pair of points in the u'-plane contained in upvec, given expected number of solutions. """

        def oneRootHelper(upvec, middle = False):
            if middle: # need two complex rays
                roots = np.zeros([len(upvec), 3, 2], dtype = complex)
                roots[0][0] = polishedRoots(lensEq, 5, 5, args = (upvec[0], coeff)) # first real root
                roots[0][1] = findComp(upvec[0]) # first complex root
                for i in range(1, len(upvec)):
                    tempreal = op.root(lensEq, roots[i-1][0].real, args = (upvec[i], coeff)) # tries to find root by using previous root as guess
                    prevcomp = roots[i-1][1]
                    tempcomp = op.root(compLensEq, [prevcomp[0].real, prevcomp[0].imag, prevcomp[1].real, prevcomp[1].imag], args = (upvec[i], coeff))
                    if tempreal.success and tempcomp.success:
                        roots[i][0] = tempreal.x
                        roots[i][1] = np.array([tempcomp.x[0] + 1j*tempcomp.x[1], tempcomp.x[2] + 1j*tempcomp.x[3]])
                    else:
                        print('Error')
                        print(upvec[i])
                        print([tempreal, tempcomp])

                upvec = np.flipud(upvec) # find second complex ray by iterating from other side
                roots = np.flipud(roots)
                roots[0][2] = findComp(upvec[0])
                for i in range(1, len(upvec)):
                    prevcomp = roots[i-1][2]
                    tempcomp = op.root(compLensEq, [prevcomp[0].real, prevcomp[0].imag, prevcomp[1].real, prevcomp[1].imag], args = (upvec[i], coeff))
                    if tempcomp.success:
                        roots[i][2] = np.array([tempcomp.x[0] + 1j*tempcomp.x[1], tempcomp.x[2] + 1j*tempcomp.x[3]])
                    else:
                        print('Error')
                        print(upvec[i])
                        print([tempreal, tempcomp])
                roots = np.flipud(roots)

            else: # only need one complex ray
                roots = np.zeros([len(upvec), 2, 2], dtype = complex)
                roots[0][0] = polishedRoots(lensEq, 5, 5, args = (upvec[0], coeff)) # first real root
                roots[0][1] = findComp(upvec[0]) # first complex root
                for i in range(1, len(upvec)):
                    tempreal = op.root(lensEq, roots[i-1][0].real, args = (upvec[i], coeff)) # tries to find root by using previous root as guess
                    prevcomp = roots[i-1][1]
                    tempcomp = op.root(compLensEq, [prevcomp[0].real, prevcomp[0].imag, prevcomp[1].real, prevcomp[1].imag], args = (upvec[i], coeff))
                    if tempreal.success and tempcomp.success:
                        roots[i][0] = tempreal.x
                        roots[i][1] = np.array([tempcomp.x[0] + 1j*tempcomp.x[1], tempcomp.x[2] + 1j*tempcomp.x[3]])
                    else:
                        print('Error')
                        print(upvec[i])
                        print([tempreal, tempcomp])
            return roots

        print(nsoln)
        
        if nsoln == 1:
             # one real root and one/two complex roots
             # check whether its necessary to iterate left to right or right to left
            if upvec[-1][0] < upcross[0][0]: # greatest upx-value of upvec is less than upx-value of first caustic, so iterate right to left
                upvec = np.flipud(upvec)
                roots = oneRootHelper(upvec)
                roots = np.flipud(roots) # return to left-right ordering
            elif ncross == 4 and upvec[0][0] > upcross[1][0] and upvec[-1][0] < upcross[2][0]: # middle region, need two complex rays
                roots = oneRootHelper(upvec, middle = True)
            else:
                roots = oneRootHelper(upvec)
        else: # three or five real roots
            roots = np.zeros([len(upvec), nsoln, 2])
            first = polishedRoots(lensEq, 5, 5, args = (upvec[0], coeff)) # finds first root
            p = np.argsort(first.T[0])
            roots[0] = first[p]
            # print(roots[0])
            for i in range(1, len(upvec)):
                root = np.ones([nsoln, 2])*100
                err = False
                for j in range(nsoln):
                    temp = op.root(lensEq, roots[i-1][j], args = (upvec[i], coeff))
                    # print(temp)
                    if temp.success: # and not close(temp.x, root)
                        root[j] = temp.x
                    else:
                        err = True
                        print(upvec[i])
                        print(temp)
                        break
                if err:
                    root = polishedRoots(lensEq, 5, 5, args = (upvec[i], coeff))
                    print(root)
                else:
                    roots[i] = root
        return roots

    def fieldCalc(roots):
        """ Calculates field for a list of roots of arbitrary dimensionality. """
        nroots = roots.shape[1]
        if nroots == 1: # only one root per coordinate
            fields = np.zeros([2, len(roots)], dtype = complex)
            for i in range(len(roots)):
                fields[0][i] = GOfieldB(roots[i][0], rF2, lc, ax, ay)
                # fields[1][i] = physField(roots[i][0], rF2, lc, ax, ay)
        else: # multiple roots per coordinate
            fields = np.zeros([2, nroots, len(roots)], dtype = complex)
            for i in range(len(roots)):
                for j in range(nroots):
                    fields[0][j][i] = GOfieldB(roots[i][j], rF2, lc, ax, ay)
                    # fields[1][j][i] = physField(roots[i][j], rF2, lc, ax, ay)
        return fields

    # Calculate coefficients
    rF2 = rFsqr(dso, dsl, f)
    uF2x, uF2y = rF2*np.array([1./ax**2, 1./ay**2])
    lc = lensc(dm, f)
    gam  = rF2*lc
    coeff = gam*np.array([1./ax**2, 1./ay**2])

    # Calculate caustic intersections
    ucross = polishedRoots(causticEqSlice, uxmax, uymax, args = (gam, m, n, ax, ay))
    ncross = len(ucross)
    upcross = mapToUp(ucross.T, gam, ax, ay)
    p = np.argsort(upcross[0])
    upcross = upcross.T[p]
    ucross = ucross[p]

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

    mincdist = 0.15*(1+m**2)**-0.5 # probably need to justify this distance using physical arguments

    if ncross == 2: # 2 dark regions with 1 image, 1 bright region with 3 images
        # Create slice by segments
        d1upx = np.linspace(xmin, upcross[0][0] - mincdist, npoints) # d = dark, b = bright
        bupx = np.linspace(upcross[0][0] + mincdist, upcross[1][0] - mincdist, npoints)
        d2upx = np.linspace(upcross[1][0] + mincdist, xmax, npoints)
        czone1 = np.linspace(upcross[0][0] - mincdist, upcross[0][0] + mincdist, 100) # set up caustic zones
        czone2 = np.linspace(upcross[1][0] - mincdist, upcross[1][0] + mincdist, 100)
        upxvecs = np.array([d1upx, bupx, d2upx])
        cvecs = np.array([czone1, czone2])
        segs = np.asarray([lineVert(upx, m, n) for upx in upxvecs])
        nsolns = np.array([1, 3, 1]) # Number of expected solutions at each segment

    elif ncross == 4:
        if dm > 0: # Positive DM. 3 dark regions with 1 image, 2 bright regions with 3 images.
            # Create slice by segments
            d1upx = np.linspace(xmin, upcross[0][0] - mincdist, npoints/2) # d = dark, b = bright
            b1upx = np.linspace(upcross[0][0] + mincdist, upcross[1][0] - mincdist, npoints)
            d2upx = np.linspace(upcross[1][0] + mincdist, upcross[2][0] - mincdist, npoints)
            b2upx = np.linspace(upcross[2][0] + mincdist, upcross[3][0] - mincdist, npoints)
            d3upx = np.linspace(upcross[3][0] + mincdist, xmax, npoints/2)
            upxvecs = np.array([d1upx, b1upx, d2upx, b2upx, d3upx])
            segs = np.asarray([lineVert(upx, m, n) for upx in upxvecs])
            nsolns = np.array([1, 3, 1, 3, 1])

        if dm < 0: # Negative DM. 2 dark regions with 1 image, 2 bright regions with 3 images, 1 bright region with 5 images.
            d1upx = np.linspace(xmin, upcross[0][0] - mincdist, npoints) # d = dark, b = bright
            b1upx = np.linspace(upcross[0][0] + mincdist, upcross[1][0] - mincdist, npoints)
            b2upx = np.linspace(upcross[1][0] + mincdist, upcross[2][0] - mincdist, npoints)
            b3upx = np.linspace(upcross[2][0] + mincdist, upcross[3][0] - mincdist, npoints)
            d2upx = np.linspace(upcross[3][0] + mincdist, xmax, npoints)
            upxvecs = np.array([d1upx, b1upx, b2upx, b3upx, d2upx])
            segs = np.asarray([lineVert(upx, m, n) for upx in upxvecs])
            nsolns = np.array([1, 3, 5, 3, 1])

    
    # Solve lens equation at points far from caustics
    allroots = []
    for i in range(len(nsolns)):
        roots = rootFinder(segs[i], nsolns[i])
        allroots.append(roots)

    # Calculate fields far from caustics
    allfields = []
    for roots in allroots:
        fields = fieldCalc(roots)
        allfields.append(fields)
    
    # Solve lens equation close to caustics
    croots = []
    for i in range(len(cvecs)):
        cvec = cvecs[i]
        polishedRoots(lensEq, 5, 5, args = (upvec[0], coeff))

    # Calculate gains
    G1 = [] # GO gains
    for i in range(len(nsolns)):
        fields = allfields[i][0]
        G = np.zeros(len(fields.T))
        for j in range(len(fields.T)):
            G[j] = np.abs(np.sum(fields.T[j]))**2
        G1.append(G)

    # causgains = np.zeros(ncross)
    # for i in range(ncross):
    #     causpt = ucross[i]
    #     causgains[i] = physGainA(causpt, rF2, lc, ax, ay)

    # KDI
    rx = np.linspace(-2*uxmax, 2*uxmax, gsizex)
    ry = np.linspace(-2*uymax, 2*uymax, gsizey)
    dux = 4*uxmax/gsizex
    duy = 4*uymax/gsizey
    extent = (-uxmax, uxmax, -uymax, uymax)
    ux, uy = np.meshgrid(rx, ry)
    # lens = np.pad(lensPhase(ux, uy, lc), ((0, 0), (0, 1028)), 'constant', constant_values = 0)
    lens = lensPhase(ux, uy, lc)
    lensfft = fft2(lens)
    # geo =  np.pad(geoPhase(ux, uy, uF2x, uF2y), ((0, 0), (0, 2048)), 'constant', constant_values = 0)
    geo = geoPhase(ux, uy, uF2x, uF2y)
    geofft = fft2(geo)
    fieldfft = lensfft*geofft
    field = fftshift(ifft2(fieldfft))
    # soln = resize(soln, (gsizex, gsizey))
    # field = fftconvolve(geo, lens, mode = 'same')
    soln = np.abs((dux*duy*field)**2/(4*pi**2*uF2x*uF2y))
    soln = soln[int(0.25*gsizex):int(0.75*gsizex), int(0.25*gsizey):int(0.75*gsizey)]
    # print(soln)

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
    ucaus = causCurve([ux, uy], uF2x*lc, uF2y*lc)
    cs = plt.contour(np.linspace(-uxmax, uxmax, gsizex), ry, ucaus, levels = [0, np.inf], linewidths = 0)
    paths = cs.collections[0].get_paths()
    uppaths = []
    for p in paths:
        cuvert = np.array(p.vertices).T
        upx, upy = mapToUp(cuvert, gam, ax, ay)
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
    for i in range(len(nsolns)):
        ax1.plot(upxvecs[i], G1[i], color = 'red') # Plot GO gain
    # ax1.plot(upxvec, G2, color = 'green')
    # ax1.scatter(upcross.T[0], causgains, color = 'black')
    ax1.set_ylim(-mincdist, np.max(G) + 1.5)
    ax1.set_xlim(np.min(rx2), np.max(rx2))
    ax1.set_xlabel(r"$u'_x$")
    ax1.set_ylabel('G')
    ax1.set_title("Slice Gain")


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
