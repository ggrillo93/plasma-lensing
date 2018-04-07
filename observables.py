from fundfunctions import *
from scipy.special import gamma as gfunc

# Phase
def phi(uvec, rF2, lc, ax, ay):
    """ Returns the phase at a stationary point. """
    ux, uy = uvec
    grad = lensg(ux, uy)
    return 0.5*rF2*lc**2*((grad[0]/ax)**2 + (grad[1]/ay)**2) + lc*lensfun(*uvec)

# Field
def GOfield(uvec, rF2, lc, ax, ay):
    """ Returns the elements of the geometrical optics field: the amplitude and the phase, including the phase shift as determined by the sign of the derivatives. """
    ux, uy = uvec
    alp = rF2*lc
    psi20, psi02, psi11 = lensh(ux, uy)
    phi20 = ax**2/rF2 + lc*psi20
    phi02 = ay**2/rF2 + lc*psi02
    phi11 = lc*psi11
    sigma = np.sign(phi02)
    H = phi20*phi02 - phi11**2
    delta = np.sign(H)
    amp = (ax*ay/rF2)*np.abs(H)**-0.5
    phase = phi(uvec, rF2, lc, ax, ay)
    pshift = pi*(delta + 1)*sigma*0.25
    return np.array([amp, phase, pshift])
    
# TOA perturbation
def deltat(uvec, tg0, tdm0, alp, ax, ay):
    """ Returns TOA perturbation in us at a stationary point. Coefficients: tg0 = dso/(2*c*dsl*dlo), tdm0 = c*re*dm/(2*pi*f**2). """
    ux, uy = uvec
    grad = lensg(ux, uy)
    return 1e6*(tg0*alp**2*((grad[0]/ax)**2 + (grad[1]/ay)**2) + tdm0*lensfun(*uvec))
    
# Field construction and helper functions

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


def obsCalcFreq(func, roots, nroots, npoints, ansdim, fvec, args=()):
    """ Calculates observable using observable function func for a list of roots of arbitrary dimensionality. Returns multidimensional array with shape [nroots, ansdim, npoints]. """
    if ansdim == 1:
        obs = np.zeros([nroots, npoints], dtype=complex)
        for i in range(npoints):
            for j in range(nroots):
                obs[j][i] = func(roots[i][j], *args)
    else:
        obs = np.zeros([nroots, ansdim, npoints], dtype=complex)
        for i in range(npoints):
            for j in range(nroots):
                ans = func(roots[i][j], *args)
                for k in range(ansdim):
                    obs[j][k][i] = ans[k]
    return obs

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
            # print([mroot1, mroot2])
            # print([merge[0][1], merge[1][1]])
            nmroots1 = list(set(range(realn)) - set(mroot1)) # indices of non merging roots at one end
            nmroots2 = list(set(range(realn)) - set(mroot2)) # indices of non merging roots at other end
            if merge[0][1] < 0.4 and merge[1][1] < 0.4: # case 1: real root merging at both ends
                # print('Double root merging')
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
                # print('Root merging at first end')
                A1, phi1 = fields[mroot1[0]][:2]
                A2, phi2 = fields[mroot1[1]][:2]
                if i <= nzones/2.:
                    amerge = bright(A1, A2, phi1, phi2, sigs[p-1])
                else:
                    amerge = bright(A1, A2, phi1, phi2, sigs[p])
                anonm = np.zeros(npoints, dtype = complex)
                for index in nmroots1:
                    anonm = anonm + constructField(*fields[index]) # sum of fields not involved in merging
                areal = amerge + anonm
            elif merge[0][1] > 0.4 and merge[1][1] < 0.4: # case 3: real root merging at second end only
                # print('Root merging at second end')
                A1, phi1 = fields[mroot2[0]][:2]
                A2, phi2 = fields[mroot2[1]][:2]
                if i <= nzones/2:
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

# Momentarily useless stuff
# # @jit(nopython=True)
# def GOAmplitude(uvec, rF2, lc, ax, ay):
#     """ Returns the geometrical optics amplitude. """
#     ux, uy = uvec
#     alp = rF2*lc
#     phi20 = ax**2/rF2 + lc*gauss20(ux, uy)
#     phi02 = ay**2/rF2 + lc*gauss02(ux, uy)
#     phi11 = lc*gauss11(ux, uy)
#     H = phi20*phi02 - phi11**2
#     ans = (ax*ay/rF2)*np.abs(H)**-0.5
#     return ans



# def GOfieldB(uvec, rF2, lc, ax, ay):
#     """ Returns the geometrical optics field, including the second order term. """
#     # Stamnes 1986, Dingle 1973
#     ux, uy = uvec
#     alp = rF2*lc
#     phi20 = ax**2/rF2 + lc*gauss20(ux, uy)
#     phi02 = ay**2/rF2 + lc*gauss02(ux, uy)
#     phi11 = lc*gauss11(ux, uy)
#     H = phi20*phi02 - phi11**2
#     phi21, phi12, phi30, phi03, phi22, phi31, phi13, phi40, phi04 = lc*np.array([gauss21(ux, uy), gauss12(ux, uy), gauss30(ux, uy), gauss03(ux, uy), gauss22(ux, uy), gauss31(ux, uy), gauss13(ux, uy), gauss40(ux, uy), gauss04(ux, uy)])
#     A = phi02**3*(5*phi30**2 - 3*phi20*phi40)
#     B = phi20**3*(5*phi03**2 - 3*phi02*phi04)
#     C = 3*phi20*phi02*(phi20*(2*phi21*phi03 + 3*phi12**2) + phi02*(2*phi12*phi30 + 3*phi21**2) - 2*phi20*phi02*phi22)
#     D = -6*phi11*(phi20**2*(5*phi12*phi03 - 2*phi02*phi13) + phi02**2*(5*phi21*phi30 - 2*phi20*phi31) + phi20*phi02*(9*phi12*phi21 + phi30*phi03))
#     E = 3*phi11**2*(phi20*(8*phi21*phi03 + 12*phi12**2 + phi20*phi04) + phi02*(8*phi12*phi30 + 12*phi21**2 + phi02*phi40) - 2*phi20*phi02*phi22)
#     F = -4*phi11**3*(3*phi20*phi13 + 3*phi02*phi31 + 9*phi12*phi21 + phi30*phi03)
#     G = 12*phi11**4*phi22
#     q2 = (A + B + C + D + E + F + G)/(24.*H**3)
#     sigma = np.sign(phi02)
#     delta = np.sign(H)
#     return ax*ay/(rF2*np.abs(H)**0.5) * exp(1j*(phi(uvec, rF2, lc, ax, ay) + pi*(delta + 1)*sigma*0.25)) * (1. + 1j*q2)

# @jit(nopython=True)
# def physField(uvec, rF2, lc, ax, ay):
#     """ Returns an approximation of the field at the caustic, using the formula from Cooke 1982. """
#     ux, uy = uvec
#     alp = rF2*lc
#     phi20, phi02 = ax**2/rF2 + lc*gauss20(ux, uy), ay**2/rF2 + lc*gauss02(ux, uy)
#     phi11 = lc*gauss11(ux, uy)
#     phi30 = lc*gauss30(ux, uy)
#     phi21 = lc*gauss21(ux, uy)
#     phi12 = lc*gauss12(ux, uy)
#     phi03 = lc*gauss03(ux, uy)
#     B = phi20**3*phi03 - 3*phi20**2*phi11*phi12 + 3*phi20*phi11**2*phi21 - phi11**3*phi30
#     U = ax*ay/(2*pi*rF2) * exp(1j*(phi(uvec, rF2, lc, ax, ay) + 0.25*pi*np.sign(phi20))) * 2.**(5./6.) * pi**(1./3.) * gfunc(1./3.) * np.abs(phi20)**0.5/(3.**(1./6.) * np.abs(B)**(1./3.))
#     return U

# # DM perturbation
# def deltaDMA(uvec, tg0, tdm0, alp, ax, ay, f, sgnG):
#     """ Returns DM perturbation as a function of root of lens equation uvec, lens parameters, coefficients tg0 = dso/(2*c*dsl*dlo), tdm0 = c*re*dm/(2*pi*f**2), alpma = -c**2*re*dsl*dlo*dm/(2*pi*f**2*dso), and signed value of G. """
#     g = gauss(*uvec)
#     gx, gy = gauss10(*uvec), gauss01(*uvec)
#     gxx, gyy = gauss20(*uvec), gauss02(*uvec)
#     gxy = gauss11(*uvec)
#     coeff = -2*alp*sgnG/((ax*ay)**2*f)
#     dxdf = coeff*(alp*gy*gxy - gx*(ay**2 + alp*gyy))
#     dydf = coeff*(alp*gx*gxy - gy*(ax**2 + alp*gxx))
#     return -pi*f**3*((tdm0 - 2*tg0*alp)*(dxdf*gx + dydf*gy) - 2*tdm0*g/f)/(c*re*pctocm)

# def deltaDMB(uvec, upvec, tg0, tdm0, alp, ax, ay, f, sgnG):
#     """ Returns DM perturbation as a function of root of lens equation uvec, observation point upvec, lens parameters, coefficients tg0 = dso/(2*c*dsl*dlo), tdm0 = c*re*dm/(2*pi*f**2), alpma = -c**2*re*dsl*dlo*dm/(2*pi*f**2*dso), and signed value of G. """
#     ux, uy = uvec
#     upx, upy = upvec
#     psix, psiy = gauss10(*uvec), gauss01(*uvec)
#     psixx, psiyy, psixy = gauss20(*uvec), gauss02(*uvec), gauss11(*uvec)
#     duxdf = -2*alp*sgnG*(alp*psiy*psixy - psix*(ay**2 + alp*psiyy))/((ax*ay)**2*f)
#     duydf = -2*alp*sgnG*(alp*psix*psixy - psiy*(ax**2 + alp*psixx))/((ax*ay)**2*f)
#     deltadm = -pi*f**3*(tdm0*(duydf*psiy + duxdf*psix - 2*gauss(*uvec)/f) + 2*tg0*(ax**2*duxdf*(ux-upx) + ay**2*duydf*(uy-upy)))/(c*re*pctocm)
#     return deltadm
