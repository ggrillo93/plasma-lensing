from fundfunctions import *
from scipy.special import gamma as gfunc

# Phase
def phi(uvec, rF2, phi0, lc, ax, ay):
    """ Returns the phase at a stationary point. """
    ux, uy = uvec
    grad = lensg(ux, uy)
    return 0.5*rF2*lc**2*((grad[0]/ax)**2 + (grad[1]/ay)**2) + lc*lensfun(*uvec) + phi0 - 0.5*pi

# Field
def GOfield(uvec, rF2, phi0, lc, ax, ay):
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
    phase = phi(uvec, rF2, phi0, lc, ax, ay)
    pshift = pi*(delta + 1)*sigma*0.25
    # return amp*np.exp(1j*(phase + pshift))
    return np.array([amp, phase, pshift])

# Amplitudes
def GOAmp(uvec, rF2, lc, ax, ay):
    """ Returns the geometrical optics amplitude. """
    ux, uy = uvec
    alp = rF2*lc
    psi20, psi02, psi11 = lensh(ux, uy)
    phi20 = ax**2/rF2 + lc*psi20
    phi02 = ay**2/rF2 + lc*psi02
    phi11 = lc*psi11
    H = phi20*phi02 - phi11**2
    ans = (ax*ay/rF2)*np.abs(H)**-0.5
    return ans
    
def causAmp(uvec, rF2, lc, ax, ay):
    """ Returns an approximation to the amplitude at the caustic, using the formula from Cooke 1982. """
    ux, uy = uvec
    psi20, psi02, psi11 = lensh(ux, uy)
    phi20 = ax**2/rF2 + lc*psi20
    phi02 = ay**2/rF2 + lc*psi02
    phi11 = lc*psi11
    phi30, phi21, phi12, phi03 = lc*np.asarray(lensgh(ux, uy))
    B = phi20**3*phi03 - 3*phi20**2*phi11*phi12 + 3*phi20*phi11**2*phi21 - phi11**3*phi30
    amp = ax*ay/(2*pi*rF2) * 2.**(5./6.) * pi**(1./2.) * gfunc(1./3.) * np.abs(phi20)**0.5/(3.**(1./6.) * np.abs(B)**(1./3.))
   #  G = 2./(pi*np.abs(phi20))*(ax*ay*gfunc(1./3.)*np.cos(pi/6.)/(rF2*(9*np.abs(phi03))**(1./3.)))**2.
    # G = (ax*ay/(2*pi*rF2))**2/3 * (gfunc(0.5)*gfunc(1./3.)/(np.abs(phi20)**0.5*np.abs(phi03)**(1./3.)))**2
    return amp**2
    
# TOA perturbation
def deltat(uvec, tg0, tdm0, alp, ax, ay):
    """ Returns TOA perturbation in us at a stationary point. Coefficients: tg0 = dso/(2*c*dsl*dlo), tdm0 = c*re*dm/(2*pi*f**2). """
    ux, uy = uvec
    grad = lensg(ux, uy)
    return 1e6*(tg0*alp**2*((grad[0]/ax)**2 + (grad[1]/ay)**2) + tdm0*lensfun(ux, uy))
    
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
    
def uniAsymp(allroots, allfields, nreal, ncomplex, nzones, sigs):
    """ Constructs the uniform asympotics for a segmented array of roots and their respective fields. """
    
    def bright(A1, A2, phi1, phi2, sig):
        if phi1[0] > phi2[0]:
            pdiff = phi1 - phi2
            g1 = A2 - A1
        else:
            pdiff = phi2 - phi1
            g1 = A1 - A2
        chi = 0.5*(phi1 + phi2)
        print(phi1)
        print(phi2)
        phase = chi + sig*0.25*pi
        xi = -(0.75*pdiff)**(2./3.)
        air = airy(xi)
        a1 = pi**0.5 *((A1 + A2)*(-xi)**0.25*air[0] - 1j*g1*(-xi)**-0.25*air[1]) * np.exp(1j*(chi + sig*0.25*pi))
        return a1
        
    def dark(A, phi, sig):
        xi = (1.5*np.abs(phi.imag))**(2./3.)
        a1 = 2*pi**0.5*A*(xi)**0.25 * airy(xi)[0] * np.exp(1j*(phi.real + sig*0.25*pi))
        return a1
    
    asymp = np.array([])
    
    for i in range(nzones):
        p = i - 1 # caustic index
        roots, fields = allroots[i], allfields[i]
        nroots = (ncomplex + nreal)[i]
        npoints = len(roots)
        realn = int(nreal[i])
        if nreal[i] == 1: # no real roots merge
            areal = constructField(*fields[0])
        else: # deal with merging real roots
            merge = [findClosest(roots[0][:realn].real), findClosest(roots[-1][:realn].real)] # find closest real roots at each end
            mroot1, mroot2 = merge[0][0], merge[1][0] # set indices of merging roots
            nmroots1 = list(set(range(realn)) - set(mroot1)) # indices of non merging roots at one end
            nmroots2 = list(set(range(realn)) - set(mroot2)) # indices of non merging roots at other end
            if merge[0][1] < 0.4 and merge[1][1] < 0.4: # case 1: real root merging at both ends
                # print('Double root merging')
                if np.all(mroot1 == mroot2): # same root merges at both ends
                    A1, phi1 = fields[mroot1[0]][:2]
                    A2, phi2 = fields[mroot1[1]][:2]
                    amerge = bright(A1, A2, phi1, phi2, sigs[p])
                    anonm = np.zeros(npoints, dtype = complex)
                    for index in nmroots1:
                        anonm = anonm + constructField(*fields[index]) # sum of fields not involved in merging
                    areal = amerge + anonm
                else: # different roots merge at each end
                    A11, A21 = np.array_split(fields[mroot1[0]][0], 2)[0], np.array_split(fields[mroot1[1]][0], 2)[0]
                    A32, A42 = np.array_split(fields[mroot2[0]][0], 2)[1], np.array_split(fields[mroot2[1]][0], 2)[1]
                    phi11, phi21 = np.array_split(fields[mroot1[0]][1], 2)[0], np.array_split(fields[mroot1[1]][1], 2)[0]
                    phi32, phi42 = np.array_split(fields[mroot2[0]][1], 2)[1], np.array_split(fields[mroot2[1]][1], 2)[1]
                    amerge1 = bright(A11, A21, phi11, phi21, sigs[p])
                    amerge2 = bright(A32, A42, phi32, phi42, sigs[p + 1])
                    nmfields1 = [constructField(*np.array_split(fields[nmroot], 2, axis = 1)[0]) for nmroot in nmroots1]
                    nmfields2 = [constructField(*np.array_split(fields[nmroot], 2, axis = 1)[1]) for nmroot in nmroots2]
                    anonm1 = np.zeros(len(amerge1), dtype = complex)
                    for j in range(len(nmroots1)):
                        anonm1 = anonm1 + nmfields1[j]
                    anonm2 = np.zeros(len(amerge2), dtype = complex)
                    for j in range(len(nmroots2)):
                        anonm2 = anonm2 + nmfields2[j]
                    areal = np.concatenate((amerge1 + anonm1, amerge2 + anonm2))
            elif merge[0][1] < 0.4 and merge[1][1] > 0.4: # case 2: real root merging at first end only
                # print('Root merging at first end')
                A1, phi1 = fields[mroot1[0]][:2]
                A2, phi2 = fields[mroot1[1]][:2]
                amerge = bright(A1, A2, phi1, phi2, sigs[p])
                anonm = np.zeros(npoints, dtype = complex)
                for index in nmroots1:
                    anonm = anonm + constructField(*fields[index]) # sum of fields not involved in merging
                areal = amerge + anonm
            elif merge[0][1] > 0.4 and merge[1][1] < 0.4: # case 3: real root merging at second end only
                # print('Root merging at second end')
                A1, phi1 = fields[mroot2[0]][:2]
                A2, phi2 = fields[mroot2[1]][:2]
                amerge = bright(A1, A2, phi1, phi2, sigs[p+1])
                anonm = np.zeros(npoints, dtype = complex)
                for index in nmroots2:
                    anonm = anonm + constructField(*fields[index]) # sum of fields not involved in merging
                areal = amerge + anonm
        if ncomplex[i] != 0: # deal with merging complex roots
            A, phi = fields[realn][:2]
            cond = np.abs(roots[0][realn][0].imag/roots[0][realn][0].real) < np.abs(roots[-1][realn][0].imag/roots[-1][realn][0].real)
            if cond: # complex root merges at first end
                # print('Complex root merging at first end')
                acomp = dark(A, phi, sigs[p])
            else: # complex root merges at second end
                acomp = dark(A, phi, sigs[p + 1])
            if ncomplex[i] == 2 and np.around(fields[realn][0][0], 3) != np.around(fields[realn + 1][0][0], 3):
                A2, phi2 = fields[realn + 1][:2]
                if cond:
                    acomp = acomp + dark(A2, phi2, sigs[p + 1])
                else:
                    acomp = acomp + dark(A2, phi2, sigs[p])
        else:
            acomp = np.zeros(npoints)
        asymp = np.append(asymp, areal + acomp)
    return np.asarray(asymp).flatten()
    
def uniAsympTOA(roots, fields, toas, realn, npoints, sig):
    """ Constructs the uniform asympotics. """
    
    def bright(A1, A2, phi1, phi2, sig):
        if phi1[0] > phi2[0]:
            pdiff = phi1 - phi2
            g1 = A2 - A1
        else:
            pdiff = phi2 - phi1
            g1 = A1 - A2
        chi = 0.5*(phi1 + phi2)
        xi = -(0.75*pdiff)**(2./3.)
        # air = airsqrenv(np.abs(xi))**0.5
        air = airy(xi)
        a1 = pi**0.5 *((A1 + A2)*(-xi)**0.25*air[0] - 1j*g1*(-xi)**-0.25*air[1]) * np.exp(1j*(chi + sig*0.25*pi))
        # a1 = pi**0.5 *((A1 + A2)*(-xi)**0.25*air) * np.exp(1j*(chi + sig*0.25*pi))
        return a1
    
    merge = [findClosest(roots[0].real), findClosest(roots[-1].real)] # find closest real roots at each end
    mroot1, mroot2 = merge[0][0], merge[1][0] # set indices of merging roots
    nmroots1 = list(set(range(realn)) - set(mroot1)) # indices of non merging roots at one end
    nmroots2 = list(set(range(realn)) - set(mroot2)) # indices of non merging roots at other end
    print(merge)
    if merge[0][1] < 0.1 and merge[1][1] < 0.1: # case 1: real root merging at both ends
            
        A1, phi1 = fields[mroot1[0]][:2]
        A2, phi2 = fields[mroot1[1]][:2]
        A3, phi3 = fields[mroot2[0]][:2]
        A4, phi4 = fields[mroot2[1]][:2]
        
        dphi1 = np.abs(phi1 - phi2)
        dphi2 = np.abs(phi3 - phi4)
        h1 = np.argwhere(dphi1 < pi).flatten()[-1]
        h2 = npoints - np.argwhere(dphi2 < pi).flatten()[0]
        
        bfields1 = np.zeros([realn - 1, h1], dtype = complex)
        btoas1 = np.zeros([realn - 1, h1])
        bfields1[0] = bright(A1[:h1], A2[:h1], phi1[:h1], phi2[:h1], sig[0])
        btoas1[0] = np.mean([toas[mroot1[0]][:h1], toas[mroot1[1]][:h1]], axis = 0)
        
        bfields2 = np.zeros([realn - 1, h2], dtype = complex)
        btoas2 = np.zeros([realn - 1, h2])
        bfields2[0] = bright(A3[-h2:], A4[-h2:], phi3[-h2:], phi4[-h2:], sig[1])
        btoas2[0] = np.mean([toas[mroot2[0]][-h2:], toas[mroot2[1]][-h2:]], axis = 0)
        
        for i in range(len(nmroots1)):
            nmroot1 = nmroots1[i]
            nmroot2 = nmroots2[i]
            bfields1[i + 1] = constructField(*fields[nmroot1][:, :h1])
            bfields2[i + 1] = constructField(*fields[nmroot2][:, -h2:])
            btoas1[i + 1] = toas[nmroot1][:h1]
            btoas2[i + 1] = toas[nmroot2][-h2:]
        
        if npoints - h1 - h2 > 0:
            infields = np.zeros([realn, npoints - h1 - h2], dtype = complex)
            intoas = np.zeros([realn, npoints - h1 - h2])
            
            for i in range(realn):
                infields[i] = constructField(*fields[i][:, h1:-h2])
                intoas[i] = toas[i][h1:-h2]
                
            return np.array([[bfields1, infields, bfields2], [btoas1, intoas, btoas2]])
            
        else:
            return np.array([ [bfields1[:h2], bfields2[-h1:]], [btoas1[:h2], btoas2[-h1:]] ])
            
    elif merge[0][1] < 0.1 and merge[1][1] > 0.1: # case 2: real root merging at first end only
        
        A1, phi1 = fields[mroot1[0]][:2]
        A2, phi2 = fields[mroot1[1]][:2]
        
        dphi = np.abs(phi1 - phi2)
        h = np.argwhere(dphi < pi).flatten()[-1]
        
        bfields = np.zeros([realn - 1, h]) # fields close to caustic
        btoas = np.zeros([realn - 1, h]) # TOAs close to caustic
        bfields[0] = bright(A1[:h], A2[:h], phi1[:h], phi2[:h], sig[0]) # combine merging images into one close to caustic
        btoas[0] = np.mean([toas[mroot1[0]][:h], toas[mroot1[1]][:h]], axis = 0)
        for i in range(len(nmroots1)):
            nmroot = nmroots1[i]
            bfields[i + 1] = constructField(*fields[nmroot][:, :h])
            btoas[i + 1] = toas[nmroot][:h]
        
        infields = np.zeros([realn, npoints - h])
        intoas = np.zeros([realn, npoints - h])
        for i in range(realn):
            infields[i] = constructField(*fields[i][:, h:])
            intoas[i] = toas[i][h:]
        
        return np.array([[bfields, infields], [btoas, intoas]])
        
    elif merge[0][1] > 0.1 and merge[1][1] < 0.1: # case 3: real root merging at second end only
        
        A1, phi1 = fields[mroot2[0]][:2]
        A2, phi2 = fields[mroot2[1]][:2]
        
        dphi = np.abs(phi1 - phi2)
        if len(np.argwhere(dphi < pi).flatten()) != 0:
            h = npoints - np.argwhere(dphi < pi).flatten()[0]
        else:
            h = npoints
        # print(h)
        
        bfields = np.zeros([realn - 1, h], dtype = complex)
        btoas = np.zeros([realn - 1, h])
        bfields[0] = bright(A1[-h:], A2[-h:], phi1[-h:], phi2[-h:], sig[1])
        # print([toas[mroot2[0]][-h:], toas[mroot2[1]][-h:]])
        # print(np.abs(phi1[-h:] - phi2[-h:]))
        btoas[0] = np.mean([toas[mroot2[0]][-h:], toas[mroot2[1]][-h:]], axis = 0)
        for i in range(len(nmroots2)):
            nmroot = nmroots2[i]
            bfields[i + 1] = constructField(*fields[nmroot][:, -h:])
            btoas[i + 1] = toas[nmroot][-h:]
            
        infields = np.zeros([realn, npoints - h], dtype = complex)
        intoas = np.zeros([realn, npoints - h])
        for i in range(realn):
            infields[i] = constructField(*fields[i][:, :-h])
            intoas[i] = toas[i][:-h]
            
        return np.array([[infields, bfields], [intoas, btoas]])

# Momentarily useless stuff

# # @jit(nopython=True)
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
