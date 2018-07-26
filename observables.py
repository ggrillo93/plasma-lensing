from fundfunctions import *
from scipy.special import gamma as gfunc

# Phase
def phi(uvec, rF2, lc, ax, ay):
    """ Returns the phase at a stationary point (does not take into account the caustic phase shift). """
    ux, uy = uvec
    grad = lensg(ux, uy)
    return 0.5*rF2*lc**2*((grad[0]/ax)**2 + (grad[1]/ay)**2) + lc*lensfun(*uvec)

# Field
def GOfield(uvec, rF2, lc, ax, ay):
    """ Returns the elements of the geometrical optics field: the amplitude and the phase. """
    ux, uy = uvec
    alp = rF2*lc
    phase = phi(uvec, rF2, lc, ax, ay)
    if phase.imag < 0:
        phase = np.conj(phase)
        ux, uy = np.conj(uvec)
    psi20, psi02, psi11 = lensh(ux, uy)
    phi20 = ax**2/rF2 + lc*psi20
    phi02 = ay**2/rF2 + lc*psi02
    phi11 = lc*psi11
    H = phi20*phi02 - phi11**2
    delta = np.sign(H)
    sigma = np.sign(phi02)
    if phase.imag == 0:
        pshift = pi*(delta + 1)*sigma*0.25
        phase = phase + pshift
        amp = (ax*ay/rF2)*np.abs(H)**-0.5
    else:
        amp = (ax*ay/rF2)*(-H)**-0.5
    return np.array([amp, phase])

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
def constructField(amp, phases):
    return amp*np.exp(1j*phases)
    
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

def findMinPhi(phis):
    """ Returns the minimum phase difference of a set of roots and the indices of the roots with minimum difference. """
    diffmat = np.abs(np.subtract.outer(phis, phis))
    ind = np.tril_indices(diffmat.shape[0], k=-1)
    # minind = np.asarray(ind).T[np.argmin(diffmat[ind])]
    # mindiff = np.min(diffmat[ind])
    p = np.argsort(diffmat[ind])
    sdiff = diffmat[ind][p] - pi/2.
    sind = np.asarray(ind).T[p]
    return [sind[:4], sdiff[:4]]

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
    
def uniAsymp(allfields, segs, nreal, ncomplex):
    """ Constructs the uniform asympotics for a segmented array of roots and their respective fields. """
    
    def bright(A1, A2, phi1, phi2):
        p1 = phi1[0] - phi2[0]
        p2 = phi1[-1] - phi2[-1]
        # print([p1, p2])
        if np.abs(p1) < np.abs(p2):
            if p1 < 0:
                pdiff = phi1 - phi2 + pi/2.
                g1 = A1 - A2
            else:
                pdiff = phi2 - phi1 + pi/2.
                g1 = A2 - A1
        else:
            if p2 < 0:
                pdiff = phi1 - phi2 + pi/2.
                g1 = A1 - A2
            else:
                pdiff = phi2 - phi1 + pi/2.
                g1 = A2 - A1
        chi = 0.5*(phi1 + phi2)
        xi = -(0.75*pdiff)**(2./3.)
        air = airy(xi)
        amp = pi**0.5 *((A1 + A2)*(-xi)**0.25*air[0] + 1j*g1*(-xi)**-0.25*air[1])
        return np.array([amp, chi])
        
    def dark(A, phi):
        # if A[0].real > A[-1].real:
        #     # pshift = np.sign(A[0].real)*pi*0.25
        #     # pshift = np.sign(A[0].real)
        # else:
        #     # pshift = np.sign(A[-1].real)*pi*0.25
        #     pshift = np.sign(A[-1].real)
        # print(pshift)
        # print(A)
        xi = (1.5*np.abs(phi.imag))**(2./3.)
        air = airy(xi)
        a1 = 2*pi**0.5*A*xi**0.25 * air[0]
        return np.array([a1, phi.real])
    
    nzones = len(segs)
    asymp = list(np.zeros(nzones))
    
    for i in range(nzones):
        fields = allfields[i]
        nroots = (ncomplex + nreal)[i]
        npoints = len(fields)
        realn = int(nreal[i])
        print(realn)
        seg = segs[i]
        zone = []
        if nreal[i] == 1: # no real roots merge
            ampfunc = interp1d(seg, fields[0][0])
            phifunc = interp1d(seg, fields[0][1])
            zone.append([ampfunc, phifunc])
        else: # deal with merging real roots
            merge = [findMinPhi(fields.T[0][1][:realn]), findMinPhi(fields.T[-1][1][:realn])] # find closest real roots at each end
            print(merge)
            mroot1, mroot2 = merge[0][0][0], merge[1][0][0] # set indices of merging roots
            nmroots1 = list(set(range(realn)) - set(mroot1)) # indices of non merging roots at one end
            nmroots2 = list(set(range(realn)) - set(mroot2)) # indices of non merging roots at other end
            if merge[0][1][0] < pi and merge[1][1][0] < pi: # case 1: real root merging at both ends
                if np.all(mroot1 == mroot2): # same root merges at both ends
                    A1, phi1 = fields[mroot1[0]]
                    A2, phi2 = fields[mroot1[1]]
                    bamp, bphi = bright(A1, A2, phi1, phi2)
                    ampfunc = interp1d(seg, bamp)
                    phifunc = interp1d(seg, bphi)
                    zone.append([ampfunc, phifunc])
                    for index in nmroots1:
                        ampfunc = interp1d(seg, fields[index][0])
                        phifunc = interp1d(seg, fields[index][1])
                        zone.append([ampfunc, phifunc])
                else: # different roots merge at each end
                    same = np.intersect1d(mroot1, mroot2)
                    if len(same) == 0:
                        A1, phi1 = fields[mroot1[0]]
                        A2, phi2 = fields[mroot1[1]]
                        A3, phi3 = fields[mroot2[0]]
                        A4, phi4 = fields[mroot2[1]]
                        bamp1, bphi1 = bright(A1, A2, phi1, phi2)
                        bamp2, bphi2 = bright(A3, A4, phi3, phi4)
                        bampfunc1 = interp1d(seg, bamp1)
                        bphifunc1 = interp1d(seg, bphi1)
                        bampfunc2 = interp1d(seg, bamp2)
                        bphifunc2 = interp1d(seg, bphi2)
                        zone.append([bampfunc1, bphifunc1])
                        zone.append([bampfunc2, bphifunc2])
                        nmroots = np.intersect1d(nmroots1, nmroots2)
                        for index in nmroots:
                            ampfunc = interp1d(seg, fields[index][0])
                            phifunc = interp1d(seg, fields[index][1])
                            zone.append([ampfunc, phifunc])
                    else:
                        A21, A22 = np.array_split(fields[same[0]][0], 2)
                        phi21, phi22 = np.array_split(fields[same[0]][1], 2)
                        left = np.nonzero(mroot1 - same[0])[0][0]
                        A11, A12 = np.array_split(fields[mroot1[left]][0], 2)
                        phi11, phi12 = np.array_split(fields[mroot1[left]][1], 2)
                        right = np.nonzero(mroot2 - same[0])[0][0]
                        A31, A32 = np.array_split(fields[mroot2[right]][0], 2)
                        phi31, phi32 = np.array_split(fields[mroot2[right]][1], 2)
                        bamp1, bphi1 = bright(A11, A21, phi11, phi21)
                        bamp2, bphi2 = bright(A22, A32, phi22, phi32)
                        bamp = np.concatenate((bamp1, bamp2))
                        bphi = np.concatenate((bphi1, bphi2))
                        bampfunc = interp1d(seg, bamp)
                        bphifunc = interp1d(seg, bphi)
                        zone.append([bampfunc, bphifunc])
                        nmamp = np.concatenate((A31, A12))
                        nmphi = np.concatenate((phi31, phi12))
                        nmampfunc = interp1d(seg, nmamp)
                        nmphifunc = interp1d(seg, nmphi)
                        zone.append([nmampfunc, nmphifunc])
                        nmroots = np.intersect1d(nmroots1, nmroots2)
                        for index in nmroots:
                            ampfunc = interp1d(seg, fields[index][0])
                            phifunc = interp1d(seg, fields[index][1])
                            zone.append([ampfunc, phifunc])
            elif merge[0][1][0] < pi and merge[1][1][0] > pi: # case 2: real root merging at first end only
                A1, phi1 = fields[mroot1[0]]
                A2, phi2 = fields[mroot1[1]]
                bamp, bphi = bright(A1, A2, phi1, phi2)
                bampfunc = interp1d(seg, bamp)
                bphifunc = interp1d(seg, bphi)
                zone.append([bampfunc, bphifunc])
                for index in nmroots1:
                    ampfunc = interp1d(seg, fields[index][0])
                    phifunc = interp1d(seg, fields[index][1])
                    zone.append([ampfunc, phifunc])
            elif merge[0][1][0] > pi and merge[1][1][0] < pi: # case 3: real root merging at second end only
                A1, phi1 = fields[mroot2[0]]
                A2, phi2 = fields[mroot2[1]]
                bamp, bphi = bright(A1, A2, phi1, phi2)
                bampfunc = interp1d(seg, bamp)
                bphifunc = interp1d(seg, bphi)
                zone.append([bampfunc, bphifunc])
                for index in nmroots2:
                    ampfunc = interp1d(seg, fields[index][0])
                    phifunc = interp1d(seg, fields[index][1])
                    zone.append([ampfunc, phifunc])
            else:
                for j in range(realn):
                    ampfunc = interp1d(seg, fields[j][0])
                    phifunc = interp1d(seg, fields[j][1])
                    zone.append([ampfunc, phifunc])
        if ncomplex[i] != 0: # deal with merging complex roots
            A, phi = fields[realn]
            camp, cphi = dark(A, phi)
            campfunc = interp1d(seg, camp)
            cphifunc = interp1d(seg, cphi)
            zone.append([campfunc, cphifunc])
            if ncomplex[i] == 2:
                A2, phi2 = fields[realn + 1]
                if np.around(A[0], 3) != np.around(A2[0], 3):
                    camp2, cphi2 = dark(A2, phi2)
                    campfunc = interp1d(seg, camp2)
                    cphifunc = interp1d(seg, cphi2)
                    zone.append([campfunc, cphifunc])
        asymp[i] = zone
    return asymp
