from fundfunctions import *
from solvers import *
from fslice import *
from observables import *
import sys

path = '/home/gian/Documents/Research/NANOGrav/Lensing/Scripts/Simulation/dspectra/'

def dspectra(fmin, fmax, uxmax, uymax, dso, dsl, dm, ax, ay, m, n, cdist=1e6, nfpts=1000, nupts=1000, comp=True):
    
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
    
    # Calculate coefficients
    fcoeff = dsl*(dso - dsl)*re*dm/(2*pi*dso)
    alpp = alpha(dso, dsl, 1., dm)
    coeff = alpp*np.array([1./ax**2, 1./ay**2])
    rF2p = rFsqr(dso, dsl, 1.)
    lcp = lensc(dm, 1.)
    coeffvec = np.array([coeff, rF2p, lcp])
    
    # Construct caustic curves
    rx = np.linspace(2*xmin, 2*xmax, 500)
    ry = np.linspace(2*ymin, 2*ymax, 500)
    uvec = np.meshgrid(rx, ry)
    A, B, C, D, E = causticFreqHelp(uvec, ax, ay, m, n)
    upxvec = np.linspace(xmin, xmax, nupts)
    freqcaus = []
    for upx in upxvec:
        eq1 = A*upx**2 + B*upx + C
        eq2 = D*upx + E
        evcaus = np.array([eq1, eq2])
        roots = polishedRootsBulk(evcaus, causEqFreq, rx, ry, args = (upx, ax, ay, m, n))
        fvec = []
        ucross = []
        for root in roots:
            ux, uy = root
            arg = fcoeff*lensg(ux, uy)[0]/(ux - upx)
            if arg > 0:
                freq = c*np.sqrt(arg)/ax
                if fmin < freq < fmax:
                    fvec.append(freq)
                    ucross.append(root)
        freqcaus.append([ucross, fvec])
    freqcaus = np.asarray(freqcaus)
    
    # Calculate lens equation invariants
    leqinv = lensEqHelp(uvec, coeff)
    upyvec = upxvec*m + n
    upvec = np.array([upxvec, upyvec]).T
    
    # Find G vs f at fixed upvec and write on file
    nspectra = int(sorted(glob.glob('*.dat'))[-1][-5]) + 1
    # f_handle = file('dspectra' + str(nspectra) + '.dat', 'a')
    mat = np.zeros([nupts, nfpts])
    for i in range(nupts):
        print(i)
        fslice = fsliceGBulk(upvec[i], fmin, fmax, freqcaus[i], dso, dsl, dm, ax, ay, leqinv, rx, ry, uvec, coeffvec, cdist, comp = comp, npoints = nfpts)
        # np.savetxt(f_handle, [fslice])
        mat[i] = fslice
    np.savetxt('dspectra' + str(nspectra) + '.dat', mat)
    return mat
    
def dspectraPar(args):
    
    fmin, fmax, allcoeff, ax, ay, m, n, procn, nspectra, cdist, nfpts, comp, upvec = args
    
    fcoeff, alpp, coeff, rF2p, lcp = allcoeff
    coeffvec = np.array([coeff, rF2p, lcp])
    
    xmin, ymin = upvec[0][0], upvec[1][0]
    xmax, ymax = upvec[0][-1], upvec[1][-1]
    
    nupts = len(upvec[0])
    
    # print([xmin, xmax, ymin, ymax])
    
    # Construct caustic curves
    rx = np.linspace(xmin - 5., xmax + 5., 500)
    ry = np.linspace(ymin - 5., ymax + 5., 500)
    uvec = np.meshgrid(rx, ry)
    A, B, C, D, E = causticFreqHelp(uvec, ax, ay, m, n)
    upxvec = upvec[0]
    freqcaus = []
    for upx in upxvec:
        eq1 = A*upx**2 + B*upx + C
        eq2 = D*upx + E
        evcaus = np.array([eq1, eq2])
        roots = polishedRootsBulk(evcaus, causEqFreq, rx, ry, args = (upx, ax, ay, m, n))
        fvec = []
        ucross = []
        for root in roots:
            ux, uy = root
            arg = fcoeff*lensg(ux, uy)[0]/(ux - upx)
            if arg > 0:
                freq = c*np.sqrt(arg)/ax
                if fmin < freq < fmax:
                    fvec.append(freq)
                    ucross.append(root)
        freqcaus.append([ucross, np.sort(fvec)])
    freqcaus = np.asarray(freqcaus)
    
    # Calculate lens equation invariants
    upvec = upvec.T
    
    leqinv = lensEqHelp(uvec, coeff)
    
    # Find G vs f at fixed upvec and write on file
    mat = np.zeros([nupts, nfpts])
    for i in range(nupts):
        if i % 50 == 0:
            print(i)
        fslice = fsliceGBulk(upvec[i], fmin, fmax, freqcaus[i], leqinv, ax, ay, rx, ry, uvec, coeffvec, cdist, comp = comp, npoints = nfpts)
        mat[i] = fslice
    np.savetxt(path + 'dspectra' + str(nspectra) + str(procn) + '.dat', mat)
    
    return

dspectraPar(np.load(path + sys.argv[1] + '.npy'))
