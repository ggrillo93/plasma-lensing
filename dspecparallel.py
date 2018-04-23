from fundfunctions import *
import glob
from joblib import Parallel, delayed
from solvers import *
import multiprocessing
from fslice import *

def caustics(upx, A, B, C, D, E, ax, ay, m, n, fcoeff):
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
    return np.array([ucross, np.sort(fvec)])

path = '/home/gian/Documents/Research/NANOGrav/Lensing/Scripts/Simulation/dspectra/'

dso, dsl, dm, ax, ay = 1.1*kpc*pctocm, 0.55*kpc*pctocm, -1e-4*pctocm, 0.15*autocm, 0.15*autocm

m, n = 0., 1.
uxmax, uymax = 5., 5.
nspectra = len(sorted(glob.glob(path + '*.dat'))) - 3.
fmin, fmax = 0.3*GHz, 1.5*GHz
nupts = 1000
ncpus = multiprocessing.cpu_count() - 1

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
    
upxvec = np.linspace(xmin, xmax, nupts)
upyvec = m*upxvec + n
upvec = np.array([upxvec, upyvec]).T

fcoeff = dsl*(dso - dsl)*re*dm/(2*pi*dso)
alpp = alpha(dso, dsl, 1., dm)
coeff = alpp*np.array([1./ax**2, 1./ay**2])
rF2p = rFsqr(dso, dsl, 1.)
lcp = lensc(dm, 1.)
nfpts = 1000
cdist = 1e4
comp = False

# Construct caustic curves
rx = np.linspace(xmin - 5., xmax + 5., 500)
ry = np.linspace(ymin - 5., ymax + 5., 500)
uvec = np.meshgrid(rx, ry)
A, B, C, D, E = causticFreqHelp(uvec, ax, ay, m, n)
freqcaus = Parallel(n_jobs = ncpus)(delayed(caustics)(*[upx, A, B, C, D, E, ax, ay, m, n, fcoeff]) for upx in upxvec)
    
# Calculate lens equation invariant
leqinv = lensEqHelp(uvec, coeff)

args = [fmin, fmax, leqinv, ax, ay, rx, ry, uvec, rF2p, lcp, coeff, cdist, nfpts, comp]
    
# Find G vs f at fixed upvec and write on file
mat = Parallel(n_jobs = ncpus)(delayed(fsliceGBulk)(*[upvec[i], freqcaus[i], fmin, fmax, leqinv, ax, ay, rx, ry, uvec, rF2p, lcp, coeff, cdist, nfpts, comp]) for i in range(nupts))
mat = np.asarray(mat)

extent = (xmin, xmax, fmin/GHz, fmax/GHz)
fig = plt.figure(figsize=(15, 10))
grid = gs.GridSpec(1, 2, width_ratios = [4, 1])
ax0 = plt.subplot(grid[0, 0])
ax1 = plt.subplot(grid[0, 1])
im0 = ax0.imshow(mat.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
cbar = fig.colorbar(im0, ax = ax0)
cbar.set_label('G')
ax0.set_xlabel(r"$u_x'$")
ax0.set_ylabel(r'$\nu$ (GHz)')
ax0.set_title('Lens shape: ' + '$%s$' % sym.latex(lensf))

col_labels = ['Parameter', 'Value']
if np.abs(dm/pctocm) < 1:
    dmlabel = "{:.2E}".format(Decimal(dm/pctocm))
else:
    dmlabel = str(dm/pctocm)
tablevals = [[r'$d_{so} \: (kpc)$', np.around(dso/pctocm/kpc, 2)], [r'$d_{sl} \: (kpc)$', np.around(dsl/pctocm/kpc, 2)], [r'$a_x \: (AU)$', np.around(ax/autocm, 2)], [r'$a_y \: (AU)$', np.around(ay/autocm, 2)], [r'$DM_l \: (pc \, cm^{-3})$', dmlabel], ['Slope', m], ['Offset', n]]
ax1.axis('tight')
ax1.axis('off')
table = ax1.table(cellText = tablevals, colWidths = [0.25, 0.25], colLabels = col_labels, loc = 'center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(3., 3.)

plt.show()

np.savetxt(path + 'dspectra' + str(nspectra) + '.dat', mat)

# args1 = np.array([fmin, fmax, allcoeff, ax, ay, m, n, 0, nspectra, cdist, nfpts, comp, upvec], dtype = 'object')
# np.save(path + 'temp1', args1)
# args2 = np.array([fmin, fmax, allcoeff, ax, ay, m, n, 1, nspectra, cdist, nfpts, comp, segs[1]], dtype = 'object')
# np.save(path + 'temp2', args2)
# args3 = np.array([fmin, fmax, allcoeff, ax, ay, m, n, 2, nspectra, cdist, nfpts, comp, segs[2]], dtype = 'object')
# np.save(path + 'temp3', args3)
# args4 = np.array([fmin, fmax, allcoeff, ax, ay, m, n, 3, nspectra, cdist, nfpts, comp, segs[3]], dtype = 'object')
# np.save(path + 'temp4', args4)
# args5 = np.array([fmin, fmax, allcoeff, ax, ay, m, n, 4, nspectra, cdist, nfpts, comp, segs[4]], dtype = 'object')
# np.save(path + 'temp5', args5)

# p1 = Popen(shlex.split('taskset -c 0 python dspectra.py temp1 &'))
# p2 = Popen(shlex.split('taskset -c 1 python dspectra.py temp2 &'))
# p3 = Popen(shlex.split('taskset -c 2 python dspectra.py temp3 &'))
# p4 = Popen(shlex.split('taskset -c 3 python dspectra.py temp4 &'))
# p5 = Popen(shlex.split('taskset -c 4 python dspectra.py temp5 &'))
# # # os.system('taskset -c 0 python dspectra.py temp1 &')
# # # os.system('taskset -c 1 python dspectra.py temp2 &')
# # # os.system('taskset -c 2 python dspectra.py temp3 &')
# # # os.system('taskset -c 3 python dspectra.py temp4')

# p1.wait()
# p2.wait()
# p3.wait()
# p4.wait()
# p5.wait()


# mat1 = np.loadtxt(path + 'dspectra' + str(nspectra) + '0.dat')
# mat2 = np.loadtxt(path + 'dspectra' + str(nspectra) + '1.dat')
# mat3 = np.loadtxt(path + 'dspectra' + str(nspectra) + '2.dat')
# mat4 = np.loadtxt(path + 'dspectra' + str(nspectra) + '3.dat')
# mat5 = np.loadtxt(path + 'dspectra' + str(nspectra) + '4.dat')
# mat = np.concatenate((mat1, mat2, mat3, mat4, mat5), axis=0)