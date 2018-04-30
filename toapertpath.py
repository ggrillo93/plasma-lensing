from toapertopt import *
from fundfunctions import *
import glob
from joblib import Parallel, delayed
from solvers import *
import multiprocessing

def caustics(upx, A, B, C, D, E, ax, ay, m, n, fcoeff):
    eq1 = A*upx**2 + B*upx + C
    eq2 = D*upx + E
    evcaus = np.array([eq1, eq2])
    roots = polishedRootsBulk(evcaus, causEqFreq, rx,
                              ry, args=(upx, ax, ay, m, n))
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

path = 'J1713+0747.Rcvr_800.GUPPI.9y.x.sum.sm'
template = pp.Archive(path).getData()

dso, dsl, dm, ax, ay = 1.1*kpc*pctocm, 0.55*kpc*pctocm, -7e-4*pctocm, 0.5*autocm, 1.1*autocm

m, n = 0.75, 0.
uxmax, uymax = 0.5, 0.5
ntoa = len(sorted(glob.glob('*.dat'))) - 1
fmin, fmax = 0.3*GHz, 3.5*GHz
period = 4.5e3
nupts = 100

ncpus = multiprocessing.cpu_count()

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
    
upxvec = np.linspace(-0.5, 0.5, nupts)
upxvec = upxvec[np.array(np.nonzero(np.logical_or(upxvec <= 0.5, upxvec >= 0.7))).flatten()]
upxvec = upxvec[np.array(np.nonzero(np.logical_or(upxvec <= 0.9, upxvec >= 1.1))).flatten()]
upxvec = upxvec[np.array(np.nonzero(np.logical_or(upxvec <= -1.8, upxvec >= -1.6))).flatten()]
upxvec = upxvec[np.array(np.nonzero(np.logical_or(upxvec <= -1.48, upxvec >= -1.44))).flatten()]
upxvec = upxvec[np.array(np.nonzero(np.logical_or(upxvec <= -1.185, upxvec >= -1.17))).flatten()]
print(upxvec)
upyvec = m*upxvec + n
upvec = np.array([upxvec, upyvec]).T
# print(upvec)

fcoeff = dsl*(dso - dsl)*re*dm/(2*pi*dso)
alpp = alpha(dso, dsl, 1., dm)
coeff = alpp*np.array([1./ax**2, 1./ay**2])
rF2p = rFsqr(dso, dsl, 1.)
lcp = lensc(dm, 1.)
tg0 = tg0coeff(dso, dsl)
tdm0p = tdm0coeff(dm, 1.)
nfpts = 500
cdist = 1e5

# Construct caustic curves
rx = np.linspace(xmin - 5., xmax + 5., 500)
ry = np.linspace(ymin - 5., ymax + 5., 500)
uvec = np.meshgrid(rx, ry)
A, B, C, D, E = causticFreqHelp(uvec, ax, ay, m, n)
freqcaus = Parallel(n_jobs = ncpus)(delayed(caustics)(*[upx, A, B, C, D, E, ax, ay, m, n, fcoeff]) for upx in upxvec)
    
# Calculate lens equation invariant
leqinv = lensEqHelp(uvec, coeff)
df = (fmax - fmin - 2*cdist)/nfpts  # frequency grid spacing
dt = period/2048.  # time axis spacing
taxis = np.linspace(-period/2., period/2., 2048)

# fslicepertBulk(upvec[0], freqcaus[0], fmin, fmax, leqinv, ax, ay, rx, ry, uvec, rF2p, lcp, alpp, tdm0p, tg0, coeff, taxis, df, dt, cdist, nfpts, template, period, True, tsize = 2048)

# args = [fmin, fmax, leqinv, ax, ay, rx, ry, uvec, rF2p, lcp, alpp, tdm0p, tg0, coeff, taxis, df, dt, cdist, nfpts, template, period, plot]
    
avs = Parallel(n_jobs = ncpus)(delayed(fslicepertBulk)(*[upvec[i], freqcaus[i], fmin, fmax, leqinv, ax, ay, rx, ry, uvec, rF2p, lcp, alpp, tdm0p, tg0, coeff, taxis, df, dt, cdist, nfpts, template, period, False, 0.2, ntoa]) for i in range(len(upxvec)))
avs = np.asarray(avs).T
# avs = np.loadtxt(path + 'toapert0.dat')
np.savetxt('avs' + str(ntoa) + '.dat', avs)

fig = plt.figure(figsize=(15, 10))
grid = gs.GridSpec(2, 2, width_ratios = [4, 1])
ax0 = plt.subplot(grid[:, 0])
ax1 = plt.subplot(grid[1, 1])
ax2 = plt.subplot(grid[0, 1])

colors = ['red', 'blue', 'green']
bands = ['820 MHz', '1400 MHz', '2300 MHz']
for i in range(3):
    ax0.scatter(upxvec, avs[i], color = colors[i], label = bands[i])
ax0.set_ylabel(r"$\overline{\Delta t}_{comb}$ ($\mu s$)")
ax0.set_xlabel(r"$u_{x}'$")
ax0.plot([-10, 10], [0, 0], ls = 'dashed', color = 'black', scalex = False, scaley = False)
ax0.set_title('Lens shape: ' + '$%s$' % sym.latex(lensf))
ax0.legend()

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

freqcaus = causCurveFreq(uxmax, uymax, ax, ay, dso, dsl, dm, m, n, plot = False, N = 250)
ax2.scatter(freqcaus[0], freqcaus[1], marker='.', color='red')
ax2.set_xlim(min(freqcaus[0]), max(freqcaus[0]))
ax2.set_xlabel(r"$u'_x$")
ax2.set_ylabel(r'$\nu$ (GHz)')
ax2.grid()
ax2.set_title("Caustic curves")
ax2.set_aspect('auto', anchor='C')
ax2.axis('tight')
ax2.set_ylim(fmin/GHz, fmax/GHz)
ax2.set_xlim(xmin, xmax)
grid.update(wspace=0.2, hspace=0.3)
plt.savefig('plot' + str(ntoa))
# plt.show()
