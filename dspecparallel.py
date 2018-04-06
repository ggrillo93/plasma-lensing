from fundfunctions import *
from matplotlib import pyplot as plt
import shlex
import os
import glob
from subprocess import *
# from dspectra import *

path = '/home/gian/Documents/Research/NANOGrav/Lensing/Scripts/Simulation/dspectra/'

dso, dsl, dm, ax, ay = 1.1*kpc*pctocm, 0.55*kpc*pctocm, -1e-4*pctocm, 0.1*autocm, 0.2*autocm

m, n = 1., 0.
uxmax, uymax = 5., 5.
nspectra = len(sorted(glob.glob(path + '*.dat'))) + 1.
fmin, fmax = 0.1*GHz, 1.5*GHz
nupts = 1000

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
upvec = np.array([upxvec, upyvec])
segs = np.split(upvec, 5, axis = 1)

fcoeff = dsl*(dso - dsl)*re*dm/(2*pi*dso)
alpp = alpha(dso, dsl, 1., dm)
coeff = alpp*np.array([1./ax**2, 1./ay**2])
rF2p = rFsqr(dso, dsl, 1.)
lcp = lensc(dm, 1.)
allcoeff = [fcoeff, alpp, coeff, rF2p, lcp]
nfpts = 2000
cdist = 1e6
comp = False

args1 = np.array([fmin, fmax, allcoeff, ax, ay, m, n, 0, nspectra, cdist, nfpts, comp, segs[0]], dtype = 'object')
np.save(path + 'temp1', args1)
args2 = np.array([fmin, fmax, allcoeff, ax, ay, m, n, 1, nspectra, cdist, nfpts, comp, segs[1]], dtype = 'object')
np.save(path + 'temp2', args2)
args3 = np.array([fmin, fmax, allcoeff, ax, ay, m, n, 2, nspectra, cdist, nfpts, comp, segs[2]], dtype = 'object')
np.save(path + 'temp3', args3)
args4 = np.array([fmin, fmax, allcoeff, ax, ay, m, n, 3, nspectra, cdist, nfpts, comp, segs[3]], dtype = 'object')
np.save(path + 'temp4', args4)
args5 = np.array([fmin, fmax, allcoeff, ax, ay, m, n, 4, nspectra, cdist, nfpts, comp, segs[4]], dtype = 'object')
np.save(path + 'temp5', args5)

p1 = Popen(shlex.split('taskset -c 0 python dspectra.py temp1 &'))
p2 = Popen(shlex.split('taskset -c 1 python dspectra.py temp2 &'))
p3 = Popen(shlex.split('taskset -c 2 python dspectra.py temp3 &'))
p4 = Popen(shlex.split('taskset -c 3 python dspectra.py temp4 &'))
p5 = Popen(shlex.split('taskset -c 4 python dspectra.py temp5 &'))
# # os.system('taskset -c 0 python dspectra.py temp1 &')
# # os.system('taskset -c 1 python dspectra.py temp2 &')
# # os.system('taskset -c 2 python dspectra.py temp3 &')
# # os.system('taskset -c 3 python dspectra.py temp4')

p1.wait()
p2.wait()
p3.wait()
p4.wait()
p5.wait()


mat1 = np.loadtxt(path + 'dspectra' + str(nspectra) + '0.dat')
mat2 = np.loadtxt(path + 'dspectra' + str(nspectra) + '1.dat')
mat3 = np.loadtxt(path + 'dspectra' + str(nspectra) + '2.dat')
mat4 = np.loadtxt(path + 'dspectra' + str(nspectra) + '3.dat')
mat5 = np.loadtxt(path + 'dspectra' + str(nspectra) + '4.dat')
mat = np.concatenate((mat1, mat2, mat3, mat4, mat5), axis=0)

extent = (xmin, xmax, fmin/GHz, fmax/GHz)
plt.figure(figsize=(10, 10))
plt.imshow(mat.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
plt.xlabel(r"$u_x'$")
plt.ylabel(r'$\nu$ (GHz)')
plt.colorbar()
plt.show()
