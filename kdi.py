from fundfunctions import *
import matplotlib.pyplot as plt
from solvers import *
from scipy.fftpack import fft2, ifft2, fftshift, fft, ifft
from scipy.ndimage import map_coordinates
from matplotlib import gridspec as gs
from decimal import Decimal

def gridToPixel(u, umax, gsize):
    return 0.5*gsize*(u/umax + 1.)

def lensPhase(ux, uy, lc):
    """ Return the lens phase perturbation. """
    arg = lc*gauss(ux, uy)
    return exp(1j*arg)

def geoPhase(ux, uy, uF2x, uF2y):
    """ Returns the quadratic phase factor. """
    arg = ux**2/(2*uF2x) + uy**2/(2*uF2y)
    return exp(1j*arg)

def solveKDI(uxmax, uymax, dso, dsl, f, dm, ax, ay, gsizex, gsizey, m = 0 , n = 0):

    rF2 = rFsqr(dso, dsl, f)
    lc = lensc(dm, f)
    uF2x = rF2/ax**2
    uF2y = rF2/ay**2
    gam = rF2*lc
    rx = np.linspace(-uxmax, uxmax, gsizex)
    ry = np.linspace(-uymax, uymax, gsizey)
    dux = 2*uxmax/gsizex
    duy = 2*uymax/gsizey
    extent = (-uxmax, uxmax, -uymax, uymax)
    ux, uy = np.meshgrid(rx, ry)
    lens = lensPhase(ux, uy, lc)
    lensfft = fft2(lens)
    geo = geoPhase(ux, uy, uF2x, uF2y)
    geofft = fft2(geo)
    fieldfft = lensfft*geofft
    field = fftshift(ifft2(fieldfft))
    soln = np.abs((dux*duy*field)**2/(4*pi**2*uF2x*uF2y))

    # Plot G across line of observer motion
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
    rx = np.linspace(xmin, xmax, gsizex)
    xx = np.linspace(gridToPixel(xmin, uxmax, gsizex), gridToPixel(xmax, uxmax, gsizex) - 1, gsizex)
    yy = np.linspace(gridToPixel(ymin, uymax, gsizey), gridToPixel(ymax, uymax, gsizey) - 1, gsizey)

    # Plots
    fig = plt.figure(figsize = (15, 10))
    grid = gs.GridSpec(2, 2, height_ratios = [3, 1])
    tableax = plt.subplot(grid[1, :])
    ax0, ax1 = plt.subplot(grid[0, 0]), plt.subplot(grid[0, 1])

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
        roots = polishedRoots(causticEqSlice, uxmax, uymax, args = (gam, m, n, ax, ay)).T
        rootupx, rootupy = mapToUp(roots, gam, ax, ay)
        ax0.scatter(rootupx, rootupy, color = 'white') # Plot caustic intersections
    ax0.plot(rx, rx*m + n, color = 'white') # Plot observer motion
    ax0.set_xlabel(r"$u'_x$")
    ax0.set_ylabel(r"$u'_y$")
    ax0.set_title("Gain in the u' plane")

    G = map_coordinates(soln.T, np.vstack((xx, yy))) # Plot gain along observer motion
    ax1.plot(rx, G, color = 'blue')
    for caus in rootupx:
        ax1.plot([caus, caus], [-10, 1000], ls = 'dashed', color = 'black')
    ax1.set_ylim(-0.1, np.max(G) + 0.5)
    ax1.set_xlim(np.min(rx), np.max(rx))
    ax1.set_xlabel(r"$u'_x$")
    ax1.set_ylabel('G')
    ax1.set_title("Slice Gain")

    col_labels = ['Parameter', 'Value'] # Create table with parameter values
    if np.abs(dm/pctocm) < 1:
        dmlabel = "{:.2E}".format(Decimal(dm/pctocm))
    else:
        dmlabel = str(dm/pctocm)
    tablevals = [[r'$d_{so} \: (kpc)$', np.around(dso/pctocm/kpc, 2)], [r'$d_{sl} \: (kpc)$', np.around(dsl/pctocm/kpc, 3)], [r'$a_x \: (AU)$', np.around(ax/autocm, 3)], [r'$a_y \: (AU)$', np.around(ay/autocm, 3)], [r'$DM_l \: (pc \, cm^{-3})$', dmlabel], [r"$\nu$ (GHz)", f/GHz], ['Slope', m], ['Offset', n]]
    tableax.axis('tight')
    tableax.axis('off')
    table = tableax.table(cellText = np.asarray(tablevals).T, colWidths = np.ones(8)*0.05, rowLabels = col_labels, loc = 'center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(2.5, 2.5)

    plt.show()
    return
