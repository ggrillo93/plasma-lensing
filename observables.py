from fundfunctions import *
from scipy.special import gamma as gfunc

# Phase
# @jit(nopython = True)
def phi(uvec, rF2, lc, ax, ay):
    """ Returns the phase at a stationary point. """
    ux, uy = uvec
    return 0.5*rF2*lc**2*((gauss10(ux, uy)/ax)**2 + (gauss01(ux, uy)/ay)**2) + lc*gauss(ux, uy)

# Amplitude
# @jit(nopython=True)
def GOAmplitude(uvec, rF2, lc, ax, ay):
    """ Returns the geometrical optics amplitude. """
    ux, uy = uvec
    alp = rF2*lc
    phi20 = ax**2/rF2 + lc*gauss20(ux, uy)
    phi02 = ay**2/rF2 + lc*gauss02(ux, uy)
    phi11 = lc*gauss11(ux, uy)
    H = phi20*phi02 - phi11**2
    ans = (ax*ay/rF2)*np.abs(H)**-0.5
    return ans

# Field
def GOfieldA(uvec, rF2, lc, ax, ay):
    """ Returns the elements of the geometrical optics field: the amplitude and the phase, including the phase shift as determined by the sign of the derivatives. """
    ux, uy = uvec
    alp = rF2*lc
    phi20 = ax**2/rF2 + lc*gauss20(ux, uy)
    phi02 = ay**2/rF2 + lc*gauss02(ux, uy)
    phi11 = lc*gauss11(ux, uy)
    sigma = np.sign(phi02)
    H = phi20*phi02 - phi11**2
    delta = np.sign(H)
    amp = (ax*ay/rF2)*np.abs(H)**-0.5
    phase = phi(uvec, rF2, lc, ax, ay)
    pshift = pi*(delta + 1)*sigma*0.25
    return np.array([amp, phase, pshift])

def GOfieldB(uvec, rF2, lc, ax, ay):
    """ Returns the geometrical optics field, including the second order term. """
    # Stamnes 1986, Dingle 1973
    ux, uy = uvec
    alp = rF2*lc
    phi20 = ax**2/rF2 + lc*gauss20(ux, uy)
    phi02 = ay**2/rF2 + lc*gauss02(ux, uy)
    phi11 = lc*gauss11(ux, uy)
    H = phi20*phi02 - phi11**2
    phi21, phi12, phi30, phi03, phi22, phi31, phi13, phi40, phi04 = lc*np.array([gauss21(ux, uy), gauss12(ux, uy), gauss30(ux, uy), gauss03(ux, uy), gauss22(ux, uy), gauss31(ux, uy), gauss13(ux, uy), gauss40(ux, uy), gauss04(ux, uy)])
    A = phi02**3*(5*phi30**2 - 3*phi20*phi40)
    B = phi20**3*(5*phi03**2 - 3*phi02*phi04)
    C = 3*phi20*phi02*(phi20*(2*phi21*phi03 + 3*phi12**2) + phi02*(2*phi12*phi30 + 3*phi21**2) - 2*phi20*phi02*phi22)
    D = -6*phi11*(phi20**2*(5*phi12*phi03 - 2*phi02*phi13) + phi02**2*(5*phi21*phi30 - 2*phi20*phi31) + phi20*phi02*(9*phi12*phi21 + phi30*phi03))
    E = 3*phi11**2*(phi20*(8*phi21*phi03 + 12*phi12**2 + phi20*phi04) + phi02*(8*phi12*phi30 + 12*phi21**2 + phi02*phi40) - 2*phi20*phi02*phi22)
    F = -4*phi11**3*(3*phi20*phi13 + 3*phi02*phi31 + 9*phi12*phi21 + phi30*phi03)
    G = 12*phi11**4*phi22
    q2 = (A + B + C + D + E + F + G)/(24.*H**3)
    sigma = np.sign(phi02)
    delta = np.sign(H)
    return ax*ay/(rF2*np.abs(H)**0.5) * exp(1j*(phi(uvec, rF2, lc, ax, ay) + pi*(delta + 1)*sigma*0.25)) * (1. + 1j*q2)

@jit(nopython=True)
def physField(uvec, rF2, lc, ax, ay):
    """ Returns an approximation of the field at the caustic, using the formula from Cooke 1982. """
    ux, uy = uvec
    alp = rF2*lc
    phi20, phi02 = ax**2/rF2 + lc*gauss20(ux, uy), ay**2/rF2 + lc*gauss02(ux, uy)
    phi11 = lc*gauss11(ux, uy)
    phi30 = lc*gauss30(ux, uy)
    phi21 = lc*gauss21(ux, uy)
    phi12 = lc*gauss12(ux, uy)
    phi03 = lc*gauss03(ux, uy)
    B = phi20**3*phi03 - 3*phi20**2*phi11*phi12 + 3*phi20*phi11**2*phi21 - phi11**3*phi30
    U = ax*ay/(2*pi*rF2) * exp(1j*(phi(uvec, rF2, lc, ax, ay) + 0.25*pi*np.sign(phi20))) * 2.**(5./6.) * pi**(1./3.) * gfunc(1./3.) * np.abs(phi20)**0.5/(3.**(1./6.) * np.abs(B)**(1./3.))
    return U


# TOA perturbation
def deltatA(uvec, tg0, tdm0, alp, ax, ay):
    """ Returns TOA perturbation in ms as a function of root of lens equation uvec, lens parameters, and coefficients tg0 = dso/(2*c*dsl*dlo), tdm0 = c*re*dm/(2*pi*f**2). """
    return 1e6*(tg0*alp**2*((gauss10(*uvec)/ax)**2 + (gauss01(*uvec)/ay)**2) + tdm0*gauss(*uvec))

def deltatB(uvec, upvec, tg0, tdm0, ax, ay):
    """ Returns TOA perturbation as a function of root of lens equation uvec, observation point upvec, lens parameters, and coefficients tg0 = dso/(2*c*dsl*dlo), tdm0 = c*re*dm/(2*pi*f**2). """
    ux, uy = uvec
    upx, upy = upvec
    return 1e6*(tg0*((ax*(ux-upx))**2 + (ay*(uy-upy))**2) + tdm0*gauss(ux, uy))

# DM perturbation
def deltaDMA(uvec, tg0, tdm0, alp, ax, ay, f, sgnG):
    """ Returns DM perturbation as a function of root of lens equation uvec, lens parameters, coefficients tg0 = dso/(2*c*dsl*dlo), tdm0 = c*re*dm/(2*pi*f**2), alpma = -c**2*re*dsl*dlo*dm/(2*pi*f**2*dso), and signed value of G. """
    g = gauss(*uvec)
    gx, gy = gauss10(*uvec), gauss01(*uvec)
    gxx, gyy = gauss20(*uvec), gauss02(*uvec)
    gxy = gauss11(*uvec)
    coeff = -2*alp*sgnG/((ax*ay)**2*f)
    dxdf = coeff*(alp*gy*gxy - gx*(ay**2 + alp*gyy))
    dydf = coeff*(alp*gx*gxy - gy*(ax**2 + alp*gxx))
    return -pi*f**3*((tdm0 - 2*tg0*alp)*(dxdf*gx + dydf*gy) - 2*tdm0*g/f)/(c*re*pctocm)

def deltaDMB(uvec, upvec, tg0, tdm0, alp, ax, ay, f, sgnG):
    """ Returns DM perturbation as a function of root of lens equation uvec, observation point upvec, lens parameters, coefficients tg0 = dso/(2*c*dsl*dlo), tdm0 = c*re*dm/(2*pi*f**2), alpma = -c**2*re*dsl*dlo*dm/(2*pi*f**2*dso), and signed value of G. """
    ux, uy = uvec
    upx, upy = upvec
    psix, psiy = gauss10(*uvec), gauss01(*uvec)
    psixx, psiyy, psixy = gauss20(*uvec), gauss02(*uvec), gauss11(*uvec)
    duxdf = -2*alp*sgnG*(alp*psiy*psixy - psix*(ay**2 + alp*psiyy))/((ax*ay)**2*f)
    duydf = -2*alp*sgnG*(alp*psix*psixy - psiy*(ax**2 + alp*psixx))/((ax*ay)**2*f)
    deltadm = -pi*f**3*(tdm0*(duydf*psiy + duxdf*psix - 2*gauss(*uvec)/f) + 2*tg0*(ax**2*duxdf*(ux-upx) + ay**2*duydf*(uy-upy)))/(c*re*pctocm)
    return deltadm
