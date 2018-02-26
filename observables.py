from fundfunctions import *
from scipy.special import gamma as gfunc

# Phase
@jit(nopython = True)
def phiA(uvec, rF2, lc, ax, ay):
    """ Returns the phase as a function of a root of the lens equation uvec and lens parameters. rF2 = Fresnel scale**2 = c*dsl*dlo/(2*pi*f*dso), lc = -c*re*dm/f. """
    ux, uy = uvec
    return 0.5*rF2*lc**2*((gauss10(ux, uy)/ax)**2 + (gauss01(ux, uy)/ay)**2) + lc*gauss(ux, uy)

@jit(nopython = True)
def phiB(uvec, upvec, rF2, lc, ax, ay):
    """ Returns the phase as a function of uvec, upvec, and lens parameters. rF2 = Fresnel scale**2 = c*dsl*dlo/(2*pi*f*dso), lc = -c*re*dm/f. """
    ux, uy = uvec
    upx, upy = upvec
    return 0.5*((ax*(ux - upx))**2 + (ay*(uy - upy))**2)/rF2 + lc*gauss(ux, uy)

@jit(nopython = True)
def phiC(uvec, gam, ax, ay):
    ux, uy = uvec
    return 0.5*np.abs(gam)*((gauss10(ux, uy)/ax)**2 + (gauss01(ux, uy)/ay)**2) + np.sign(gam)*gauss(ux, uy)

# Field
def GOfieldA(uvec, rF2, lc, ax, ay):
    ux, uy = uvec
    sig = np.sign(lc)
    lc = np.abs(lc)
    gam = rF2*lc
    f20 = ax**2/gam + sig*gauss20(ux, uy)
    f02 = ay**2/gam + sig*gauss02(ux, uy)
    f11 = gauss11(ux, uy)
    sigma = np.sign(f02)
    det = f20*f02 - f11**2
    delta = np.sign(det)
    ans = (ax*ay/gam)*np.abs(det)**-0.5*exp(1j*(lc*phiC(uvec, sig*gam, ax, ay) + pi*(delta + 1)*sigma*0.25)) # Cooke 1982
    # if f20*f02 > f11**2 and f20 > 0:
    #     sigma = 1
    # elif f20*f02 > f11**2 and f20 < 0:
    #     sigma = -1
    # else:
    #     sigma = -1j
    # ans = (1j*sigma*ax*ay/gam)*np.abs(det)**-0.5*exp(1j*(lc*phiC(uvec, sig*gam, ax, ay))) # Born & Wolf
    return ans

def GOfieldB(uvec, rF2, lc, ax, ay):
    # Stamnes 1986, Dingle 1973
    ux, uy = uvec
    sig = np.sign(lc)
    lc = np.abs(lc)
    gam = rF2*lc
    f20 = ax**2/gam + sig*gauss20(ux, uy)
    f02 = ay**2/gam + sig*gauss02(ux, uy)
    f11 = sig*gauss11(ux, uy)
    H = f20*f02 - f11**2
    f21, f12, f30, f03, f22, f31, f13, f40, f04 = sig*np.array([gauss21(ux, uy), gauss12(ux, uy), gauss30(ux, uy), gauss03(ux, uy), gauss22(ux, uy), gauss31(ux, uy), gauss13(ux, uy), gauss40(ux, uy), gauss04(ux, uy)])
    A = f02**3*(5*f30**2 - 3*f20*f40)
    B = f20**3*(5*f03**2 - 3*f02*f04)
    C = 3*f20*f02*(f20*(2*f21*f03 + 3*f12**2) + f02*(2*f12*f30 + 3*f21**2) - 2*f20*f02*f22)
    D = -6*f11*(f20**2*(5*f12*f03 - 2*f02*f13) + f02**2*(5*f21*f30 - 2*f20*f31) + f20*f02*(9*f12*f21 + f30*f03))
    E = 3*f11**2*(f20*(8*f21*f03 + 12*f12**2 + f20*f04) + f02*(8*f12*f30 + 12*f21**2 + f02*f40) - 2*f20*f02*f22)
    F = -4*f11**3*(3*f20*f13 + 3*f02*f31 + 9*f12*f21 + f30*f03)
    G = 12*f11**4*f22
    q2 = (A + B + C + D + E + F + G)/(24.*H**3)
    # print([A, B, C, D, E, F, G])
    # print(q2)
    if H < 0:
        sigma = 1
    else:
        if f20 > 0:
            sigma = 1j
        else:
            sigma = -1j
    return sigma*ax*ay/(gam*np.abs(H)**0.5) * exp(1j*lc*phiC(uvec, sig*gam, ax, ay)) * (1. + 1j*q2/lc)

def physField(uvec, rF2, lc, ax, ay):
    ux, uy = uvec
    sig = np.sign(lc)
    lc = np.abs(lc)
    gam = rF2*lc
    f20, f02 = ax**2/gam + sig*gauss20(*uvec), ay**2/gam + sig*gauss02(*uvec)
    f11 = sig*gauss11(*uvec)
    f30 = sig*gauss30(*uvec)
    f21 = sig*gauss21(*uvec)
    f12 = sig*gauss12(*uvec)
    f03 = sig*gauss03(*uvec)
    B = f20**3*f03 - 3*f20**2*f11*f12 + 3*f20*f11**2*f21 - f11**3*f30
    U = ax*ay/(2*pi*rF2) * exp(1j*(lc*phiC(uvec, sig*gam, ax, ay) + 0.25*pi*np.sign(f20))) * 2.**(5./6.) * pi**(1./3.) * gfunc(1./3.) * np.abs(f20)**0.5/(3.**(1./6.) * np.abs(B)**(1./3.) * lc**(5./6.))
    return U

# Intensities
@jit(nopython=True)
def GOgain(uvec, gamma, ax, ay, absolute = True):
    """ Calculates geometrical optics gain at coordinates given by uvec and parameters gamma, ax, and ay. """
    ux, uy = uvec
    arg = (1 + gamma*gauss20(ux, uy)/ax**2)*(1 + gamma*gauss02(ux, uy)/ay**2) - (gamma*gauss11(ux, uy)/(ax*ay))**2
    if absolute:
        return np.abs(1./arg)
    else:
        return 1./arg

def GOgainB(uvec, gamma, ax, ay):
    ux, uy = uvec
    f20 = ax**2/gamma + gauss20(ux, uy)
    f02 = ay**2/gamma + gauss02(ux, uy)
    f11 = gauss11(ux, uy)
    return (ax*ay)**2/(np.abs(f02*f20 - f11**2)*gamma**2)

def physGainA(uvec, rF2, lc, ax, ay):
    """ Returns the physical optics gain as a function of a root of the lens equation uvec, and lens parameters. rF2 = Fresnel scale**2 = c*dsl*dlo/(2*pi*f*dso), lc = -c*re*dm/f. Implements the formula given by Cooke (1982). """
    ux, uy = uvec
    gam = rF2*lc
    f20, f02 = ax**2/gam + gauss20(*uvec), ay**2/gam + gauss02(*uvec)
    f11 = gauss11(*uvec)
    f30 = gauss30(*uvec)
    f21 = gauss21(*uvec)
    f12 = gauss12(*uvec)
    f03 = gauss03(*uvec)
    B = f20**3*f03 - 3*f20**2*f11*f12 + 3*f20*f11**2*f21 - f11**3*f30
    G = (ax*ay/(2*pi*rF2))**2*np.abs(f20) * np.abs(2**(5./6.) * pi**(1./3.) * gfunc(1./3.)/(3**(1./6.) * np.abs(B)**(1./3.)*(lc + 0j)**(5./6.)))**2
    return G

def physGainB(uvec, rF2, lc, ax, ay, absolute = True):
    """ Returns the physical optics gain as a function of a root of the lens equation uvec, observation point upvec, and lens parameters. rF2 = Fresnel scale**2 = c*dsl*dlo/(2*pi*f*dso), lc = -c*re*dm/f. Implements the formula given by Chako (1965), which appears to be wrong. """
    ux, uy = uvec
    phixx = ax**2/rF2 + lc*gauss20(*uvec)
    phiyyy = lc*gauss03(*uvec)
    return 2./(pi*phixx)*(ax*ay*gfunc(1./3.)*np.cos(pi/6.)/(rF2*(9*np.abs(phiyyy))**(1./3.)))**2.

# TOA perturbation
def deltatA(uvec, tg0, tdm0, gam, ax, ay):
    """ Returns TOA perturbation in ms as a function of root of lens equation uvec, lens parameters, and coefficients tg0 = dso/(2*c*dsl*dlo), tdm0 = c*re*dm/(2*pi*f**2). """
    return 1e6*(tg0*gam**2*((gauss10(*uvec)/ax)**2 + (gauss01(*uvec)/ay)**2) + tdm0*gauss(*uvec))

def deltatB(uvec, upvec, tg0, tdm0, ax, ay):
    """ Returns TOA perturbation as a function of root of lens equation uvec, observation point upvec, lens parameters, and coefficients tg0 = dso/(2*c*dsl*dlo), tdm0 = c*re*dm/(2*pi*f**2). """
    ux, uy = uvec
    upx, upy = upvec
    return 1e6*(tg0*((ax*(ux-upx))**2 + (ay*(uy-upy))**2) + tdm0*gauss(ux, uy))

# DM perturbation
def deltaDMA(uvec, tg0, tdm0, gam, ax, ay, f, sgnG):
    """ Returns DM perturbation as a function of root of lens equation uvec, lens parameters, coefficients tg0 = dso/(2*c*dsl*dlo), tdm0 = c*re*dm/(2*pi*f**2), gamma = -c**2*re*dsl*dlo*dm/(2*pi*f**2*dso), and signed value of G. """
    g = gauss(*uvec)
    gx, gy = gauss10(*uvec), gauss01(*uvec)
    gxx, gyy = gauss20(*uvec), gauss02(*uvec)
    gxy = gauss11(*uvec)
    coeff = -2*gam*sgnG/((ax*ay)**2*f)
    dxdf = coeff*(gam*gy*gxy - gx*(ay**2 + gam*gyy))
    dydf = coeff*(gam*gx*gxy - gy*(ax**2 + gam*gxx))
    return -pi*f**3*((tdm0 - 2*tg0*gam)*(dxdf*gx + dydf*gy) - 2*tdm0*g/f)/(c*re*pctocm)

def deltaDMB(uvec, upvec, tg0, tdm0, gam, ax, ay, f, sgnG):
    """ Returns DM perturbation as a function of root of lens equation uvec, observation point upvec, lens parameters, coefficients tg0 = dso/(2*c*dsl*dlo), tdm0 = c*re*dm/(2*pi*f**2), gamma = -c**2*re*dsl*dlo*dm/(2*pi*f**2*dso), and signed value of G. """
    ux, uy = uvec
    upx, upy = upvec
    psix, psiy = gauss10(*uvec), gauss01(*uvec)
    psixx, psiyy, psixy = gauss20(*uvec), gauss02(*uvec), gauss11(*uvec)
    duxdf = -2*gam*sgnG*(gam*psiy*psixy - psix*(ay**2 + gam*psiyy))/((ax*ay)**2*f)
    duydf = -2*gam*sgnG*(gam*psix*psixy - psiy*(ax**2 + gam*psixx))/((ax*ay)**2*f)
    deltadm = -pi*f**3*(tdm0*(duydf*psiy + duxdf*psix - 2*gauss(*uvec)/f) + 2*tg0*(ax**2*duxdf*(ux-upx) + ay**2*duydf*(uy-upy)))/(c*re*pctocm)
    return deltadm
