from observables import *
from solvers import *
import time

def fslicepert(upvec, fmin, fmax, dso, dsl, dm, ax, ay, template, period, npoints = 3000, tsize = 50000, plot = True):
    
    start = time.time()
    
    def tempmatch(data):
        pulse = pp.SinglePulse(data)
        shift = pulse.fitPulse(newtemp**2)[1]
        return shift*dt
    
    # Calculate coefficients
    fcoeff = dsl*(dso - dsl)*re*dm/(2*pi*dso)
    alpp = alpha(dso, dsl, 1., dm)
    coeff = alpp*np.array([1./ax**2, 1./ay**2])
    rF2p = rFsqr(dso, dsl, 1.)
    lcp = lensc(dm, 1.)
    tg0 = tg0coeff(dso, dsl)
    tdm0p = tdm0coeff(dm, 1.)
    
    # Find frequency caustics
    upx, upy = upvec
    ucross = polishedRoots(causEqFreq, np.abs(upx) + 3., np.abs(upy) + 3., args = (upx, ax, ay, upy/upx, 0))
    fcross = []
    ucrossb = []
    for uvec in ucross:
        ux, uy = uvec
        arg = fcoeff*lensg(ux, uy)[0]/(ux - upx)
        if arg > 0:
            freq = c*np.sqrt(arg)/ax
            if fmin < freq < fmax:
                fcross.append(freq)
                ucrossb.append([ux, uy])
    fcross = np.asarray(fcross)
    p = np.argsort(fcross)
    fcross = fcross[p]
    ucrossb = np.asarray(ucrossb)[p]
    ncross = len(fcross)
    
    # Increase size of template grid
    taxor = np.linspace(-period/2., period/2., 2048)
    inttemp = interp1d(taxor, template)
    taxis = np.linspace(-period/2., period/2., tsize)
    newtemp = inttemp(taxis)
    
    # Calculate sign of second derivative at caustics
    sigs = np.zeros(ncross)
    for i in range(ncross):
        rF2 = rFsqr(dso, dsl, fcross[i])
        lc = lensc(dm, fcross[i])
        sigs[i] = np.sign(ax**2/rF2 + lc*lensh(ucrossb[i][0], ucrossb[i][1])[0])
        
    cdist = 1e6 # set minimum caustic distance

    # Set up boundaries
    bound = np.insert(fcross, 0, fmin)
    bound = np.append(bound, fmax)
    midpoints = [(bound[i] + bound[i+1])/2. for i in range(len(bound) - 1)] # find middle point between boundaries
    nzones = len(midpoints)
    nreal = np.zeros(nzones, dtype = int)
    for i in range(nzones):
        mpoint = midpoints[i]
        leqcoeff = coeff/mpoint**2
        nreal[i] = int(len(findRoots(lensEq, np.abs(upx) + 3., np.abs(upy) + 3., args = (upvec, leqcoeff), N = 1000)))
    segs = np.array([np.linspace(bound[i-1] + cdist, bound[i] - cdist, npoints) for i in range(1, ncross + 2)])
    ncomplex = np.zeros(nzones)
    df = (fmax - fmin - 2*cdist)/npoints
    dt = period/tsize
    print(nreal)
    
    # Solve lens equation at each coordinate
    allroots = rootFinderFreq(segs, nreal, ncomplex, npoints, ucrossb, upvec, coeff)
    
    # Calculate field components, TOAs
    allfields = []
    alltoas = []
    for l in range(nzones):
        nroots = len(allroots[l][0])
        fvec = segs[l]
        roots = allroots[l]
        fields = np.zeros([nroots, 3, npoints], dtype = complex)
        toas = np.zeros([nroots, npoints])
        for i in range(npoints):
            freq = fvec[i]
            rF2 = rF2p/freq
            lc = lcp/freq
            alp = rF2*lc
            tdm0 = tdm0p/freq**2
            for j in range(nroots):
                ans = GOfield(roots[i][j], rF2, lc, ax, ay)
                toas[j][i] = deltat(roots[i][j].real, tg0, tdm0, alp, ax, ay)
                for k in range(3):
                    fields[j][k][i] = ans[k]
        allfields.append(fields)
        alltoas.append(toas)
    
    # Calculate combined fields for merging roots using uniform asymptotics
    merged = []
    for i in range(nzones):
        if nreal[i] > 1:
            merged.append(uniAsympTOA(allroots[i], allfields[i], nreal[i], npoints, sigs[i]))
        else:
            merged.append(0)
    # print(merged)
    
    # Combine field components for all roots
    combfields = []
    for i in range(nzones):
        arrsh = allroots[i].shape
        nroots = nreal[i]
        totfield = np.zeros([nroots, npoints], dtype=complex)
        for j in range(nroots):
            totfield[j] = constructField(*allfields[i][j])
        combfields.append(totfield)
    
    # Create pulses and calculate TOAs
    h = int(100*cdist/df) # inner boundary
    
    toapert = np.zeros([nzones, npoints])
    for i in range(nzones):
        nroots = nreal[i]
        toas = alltoas[i]
        if nroots == 1:
            toapert[i] = toas[0]
        else: # need to combine pulses            
            fields = combfields[i]
            merge, mroot1, mroot2, nmroots1, nmroots2, cond = merged[i]
            print(cond)
            if cond == 2: # root merging at first end only
                infields, intoas = fields[:, h:], toas[:, h:] # fields and TOAs far away from caustics
                bfields, btoas = fields[:, :h], toas[:, :h] # fields and TOAs at the boundary
                mfield, mtoas = merge[:h], np.mean([btoas[mroot1[0]], btoas[mroot1[1]]], axis = 0)[:h] # fields and TOAs of combined merging images close to the caustic
                nmfields, nmtoas = bfields[nmroots1], btoas[nmroots1] # fields and TOAs of nonmerging images close to the caustic
                inpts = len(infields[0])
                bpts = len(mfield)
                
                for j in range(bpts): # TOA perturbation at boundaries
                    tpulse = np.roll(newtemp*mfield[j], int(mtoas[j]/dt))
                    for k in range(len(nmfields)):
                        pulse = np.roll(newtemp*nmfields[k][j], int(nmtoas[k][j]/dt))
                        tpulse = tpulse + pulse
                    toapert[i][j] = tempmatch(np.abs(tpulse)**2)
                    
                for j in range(inpts): # TOA perturbation at inner points
                    tpulse = np.zeros(tsize)
                    for k in range(nroots):
                        pulse = np.roll(newtemp*infields[k][j], int(intoas[k][j]/dt)) # shift template
                        tpulse = tpulse + pulse
                    toapert[i][h + j] = tempmatch(np.abs(tpulse)**2)  
                
            elif cond == 3: # root merging at second end only
                infields, intoas = fields[:, :-h], toas[:, :-h]
                bfields, btoas = fields[:, -h:], toas[:, -h:] # fields and TOAs at the boundary
                mfield, mtoas = merge[-h:], np.mean([btoas[mroot2[0]], btoas[mroot2[1]]], axis = 0)[-h:] # fields and TOAs of combined merging images close to the caustic
                nmfields, nmtoas = bfields[nmroots2], btoas[nmroots2] # fields and TOAs of nonmerging images close to the caustic
                inpts = len(infields[0])
                bpts = len(mfield)
                
                for j in range(bpts): # TOA perturbation at boundaries
                    tpulse = np.roll(newtemp*mfield[j], int(mtoas[j]/dt))
                    for k in range(len(nmfields)):
                        pulse = np.roll(newtemp*nmfields[k][j], int(nmtoas[k][j]/dt))
                        tpulse = tpulse + pulse
                    toapert[i][-h + j] = tempmatch(np.abs(tpulse)**2)
                    
                for j in range(inpts): # TOA perturbation at inner points
                    tpulse = np.zeros(tsize)
                    for k in range(nroots):
                        pulse = np.roll(newtemp*infields[k][j], int(intoas[k][j]/dt)) # shift template
                        tpulse = tpulse + pulse
                    toapert[i][j] = tempmatch(np.abs(tpulse)**2)  
                    
            else: # root merging at both ends
                infields, intoas = fields[:, h:-h], toas[:, h:-h]
                bfields1, btoas1 = fields[:, :h], toas[:, :h]
                bfields2, btoas2 = fields[:, -h:], toas[:, -h:]
                mfield1, mtoas1 = merge[:h], np.mean([btoas1[mroot1[0]], btoas1[mroot1[1]]], axis = 0)[:h]
                mfield2, mtoas2 = merge[-h:], np.mean([btoas2[mroot2[0]], btoas2[mroot2[1]]], axis = 0)[-h:]
                nmfields1, nmtoas1 = bfields1[nmroots1], btoas1[nmroots1]
                nmfields2, nmtoas2 = bfields2[nmroots2], btoas2[nmroots2]
                inpts = len(infields[0])
                bpts = len(mfield1)
                
                for j in range(inpts):  # TOA perturbation at inner points
                    tpulse = np.zeros(tsize)
                    for k in range(nroots):
                        pulse= np.roll(newtemp*infields[k][j], int(intoas[k][j]/dt)) # shift template
                        tpulse= tpulse + pulse
                    toapert[i][h + j] = tempmatch(np.abs(tpulse)**2)
                
                for j in range(bpts): # TOA perturbation at first end
                    tpulse = np.roll(newtemp*mfield1[j], int(mtoas1[j]/dt))
                    for k in range(len(nmfields)):
                        pulse = np.roll(newtemp*nmfields1[k][j], int(nmtoas1[k][j]/dt))
                        tpulse = tpulse + pulse
                    toapert[i][j] = tempmatch(np.abs(tpulse)**2)
                
                for j in range(bpts): # TOA perturbation at second end
                    tpulse = np.roll(newtemp*mfield2[j], int(mtoas2[j]/dt))
                    for k in range(len(nmfields)):
                        pulse = np.roll(newtemp*nmfields2[k][j], int(nmtoas2[k][j]/dt))
                        tpulse = tpulse + pulse
                    toapert[i][-h + j] = tempmatch(np.abs(tpulse)**2)
    
    print 'It took', time.time()-start, 'seconds.'
    
    if plot:
        fig, axarr = plt.subplots(3, sharex = True)
        axarr[2].plot([-1, 10], [0, 0], ls='dashed', color='black')
        axarr[2].plot(segs.flatten()/GHz, toapert.flatten(), color = 'red')
        axarr[2].set_ylabel(r'$\Delta t_{comb}$ ($\mu s$)')
        axarr[2].set_xlabel(r'$\nu$ (GHz)')
        axarr[2].set_xlim([fmin/GHz, fmax/GHz])
        for i in range(len(segs)):
            field = combfields[i]
            toas = alltoas[i]
            for j in range(len(field)):
                axarr[0].plot(segs[i]/GHz, np.abs(field[j])**2, color = 'black')
                axarr[1].plot(segs[i]/GHz, toas[j], color = 'blue')
        axarr[0].set_yscale('log')
        axarr[0].set_ylabel('G')
        axarr[1].set_ylabel(r'$\Delta t_{ind}$ ($\mu s$)')
        axarr[1].set_yscale('symlog')
        plt.show()
        
    return
