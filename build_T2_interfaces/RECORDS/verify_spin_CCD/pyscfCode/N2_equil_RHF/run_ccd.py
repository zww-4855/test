import numpy as np
import pyscf
from pyscf import ao2mo
from pyscf import cc
import t2energy as t2energy
import t2residEqns as t2residEqns
import sys

def ccd_main(mf,mol,orb):
    na,nb,nvirta,nvirtb,occ_aa,occ_bb,virt_aa,virt_bb,faa,fbb,gaaaa,gbbbb,gabab,eabij_aa,eabij_bb,eabij_ab=convertSCFinfo(mf,mol,orb)
    hf_energy=mf.e_tot
    print('hf energy:', hf_energy)
    nucE=mf.energy_nuc()
    print('nuclear repulsion:',nucE)
    print(np.shape(gaaaa))
    t2aaaa,t2bbbb,t2abab,currentE=ccd_kernel(na,nb,nvirta,nvirtb,occ_aa,virt_aa,occ_bb,virt_bb,faa,fbb,gaaaa,gbbbb,gabab,eabij_aa,eabij_bb,eabij_ab,hf_energy,nucE,15, 3)


def ccd_kernel(na,nb,nvirta,nvirtb,occaa,virtaa,occbb,virtbb,faa,fbb,gaaaa,gbbbb,gabab,eabij_aa,eabij_bb,eabij_ab,hf_energy,nucE,diis_size=None, diis_start_cycle=4):
    fock_e_abij_aa = np.reciprocal(eabij_aa)
    fock_e_abij_bb = np.reciprocal(eabij_bb)
    fock_e_abij_ab = np.reciprocal(eabij_ab)


    
    t2aaaa=np.zeros((nvirta,nvirta,na,na))
    t2bbbb=np.zeros((nvirtb,nvirtb,nb,nb))
    t2abab=np.zeros((nvirta,nvirtb,na,nb))

    if diis_size is not None:
        from diis import DIIS
        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        t2aaaa_dim = t2aaaa.size
        t2bbbb_dim = t2bbbb.size
        t2abab_dim = t2abab.size

        old_vec = np.hstack((t2aaaa.flatten(), t2bbbb.flatten(),t2abab.flatten()))



    old_energy = t2energy.ccd_energy_with_spin(t2aaaa, t2bbbb, t2abab, faa,fbb,gaaaa,gbbbb,gabab,occaa,occbb,virtaa,virtbb)

    print('initial energy:',old_energy)
    max_iter=75
    stopping_eps=1E-12
    print("    ==> CCSDT amplitude equations <==")
    print("")
    print("     Iter              Corr. Energy                 |dE|    |dT|")
    print(flush=True)
    for idx in range(max_iter):
        resid_aaaa = t2residEqns.ccd_t2_aaaa_residual(t2aaaa, t2bbbb, t2abab, faa, fbb, gaaaa, gbbbb, gabab, occaa, occbb, virtaa, virtbb)+fock_e_abij_aa*t2aaaa

        print('there zww hi')
        resid_bbbb = t2residEqns.ccd_t2_bbbb_residual(t2aaaa, t2bbbb, t2abab, faa, fbb, gaaaa, gbbbb, gabab, occaa, occbb, virtaa, virtbb)+fock_e_abij_bb*t2bbbb
        print('still here')
        resid_abab = t2residEqns.ccd_t2_abab_residual(t2aaaa, t2bbbb, t2abab, faa, fbb, gaaaa, gbbbb, gabab, occaa, occbb, virtaa, virtbb)+ fock_e_abij_ab*t2abab
        print('hi')
#doubles_residual(t1, t2, t3, fock, g, o, v,e_abij,t4)
#residual_quads   = Qf.get_t4(t2,t3,t4,g,o,v,fock)
        res_norm = np.linalg.norm(resid_aaaa)+np.linalg.norm(resid_bbbb)+np.linalg.norm(resid_abab)



        #doubles_res_aaaa = resid_aaaa + fock_e_abij_aa * t2aaaa
        #doubles_res_bbbb = resid_bbbb + fock_e_abij_bb * t2bbbb
        #doubles_res_abab = resid_abab + fock_e_abij_ab * t2abab

        new_doubles_aaaa = resid_aaaa*eabij_aa #doubles_res_aaaa * eabij_aa
        new_doubles_bbbb = resid_bbbb*eabij_bb #doubles_res_bbbb * eabij_bb
        new_doubles_abab = resid_abab*eabij_ab #doubles_res_abab * eabij_ab
        # diis update
        if diis_size is not None:
            vectorized_iterate = np.hstack(
                (new_doubles_aaaa.flatten(), new_doubles_bbbb.flatten(), new_doubles_abab.flatten()))
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate,
                                                                 error_vec)
            new_doubles_aaaa = new_vectorized_iterate[:t2aaaa_dim].reshape(t2aaaa.shape)
            new_doubles_bbbb = new_vectorized_iterate[t2aaaa_dim:t2aaaa_dim+t2bbbb_dim].reshape(t2bbbb.shape)
            new_doubles_abab = new_vectorized_iterate[t2aaaa_dim+t2bbbb_dim:].reshape(t2abab.shape)
            old_vec = new_vectorized_iterate


        current_energy = t2energy.ccd_energy_with_spin(new_doubles_aaaa, new_doubles_bbbb, new_doubles_abab, faa,fbb,gaaaa,gbbbb,gabab,occaa,occbb,virtaa,virtbb)
        delta_e = np.abs(old_energy - current_energy)

        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}".format(idx, nucE+current_energy - hf_energy, delta_e, res_norm))
        print(flush=True)
        if delta_e < stopping_eps:# and res_norm < stopping_eps:
            # assign t1 and t2 variables for future use before breaking
            t2aaaa = new_doubles_aaaa
            t2bbbb = new_doubles_bbbb
            t2abab = new_doubles_abab
            break
        else:
            # assign t1 and t2 and old_energy for next iteration
            t2aaaa = new_doubles_aaaa
            t2bbbb = new_doubles_bbbb
            t2abab = new_doubles_abab
            old_energy = current_energy
    else:
        raise ValueError("CCSDT iterations did not converge")

    
    print('\n\n\n\n\n\n')
    print('Final, total correlated energy:', nucE+current_energy)
    print('Final CCD correlation contribution:', nucE+current_energy-hf_energy)
    return t2aaaa,t2bbbb,t2abab,current_energy



def convertSCFinfo(mf,mol,orb):
# Means we are running RHF; must generalize data structs for use in UHF code
    if orb.ndim <= 2:
        h1e = np.array((mf.get_hcore(), mf.get_hcore()))
        f   = np.array((mf.get_fock(),mf.get_fock()))
        na  = mol.nelectron//2
        nb  = na
        nvirta=f[0].shape[0]-na
        nvirtb=nvirta
        orb = np.array((orb,orb))
        print('shape of numpy coeff rhf:', np.shape(orb))
        moE_aa=mf.mo_energy
        moE_bb=moE_aa
    elif orb.ndim > 2: # MEANS IM RUNNING UHF CALC
        h1e=np.array((mf.get_hcore(), mf.get_hcore()))
        f=mf.get_fock()
        na,nb=mf.nelec
        nvirta=f[0].shape[0]-na
        nvirtb=f[1].shape[0]-nb
        moE_aa=mf.mo_energy[0]
        moE_bb=mf.mo_energy[1]
        print('mo energy:',np.shape(moE_aa))
    
    faa=f[0]
    fbb=f[1]
    #import sys
    #print(na,nb,nvirta,nvirtb,faa.shape,fbb.shape)
    #sys.exit()
    faa,fbb,g_aaaa,g_bbbb,g_abab=generalUHF(mf,mol,h1e,f,na,nb,orb)


    n=np.newaxis
    occ_aa=slice(None, na)
    virt_aa=slice(na,None)
    occ_bb=slice(None,nb)
    virt_bb=slice(nb,None)
    print(occ_aa,virt_aa)
    epsaa=moE_aa
    epsbb=moE_bb

    eabij_aa=1.0/(-epsaa[virt_aa,n,n,n]-epsaa[n,virt_aa,n,n]+epsaa[n,n,occ_aa,n]+epsaa[n,n,n,occ_aa])
    eabij_bb=1.0/(-epsbb[virt_bb,n,n,n]-epsbb[n,virt_bb,n,n]+epsbb[n,n,occ_bb,n]+epsbb[n,n,n,occ_bb])
    eabij_ab=1.0/(-epsaa[virt_aa,n,n,n]-epsbb[n,virt_bb,n,n]+epsaa[n,n,occ_aa,n]+epsbb[n,n,n,occ_bb])

    print('eabij_aa:', eabij_aa,np.shape(eabij_aa))
    return na,nb,nvirta,nvirtb,occ_aa,occ_bb,virt_aa,virt_bb,faa,fbb,g_aaaa,g_bbbb,g_abab,eabij_aa,eabij_bb,eabij_ab

def generalUHF(mf,mol,h1e,f,na,nb,orb):
    #h1e = mf.get_hcore()
    h1aa=orb[0].T@h1e[0]@orb[0]
    h1bb=orb[1].T@h1e[1]@orb[1]

    #f=mf.get_fock()
    faa=orb[0].T@f[0]@orb[0]
    fbb=orb[1].T@f[1]@orb[1]

    #nelec=mol.nelectron
    #na, nb = mf.nelec
    eri = mol.intor('int2e', aosym='s1')
    g_aaaa = ao2mo.incore.general(eri, (orb[0],orb[0],orb[0],orb[0]))
    g_bbbb = ao2mo.incore.general(eri, (orb[1],orb[1],orb[1],orb[1]))
    g_abab = ao2mo.incore.general(eri, (orb[0],orb[0],orb[1],orb[1]))


# Verify the 2e- integral coulomb energy
    ga=g_aaaa.transpose(0,2,1,3)
    gb=g_bbbb.transpose(0,2,1,3)
    e_coul=np.einsum('ijij',ga[:na,:na,:na,:na])+np.einsum('ijij',gb[:nb,:nb,:nb,:nb])
    e_exch=0.5*np.einsum('ijji',ga[:na,:na,:na,:na])+0.5*np.einsum('ijji',gb[:nb,:nb,:nb,:nb])


    print('total 2e- integral energy:',e_coul-e_exch)


# Now, convert to Dirac notation, and antisymmetrize g_aaaa/g_bbbb
    g_aaaa = g_aaaa.transpose(0,2,1,3)-g_aaaa.transpose(0,3,2,1)#(0,3,1,2)
    g_bbbb = g_bbbb.transpose(0,2,1,3)-g_bbbb.transpose(0,3,2,1)
    g_abab = g_abab.transpose(0,2,1,3)
    import sys
    print(np.shape(g_aaaa))
    #sys.exit()

# Now, verify the UHF energy
    e1=0.5*np.einsum('ii',h1aa[:na,:na]) +0.5*np.einsum('ii',h1bb[:nb,:nb])
    e2=0.5*np.einsum('ii',faa[:na,:na])  +0.5*np.einsum('ii',fbb[:nb,:nb])
    totSCFenergy=e1+e2+mf.energy_nuc()
    print('final rhf/uhf energy:', totSCFenergy)
    return faa,fbb,g_aaaa,g_bbbb,g_abab


def test_rhf_energy(mol,mf,orb):
    eri = ao2mo.full(mol, orb, verbose=0)
    print('eri:', eri, np.shape(eri))
    eriFull=ao2mo.restore('s1', eri, orb.shape[1])
    print('full', eriFull,np.shape(eriFull))
    eriFull=eriFull.transpose(0,2,1,3)

    hcore=mf.get_hcore()
    hcoreMO=orb.T @ hcore @ orb

    f=mf.get_fock()
    fock=orb.T @ f @ orb

    nelec=mol.nelectron
    nocc = nelec // 2

    test_e=np.einsum('ii',hcoreMO[:nocc,:nocc])+np.einsum('ii',fock[:nocc,:nocc])

    teint_energy=2.0*np.einsum('ijij',eriFull[:nocc, :nocc, :nocc, :nocc])- np.einsum('ijji',eriFull[:nocc, :nocc, :nocc, :nocc])
    test_e2=np.einsum('ii',hcoreMO[:nocc,:nocc])*2.0+teint_energy

    print(mf.e_tot,test_e+mf.energy_nuc(),test_e2+mf.energy_nuc())

## TODO: 
## Construct general code for both RHF and UHF
## Write test to verify I get same HF SCF energy using 1 and 2 e- ints
