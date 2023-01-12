import numpy as np
import pyscf
from pyscf import ao2mo
from pyscf import cc

def main():
    """ Initialize calculation details, 2e- integrals, Fock matrix, etc """

    # run pyscf for some reason
    basis = '6-31G'
    mol = pyscf.M(
        atom='H 0 0 0; F 0 0 {}'.format(0.917),
        #atom='Be 0 0 0'
        verbose=5,
        basis=basis)

    mf = mol.RHF()
    mf.run()


    occ=mf.mo_occ
    nelec=mol.nelectron
    nocc = nelec // 2

    orb=mf.mo_coeff
    f=mf.get_fock()
    eri=ao2mo.kernel(mol, orb)



    e_hf=mf.e_tot
    print('e_hf:', e_hf)
    mycc=cc.CCSD(mf)
    eris = mycc.ao2mo(mycc.mo_coeff)
    print('eris', eris,np.shape(eris))


    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    print('fock mat:',fock)
    eints=mol.intor('int2e', aosym='s1')
    print('SHAPE OF LOADED 2E INTS:', np.shape(eints))
    eri_ints = mol.ao2mo(eints,orb,aosym='s1')
    print(np.shape(eri_ints),nocc,np.shape(fock))
    eri8 = ao2mo.restore('s1', eints, orb.shape[1])
    print(eri8.shape)# in format (ij|kl)
    # change to format <ik|jl>
    eri8=eri8.transpose(0,2,3,1)

    print(orb.shape[1])
    gtei=2.0*eri8 #- eri8.transpose(0,1,3,2)#np.einsum('ijkl',eri8) - np.einsum('ijlk',eri8)
    g=gtei
#    g=gtei.transpose(0,1,3,2)

    hcore=mf.get_hcore()
    hcoreMO=orb.T @ hcore @ orb


    test_e=np.einsum('ii',hcoreMO[:nocc,:nocc])+np.einsum('ii',fock[:nocc,:nocc])
    print('test_e:', test_e)
    print('diagonal of hcore and f', np.einsum('ii',fock[:nocc,:nocc]),np.einsum('ii',hcoreMO[:nocc,:nocc]))
    #print('true <ij||ij> energy:', mf.e2)
    #print('true 1e energy:', mf.e1)
    for i in range(4):
        eri8 = np.tensordot(eri8, orb, axes=1).transpose(3, 0, 1, 2)
    eri8 = eri8.transpose(0, 2, 3, 1)

    print(np.einsum('ii',fock[:nocc,:nocc])*2.0 -np.einsum('jiji',eri8[:nocc, :nocc, :nocc, :nocc]))#np.einsum('ijij',eri8[:nocc, :nocc, :nocc, :nocc])-np.einsum('ijij',eri8[:nocc, :nocc, :nocc, :nocc].transpose(0,1,3,2)))
    eri = ao2mo.full(mol, orb, verbose=0)
    print('eri:', eri, np.shape(eri))
    eriFull=ao2mo.restore('s1', eri, orb.shape[1])
    print('full', eriFull,np.shape(eriFull))
    eriFull=eriFull.transpose(0,2,1,3)

    print(2.0*np.einsum('ijij',eriFull[:nocc, :nocc, :nocc, :nocc])- np.einsum('ijji',eriFull[:nocc, :nocc, :nocc, :nocc]))
#+np.einsum('ijij', g[:nocc, :nocc, :nocc, :nocc]))

    test_rhf_energy(mol,mf,orb)


    mf = mol.UHF()
    mf.run()
    orb=mf.mo_coeff
    print(np.shape(orb),orb.ndim)
    if orb.ndim > 2: # MEANS IM RUNNING UHF CALC
        h1e = mf.get_hcore()
        h1aa=orb[0].T@h1e@orb[0]
        h1bb=orb[1].T@h1e@orb[1]

        f=mf.get_fock()
        faa=orb[0].T@f[0]@orb[0]
        fbb=orb[1].T@f[1]@orb[1]

        mo_energy = mf.mo_energy
        import sys
        print(mo_energy[0],mo_energy[1])
        e_idx_a = np.argsort(mo_energy[0])
        e_idx_b = np.argsort(mo_energy[1])
        e_sort_a = mo_energy[0][e_idx_a]
        e_sort_b = mo_energy[1][e_idx_b]
        nmo = mo_energy[0].size
        n_a, n_b = mf.nelec
        print(np.allclose(orb[0],orb[1]),np.shape(orb[0]))
        eri = mol.intor('int2e', aosym='s1')
        g_aaaa = ao2mo.incore.general(eri, (orb[0],orb[0],orb[0],orb[0]))
        g_bbbb = ao2mo.incore.general(eri, (orb[1],orb[1],orb[1],orb[1]))
        g_abab = ao2mo.incore.general(eri, (orb[0],orb[0],orb[1],orb[1]))
        print(g_aaaa[0,0,:3,:3], g_bbbb[0,0,:3,:3],  g_abab[0,0,:3,:3])


# Verify the 2e- integral coulomb energy
        na=n_a
        nb=n_b
        ga=g_aaaa.transpose(0,2,1,3)
        gb=g_bbbb.transpose(0,2,1,3)
        e_coul=np.einsum('ijij',ga[:na,:na,:na,:na])+np.einsum('ijij',gb[:nb,:nb,:nb,:nb])
        e_exch=0.5*np.einsum('ijji',ga[:na,:na,:na,:na])+0.5*np.einsum('ijji',gb[:nb,:nb,:nb,:nb])


        print('total 2e- integral energy:',e_coul-e_exch)


# Now, convert to Dirac notation, and antisymmetrize g_aaaa/g_bbbb
        g_aaaa = g_aaaa.transpose(0,2,1,3)-g_aaaa.transpose(0,3,1,2)
        g_bbbb = g_bbbb.transpose(0,2,1,3)-g_bbbb.transpose(0,3,1,2)
        g_abab = g_abab.transpose(0,2,1,3)


# Now, verify the UHF energy
        e1=0.5*np.einsum('ii',h1aa[:n_a,:n_a]) +0.5*np.einsum('ii',h1bb[:n_b,:n_b])
        e2=0.5*np.einsum('ii',faa[:n_a,:n_a])  +0.5*np.einsum('ii',fbb[:n_b,:n_b])
        print('final uhf energy:', e1+e2)



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
main()
