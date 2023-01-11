import numpy as np

def main():
    """ Initialize calculation details, 2e- integrals, Fock matrix, etc """
    import pyscf
    from pyscf import ao2mo
    from pyscf import cc

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
main()
