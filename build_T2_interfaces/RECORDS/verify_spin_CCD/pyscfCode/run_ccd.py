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
    e_hf=mf.e_tot
    print('e_hf:', e_hf)

# Input to CCD driver: mol, mf, orb

# Check and make sure integrals are transformed to MO basis correctly for RHF
    #test_rhf_energy(mol,mf,orb)

# Now check the same for UHF
    mf = mol.UHF()
    mf.run()
    orb=mf.mo_coeff
    print(np.shape(mf.get_hcore()))
    print(np.shape(orb))

# Means we are running RHF; must generalize data structs for use in UHF code
    if orb.ndim <= 2:
        h1e = np.array((mf.get_hcore(), mf.get_hcore()))
        f   = np.array((mf.get_fock(),mf.get_fock()))
        na  = mol.nelectron//2
        nb  = na
        orb = np.array((orb,orb))
        print('shape of numpy coeff rhf:', np.shape(orb))

    elif orb.ndim > 2: # MEANS IM RUNNING UHF CALC
        h1e=np.array((mf.get_hcore(), mf.get_hcore()))
        f=mf.get_fock()
        na,nb=mf.nelec

    

    g_aaaa,g_bbbb,g_abab=generalUHF(mf,mol,h1e,f,na,nb,orb)
    import sys
    sys.exit()


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
    g_aaaa = g_aaaa.transpose(0,2,1,3)-g_aaaa.transpose(0,3,1,2)
    g_bbbb = g_bbbb.transpose(0,2,1,3)-g_bbbb.transpose(0,3,1,2)
    g_abab = g_abab.transpose(0,2,1,3)


# Now, verify the UHF energy
    e1=0.5*np.einsum('ii',h1aa[:na,:na]) +0.5*np.einsum('ii',h1bb[:nb,:nb])
    e2=0.5*np.einsum('ii',faa[:na,:na])  +0.5*np.einsum('ii',fbb[:nb,:nb])
    totSCFenergy=e1+e2+mf.energy_nuc()
    print('final rhf/uhf energy:', totSCFenergy)
    return g_aaaa,g_bbbb,g_abab


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
