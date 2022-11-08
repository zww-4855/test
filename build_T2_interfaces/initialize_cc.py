import numpy as np
from numpy import einsum
from pyscf import gto, scf, mp, mcscf
from pyscf import cc
from pyscf import ao2mo
from openfermion.chem.molecular_data import spinorb_from_spatial
import openfermion as of
from openfermionpyscf import run_pyscf

def Initialize_CC(mf,mol,convg_C):

    hcore=mf.get_hcore()
    coeff=mf.mo_coeff
    oei=(coeff.T@hcore)@coeff


    twoInts=ao2mo.kernel(mol,mf.mo_coeff)
    tei = ao2mo.restore(1, twoInts, coeff.shape[0])
    tei = np.asarray(tei.transpose(0, 2, 3, 1), order='C')

    molecule = of.MolecularData(geometry=[['H', (0, 0, 0)], ['B', (0, 0, 0.9)]],
                                basis='cc-pvdz', charge=0, multiplicity=1)
    molecule = run_pyscf(molecule,run_ccsd=False)

    # 1-, 2-electron integrals
    oei, tei = molecule.get_integrals()


    norbs = int(mf.mo_coeff.shape[1])
    occ = mf.mo_occ
    nele = int(sum(occ))
    nocc = nele // 2

    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)

    # put in physics notation. OpenFermion stores <12|2'1'>
    gtei = astei.transpose(0, 1, 3, 2)

    eps = np.kron(molecule.orbital_energies, np.ones(2))
    n = np.newaxis
    o = slice(None, 2 * nocc)
    v = slice(2 * nocc, None)


    e_abcijk=0 #FOR NOW
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    fock = soei + np.einsum('piiq->pq', astei[:, o, o, :])
    hf_energy = 0.5 * np.einsum('ii', (fock + soei)[o, o])
    hf_energy_test = 1.0 * einsum('ii', fock[o, o]) -0.5 * einsum('ijij', gtei[o, o, o, o])
    print('Estimated HF energy:',hf_energy+mf.energy_nuc())
    print('hf_energy:',hf_energy)
    print('nuclear energy:', molecule.nuclear_repulsion)
    g = gtei
    nsvirt = 2 * (norbs - nocc)
    nsocc = 2 * nocc

    return nsvirt, nsocc, fock, g, o, v, e_ai, e_abij, e_abcijk

