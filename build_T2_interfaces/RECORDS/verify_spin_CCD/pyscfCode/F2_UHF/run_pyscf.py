import numpy as np
import pyscf
from pyscf import ao2mo


def main():
    """ Initialize calculation details, 2e- integrals, Fock matrix, etc """

    # run pyscf for some reason
    basis = 'augccpvdz'
    mol = pyscf.M(
        atom='F 0 0 0; F 0 0 {}'.format(6.000), #2.0749193),
        #atom='Be 0 0 0'
        verbose=5,
        unit='b',
        basis=basis)



    mf = mol.UHF()
    mf.run()
    orb=mf.mo_coeff
    print(np.shape(mf.get_hcore()))
    print(np.shape(orb))
    import run_ccd as run_ccd 
 
    run_ccd.ccd_main(mf,mol,orb)


main()
