import numpy as np
import pyscf
from pyscf import ao2mo


def main():
    """ Initialize calculation details, 2e- integrals, Fock matrix, etc """

    # run pyscf for some reason
    basis = 'ccpvdz'
    mol = pyscf.M(
        atom='N 0 0 0; N 0 0 {}'.format(2.0749193),
        #atom='Be 0 0 0'
        verbose=5,
        unit='b',
        basis=basis)


    mf = mol.UHF()
    mf.init_guess='1e'
    mf.conv_tol_grad=1E-10
    mf.run()

    from pyscf.lib import logger

    def stable_opt_internal(mf):
        log = logger.new_logger(mf)
        mo1, _, stable, _ = mf.stability(return_status=True)
        cyc = 0
        while (not stable and cyc < 10):
            log.note('Try to optimize orbitals until stable, attempt %d' % cyc)
            dm1 = mf.make_rdm1(mo1, mf.mo_occ)
            mf = mf.run(dm1)
            mo1, _, stable, _ = mf.stability(return_status=True)
            cyc += 1
        if not stable:
            log.note('Stability Opt failed after %d attempts' % cyc)
        return mf
    print('loop example')
    mf = stable_opt_internal(mf)

    orb=mf.mo_coeff
    print(np.shape(mf.get_hcore()))
    print(np.shape(orb))
    import run_ccd as run_ccd 
 
    cc_runtype={"ccdType":"ccd"}

    run_ccd.ccd_main(mf,mol,orb,cc_runtype)


main()
