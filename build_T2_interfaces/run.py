#!/usr/bin/env python

'''
Orbital optimized MP2

See also pyscf/example/cc/42-as_casci_fcisolver.py
'''

import numpy
from pyscf import gto, scf, mp, mcscf
from pyscf import cc
from oo_mp2 import *
from initialize_cc import *
import ultT2_cc
from pyscf import ao2mo
import sys
mol = gto.M(atom = 'H 0 0 0; B 0 0 0.9',
            basis = 'ccpvdz',
            verbose = 4)
mf = scf.RHF(mol).run()

# Call OO-MP2 routine
#convg_C=Run_OO_MP2(mf,mol)
convg_C=mf.mo_coeff

#mf.mo_coeff=convg_C

# Initialize CC method; harvest 1/2 e- integrals
nsvirt, nsocc, fock, g, o, v, e_ai, e_abij, e_abcijk = Initialize_CC(mf,mol,convg_C)

# Call CC method based on T2
t2f = ultT2_cc.CCDkernel(np.zeros((nsvirt, nsocc)), np.zeros((nsvirt, nsvirt, nsocc, nsocc)), fock, g, o, v, e_ai, e_abij,mf.energy_nuc(),diis_size=8)


sys.exit()


# Use OO-MP2 orbitals to extract new correlation energy
mf2=scf.RHF(mol)
mo_occ=mf.mo_occ
print(mo_occ)
dm1=mf2.make_rdm1(convg_C,mo_occ)
mf2.init_guess=dm1
mf2.run(max_cycle=0)

mycc = cc.CCSD(mf2)
old_update_amps = mycc.update_amps
def update_amps(t1, t2, eris):
    t1, t2 = old_update_amps(t1, t2, eris)
    return t1*0, t2
mycc.update_amps = update_amps
mycc.kernel()

print('OO-MP2 based CCD correlation energy', mycc.e_corr)
