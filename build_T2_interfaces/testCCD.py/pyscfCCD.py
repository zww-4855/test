#!/usr/bin/env python

'''
Orbital optimized MP2

See also pyscf/example/cc/42-as_casci_fcisolver.py
'''

import numpy
from pyscf import gto, scf, mp, mcscf
from pyscf import cc
from pyscf import ao2mo
import sys
mol = gto.M(atom = 'H 0 0 0; F 0 0 1.2',
            basis = 'ccpvdz',
            verbose = 4)
mf = scf.RHF(mol).run()


mycc = cc.CCSD(mf)
old_update_amps = mycc.update_amps
def update_amps(t1, t2, eris):
    t1, t2 = old_update_amps(t1, t2, eris)
    return t1*0, t2
mycc.update_amps = update_amps
mycc.kernel()

print('OO-MP2 based CCD correlation energy', mycc.e_corr)
