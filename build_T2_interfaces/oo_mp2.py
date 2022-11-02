#!/usr/bin/env python

'''
Orbital optimized MP2

See also pyscf/example/cc/42-as_casci_fcisolver.py
'''

import numpy
from pyscf import gto, scf, mp, mcscf
from pyscf import cc

class MP2AsFCISolver(object):
    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        fakemol = gto.M(verbose=0)
        fakemol.nelectron = sum(nelec)

        # Build a mean-field object fake_hf without SCF iterations
        fake_hf = scf.RHF(fakemol)
        fake_hf._eri = h2
        fake_hf.get_hcore = lambda *args: h1
        fake_hf.get_ovlp = lambda *args: numpy.eye(norb)
        fake_hf.mo_coeff = numpy.eye(norb)
        fake_hf.mo_occ = numpy.zeros(norb)
        fake_hf.mo_occ[:fakemol.nelectron//2] = 2

        self.mp2 = mp.MP2(fake_hf)
        e_corr, t2 = self.mp2.kernel()
        e_tot = self.mp2.e_tot + ecore
        return e_tot, t2

    def make_rdm1(self, t2, norb, nelec):
        return self.mp2.make_rdm1(t2)

    def make_rdm12(self, t2, norb, nelec):
        dm1 = self.mp2.make_rdm1(t2)
        dm2 = self.mp2.make_rdm2(t2)
        return dm1, dm2


############
# This section goes inside calling program
mol = gto.M(atom = 'H 0 0 0; F 0 0 1.2',
            basis = 'ccpvdz',
            verbose = 4)
mf = scf.RHF(mol).run()
#############
# Call Run_OO_MP2(mf,mol)
mo_occ=mf.mo_occ
norb = mf.mo_coeff.shape[1]
nelec = mol.nelectron
mc = mcscf.CASSCF(mf, norb, nelec)
mc.fcisolver = MP2AsFCISolver()
# Internal rotation needs to be enabled so that orbitals are optimized in
# active space which were modeled by MP2
mc.internal_rotation = True
mc.kernel()


convg_C=mc.mo_coeff
print(convg_C)



### Returns set of new, optimized MO coefficients

mf2=scf.RHF(mol)
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

print('CCD correlation energy', mycc.e_corr)
