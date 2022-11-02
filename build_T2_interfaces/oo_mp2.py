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



def Run_OO_MP2(mf,mol):
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
    return convg_C



