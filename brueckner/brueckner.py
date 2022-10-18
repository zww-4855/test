#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
To avoid recomputing AO to MO integral transformation, integrals for CCSD,
CCSD(T), CCSD lambda equation etc can be reused.
'''

import numpy 
from pyscf import gto, scf, cc

mol = gto.M(verbose = 4,
    atom = 'H 0 0 0; H 0 0 1.1',
    basis = 'ccpvdz')


mf=scf.RHF(mol)
mf.chkfile='coeff'
mf.kernel()
dm1=mf.make_rdm1()
print('converged DM',dm1)
#print(scf.hf.energy_elec(mf, dm1))
#mf = scf.RHF(mol).run()
coeff=mf.mo_coeff
mo_occ=mf.mo_occ

print('mo_occ:',mo_occ)
dm2=mf.make_rdm1(coeff,mo_occ)
print(dm1==dm2)

print(coeff[:,mo_occ>0])

#mf.init_guess=dm2
#mf.kernel(max_cycle=0)

#mycc = cc.CCSD(mf)
#mycc.kernel()

#t1=mycc.t1
#print('t1:',t1)
#print('mo coeff',coeff)
#nocc,nvirt=t1.shape
#print('nocc:',nocc)
#print('shape of coeff:', coeff.shape)





def run_brueckner(mf,t1,coeff,nocc):
    dm_init_guess=coeff[:,:nocc]@coeff[:,:nocc].T
    mf=scf.RHF(mol)#.run(dm0=dm_init_guess,max_cycle=0)
    #dm_init_guess=#mf.make_rdm1(coeff,nocc)
    mf.init_guess=dm_init_guess
    mf.run(max_cycle=0)
    #mf.kernel(dm0=dm_init_guess,max_cycle=0)

#run_brueckner(mf,t1,coeff,nocc)
    
