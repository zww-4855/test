#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
To avoid recomputing AO to MO integral transformation, integrals for CCSD,
CCSD(T), CCSD lambda equation etc can be reused.
'''

import numpy
import numpy as np 
from pyscf import gto, scf, cc

mol = gto.M(verbose = 4,
    atom = 'H 0 0 0; H 0 0 1.1',
    basis = 'ccpvdz')


mf=scf.RHF(mol)
mf.kernel()
dm1=mf.make_rdm1()
coeff=mf.mo_coeff
mo_occ=mf.mo_occ

mycc = cc.CCSD(mf)
mycc.kernel()

t1=mycc.t1
print('initial t1:',t1)
nocc,nvirt=t1.shape





def run_brueckner(mf,t1,coeff,nocc,mo_occ,t1TOL=10E-4):
    t1norm=np.linalg.norm(t1)
    t1matShape=np.shape(coeff)
#    while t1norm > t1TOL:
    for i in range(35):
        print('norm of t1:',t1norm)
        newC=coeff
        t1mat=np.zeros(t1matShape)
        print('t1mat:',t1mat.shape,t1.shape,nocc,mo_occ)
        t1mat[:nocc,nocc:]=-1.0*t1
        t1mat[nocc:,:nocc]=1.0*t1.T
        newC=coeff+coeff@t1mat
        newC,r=np.linalg.qr(newC)
        print('newC:',newC)
        mf=scf.RHF(mol)
        dm1=mf.make_rdm1(newC,mo_occ)
        mf.init_guess=dm1
        mf.run(max_cycle=0)
       

        mycc=cc.CCSD(mf)
        mycc.kernel()
        t1=mycc.t1
        print('t1:',t1)
        print('max of t1: ',np.max(t1))
        t1norm=np.linalg.norm(t1)
        print('****final t1norm:',t1norm)

    print('new C: ', newC)   
    return mf,dm1
   

mf,dm1=run_brueckner(mf,t1,coeff,nocc,mo_occ)
mf.init_guess=dm1
mf.run(max_cycle=0)   


mycc=cc.CCSD(mf)
old_update_amps = mycc.update_amps
def update_amps(t1, t2, eris):
    t1, t2 = old_update_amps(t1, t2, eris)
    print('ccsd new t1', t1)
    return t1, t2
mycc.update_amps = update_amps
mycc.kernel()

print('CCSD correlation energy', mycc.e_corr)
 
