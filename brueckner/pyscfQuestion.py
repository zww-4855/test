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
    #atom = 'H 0 0 0; H 0 0 0.7414',    
    atom= 'H 0 0 0; B 0 0 1.14',
    basis = 'ccpvdz')


mf=scf.RHF(mol)
mf.conv_tol=10E-13
mf.conv_tol_grad=10E-12
mf.kernel()
dm1=mf.make_rdm1()
coeff=mf.mo_coeff
mo_occ=mf.mo_occ

mycc = cc.CCSD(mf)
mycc.conv_tol=10E-11
mycc.conv_tol_normt=10E-12
mycc.kernel()

t1=mycc.t1
nocc,nvirt=t1.shape


def orthogonalizeMO(mf,C):
    ovrlap=mf.get_ovlp()
    ovrlapMO=(C.T@ovrlap)@C
    roots,vecs=np.linalg.eig(ovrlapMO)
    indx = roots.argsort()
    print('orig roots:',np.diag(roots))
    newroots=np.zeros((len(indx),len(indx)))
    for i in range(len(indx)):
        newroots[i][i]=1.0/np.sqrt(roots[i])
    print('roots', newroots)
    vecs = vecs[:, indx]
    X=(vecs@newroots)@vecs.T
    print('canonical orth. transform:', X)
    orthog_C=C@X
    print('ortho C: ', orthog_C)
    return orthog_C


import sys

def T1rotate(coeff,t1):
    newcoeff=coeff
    nocc,nvirt=t1.shape
    print(t1.shape)
    newcoeff[:,nocc:]=newcoeff[:,nocc:]-coeff[:,:nocc]@t1
    newcoeff[:,:nocc]=newcoeff[:,:nocc]+coeff[:,nocc:]@t1.T

    return newcoeff

def run_brueckner(mf,t1,coeff,nocc,mo_occ,t1TOL=10E-4):
    t1norm=np.linalg.norm(t1)
    t1matShape=np.shape(coeff)
#    while t1norm > t1TOL:
    newC=coeff
    for i in range(25):
        print('norm of t1:',t1norm)
        t1mat=np.zeros(t1matShape)
        t1mat[:nocc,nocc:]=-1.0*t1
        t1mat[nocc:,:nocc]=1.0*t1.T
        print('t1 mat: ', t1mat)
        newC=T1rotate(coeff,t1)
#        newC=newC+newC@t1mat
        #newC=orthogonalizeMO(mf,newC)
#        newC,r=np.linalg.qr(newC)
        #sys.exit()
        mf=scf.RHF(mol)
        dm1=mf.make_rdm1(newC,mo_occ)
        mf.init_guess=dm1
        mf.run(max_cycle=0)
        print('mo energy:', mf.mo_energy)
        newC=mf.mo_coeff
        #sys.exit()

        mycc=cc.CCSD(mf)
        mycc.kernel()
        t1=mycc.t1
        print('max of t1: ',np.max(np.abs(t1)))
        t1norm=np.linalg.norm(t1)

    return mf,dm1
   

mf,dm1=run_brueckner(mf,t1,coeff,nocc,mo_occ)


