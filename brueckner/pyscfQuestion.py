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
import sys
from pyscf import ao2mo

mol = gto.M(verbose = 4,
    atom= 'O -3.10201284    1.32308063   -0.01721551;H -2.15940902    1.35214583   -0.01721551;H -3.38926240    2.22132026   -0.01721551',
    basis = 'cc-pvdz')


mf=scf.RHF(mol)
mf.conv_tol=10E-13
mf.conv_tol_grad=10E-12
mf.kernel()
coeff=mf.mo_coeff
mo_occ=mf.mo_occ


mycc = cc.CCSD(mf)
mycc.conv_tol=10E-9
mycc.conv_tol_normt=10E-10
mycc.kernel()

t1=mycc.t1
nocc,nvirt=t1.shape


def orthogonalizeMO(mf,C):
    ovrlap=mf.get_ovlp()
    ovrlapMO=(C.T@ovrlap)@C
    roots,vecs=np.linalg.eigh(ovrlapMO)
    indx = roots.argsort()
    vecs = vecs[:, indx]
    roots=roots[indx]
    newroots=np.zeros((len(roots),len(roots)))
    for i in range(len(roots)):
        newroots[i][i]=1.0/np.sqrt(roots[i])

    X=(vecs@newroots)@vecs.T
    orthog_C=C@X
    print('Verify C^tSC==1',np.allclose((orthog_C.T@ovrlap)@orthog_C,np.eye(X.shape[0])))
    return orthog_C


# Performs Brueckner rotation on occ/virt orbitals 
def T1rotate(coeff,t1):
    newcoeff=coeff
    nocc,nvirt=t1.shape

    C_occ=coeff[:,:nocc]
    C_virt=coeff[:,nocc:]
    C_virtold=C_virt

    C_virt=C_virt-C_occ@t1
    C_occ=C_occ+C_virtold@t1.T

    newcoeff=np.zeros(coeff.shape)
    newcoeff[:,:nocc]=C_occ
    newcoeff[:,nocc:]=C_virt

    return newcoeff


def run_brueckner(mf,t1,coeff,nocc,mo_occ,t1TOL=10E-4):
    t1norm=np.linalg.norm(t1)
    newC=coeff
    for i in range(4):
        print('Norm of t1:',t1norm)
        newC=T1rotate(newC,t1)
        newC=orthogonalizeMO(mf,newC)

        # Rebuild Fock matrix
        mf=scf.RHF(mol)
        dm1=mf.make_rdm1(newC,mo_occ)
        mf.init_guess=dm1
        mf.run(max_cycle=0)

        # Converge CCSD eqns & extract new T1
        mycc=cc.CCSD(mf)
        mycc.kernel()
        t1=mycc.t1

        print('max of t1: ',np.max(np.abs(t1)))
        t1norm=np.linalg.norm(t1)

    return mf,dm1
   

mf,dm1=run_brueckner(mf,t1,coeff,nocc,mo_occ)


