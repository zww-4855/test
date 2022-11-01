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
    #atom='Be 0 0 0',
    #basis='6-31G')
    #atom = 'H 0 0 0; H 0 0 0.7414',    
    #atom= 'H 0 0 0; B 0 0 1.14',
    atom= 'O -3.10201284    1.32308063   -0.01721551;H -2.15940902    1.35214583   -0.01721551;H -3.38926240    2.22132026   -0.01721551',
    basis = 'cc-pvdz')


mf=scf.RHF(mol)
mf.conv_tol=10E-13
mf.conv_tol_grad=10E-12
mf.kernel()
print('mo energy:', mf.mo_energy)
dm1=mf.make_rdm1()
coeff=mf.mo_coeff
fock=mf.get_fock()
#print('diag fock?:', (coeff.T@fock)@coeff)
#print('Orthogonalized?', coeff.T@coeff)
mo_occ=mf.mo_occ

ovrlap=mf.get_ovlp()
#print('MO ovrlap: ', (coeff.T@ovrlap)@coeff)
#print('C.T @ C: ', coeff.T@coeff)

u,v=np.linalg.eigh(ovrlap)
inx=u.argsort()
u=np.diag(u)#[inx])
#v=v[:,inx]
tmp=np.matmul(np.matmul(v,u),v.T)#(v@u)@v.T
print('all close',np.allclose(tmp,ovrlap))
print('tmp',tmp)
print('overlap:',ovrlap)
print(np.trace(ovrlap),np.trace(v))

mycc = cc.CCSD(mf)
mycc.conv_tol=10E-9
mycc.conv_tol_normt=10E-10
mycc.kernel()

t1=mycc.t1
nocc,nvirt=t1.shape
print('max of t1: ',np.max(np.abs(t1)))

import sys
def orthogonalizeMO(mf,C):
    ovrlap=mf.get_ovlp()
    print('AO overlap: ', ovrlap)
    # see if i can reconstruct the overlap matrix
    u,v=np.linalg.eigh(ovrlap)
    inx=u.argsort()
    u=np.diag(u)#[inx])
    #v=v[:,inx]
    tmp=(v@u)@v.T
    print('test to reconstruct:', np.allclose(tmp,ovrlap))
    print('reconstruct:',tmp)

    ovrlapMO=(C.T@ovrlap)@C
    print('overlap MO', ovrlapMO)
    #sys.exit()
    roots,vecs=np.linalg.eigh(ovrlapMO)
    indx = roots.argsort()
    print('index:', indx)
    roots_trans=np.diag(roots[indx])
    vecs = vecs[:, indx]
    product=vecs@(roots_trans@vecs.T)
    print(np.allclose(product, ovrlapMO))
    #sys.exit()
    print('ordering of roots: ', np.diag(roots))
    print('orig roots:',np.diag(roots),np.diag(np.sqrt(roots)))
    roots=roots[indx]
    newroots=np.zeros((len(roots),len(roots)))
    for i in range(len(roots)):
        newroots[i][i]=1.0/np.sqrt(roots[i])
    print('roots', newroots)

    X=(vecs@newroots)@vecs.T
    print('canonical orth. transform:', X)
    print('vecs.T @ vecs:', vecs.T@vecs) 
    orthog_C=C@X
    print('ortho C ovrlap: ', (orthog_C.T@ovrlap)@orthog_C)
    print('diag should be 1: ', np.diag((orthog_C.T@ovrlap)@orthog_C))
    print('verify C^tSC==1',np.allclose((orthog_C.T@ovrlap)@orthog_C,np.eye(X.shape[0])))
    #sys.exit()
    return orthog_C


import sys

def T1rotate(coeff,t1):
    newcoeff=coeff
    nocc,nvirt=t1.shape
    print(t1.shape)
    print('old coeff:', coeff)
    C_occ=coeff[:,:nocc]
    C_virt=coeff[:,nocc:]
    print('shape of nocc and nvirt coeff:', np.shape(C_occ),np.shape(C_virt))
    C_virt=C_virt-C_occ@t1
    C_occ=C_occ+C_virt@t1.T
    newcoeff=np.zeros(coeff.shape)
    newcoeff[:,:nocc]=C_occ
    newcoeff[:,nocc:]=C_virt
    print('new coeff:', newcoeff)
    #newcoeff[:,nocc:]=newcoeff[:,nocc:]-coeff[:,:nocc]@t1
    #newcoeff[:,:nocc]=newcoeff[:,:nocc]+coeff[:,nocc:]@t1.T

    return newcoeff

def defineMOL():
    mol = gto.M(verbose = 4,
    #atom='Be 0 0 0',
    #basis='6-31G')
    #atom = 'H 0 0 0; H 0 0 0.7414',
    #atom= 'H 0 0 0; B 0 0 1.14',
    atom= 'O -3.10201284    1.32308063   -0.01721551;H -2.15940902    1.35214583   -0.01721551;H -3.38926240    2.22132026   -0.01721551',
    basis = 'cc-pvdz')
    return mol

def run_brueckner(mf,t1,coeff,nocc,mo_occ,t1TOL=10E-4):
    t1norm=np.linalg.norm(t1)
    t1matShape=np.shape(coeff)
#    while t1norm > t1TOL:
    newC=coeff
    for i in range(2):
        print('norm of t1:',t1norm)
        newC=T1rotate(coeff,t1)
#        newC=newC+newC@t1mat
        #newC=coeff
        newC=orthogonalizeMO(mf,newC)
        #newC,r=np.linalg.qr(newC)
        #S=mf.get_ovlp()
        #print('ovrlap C.T @ S @ C QR:', (newC.T@S)@newC)
        #sys.exit()

        mol = gto.M(verbose = 4,
        #atom='Be 0 0 0',
        #basis='6-31G')
        #atom = 'H 0 0 0; H 0 0 0.7414',
        #atom= 'H 0 0 0; B 0 0 1.14',
        atom= 'O -3.10201284    1.32308063   -0.01721551;H -2.15940902    1.35214583   -0.01721551;H -3.38926240    2.22132026   -0.01721551',
        basis = 'cc-pvdz')


        mf=scf.RHF(mol)
        dm1=mf.make_rdm1(newC,mo_occ)
        mf.init_guess=dm1
        mf.run(max_cycle=0)
        print('mo energy:', mf.mo_energy)
        #newC=mf.mo_coeff
        #sys.exit()
        #eri=ao2mo.kernel(mol, newC)
        mycc=cc.CCSD(mf)
        mycc.kernel()
        t1=mycc.t1
        print('max of t1: ',np.max(np.abs(t1)))
        t1norm=np.linalg.norm(t1)

    return mf,dm1
   

mf,dm1=run_brueckner(mf,t1,coeff,nocc,mo_occ)


