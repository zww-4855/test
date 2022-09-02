#!/usr/bin/env python

'''
Scan H2 molecule dissociation curve.
See also 30-scan_pes.py
'''

import numpy
from pyscf import scf
from pyscf import gto
from pyscf import ao2mo

mol = gto.M(atom=[["H", 0., 0., 0.],
                  ["H", 0., 0., 1.2 ]], basis='STO-6G', verbose=3)
mf = scf.RHF(mol)
mf.kernel()

fock=mf.get_fock()
print('fock',fock)

hcore=mf.get_hcore()
print('hcore:',hcore)

# Transform 1 and 2 electron integrals from AO to MO basis 
orb=mf.mo_coeff
eri=ao2mo.kernel(mol, orb)
print('eri:', eri)

eri8 = ao2mo.restore('s1', eri, orb.shape[1])
print(eri8.shape)
print('eri8:',eri8)

hcoreMO=orb.T @ hcore @ orb

print('hcoreMO:', hcoreMO)



nelec=mol.nelectron
n=int(nelec/2) # defined 'n' according to J. Chem. Phys. 36, 2247 (1962)
print('number of electrons',nelec)


def get_E_Ebar(hcoreMO,n):
    Ebar=0.0
    E=0.0
    for i in range(n):
        E+=hcoreMO[i,i]
        Ebar+=hcoreMO[n+i,n+i]
    return E,Ebar

E,Ebar=get_E_Ebar(hcoreMO,n)
print("check: ",E,2*E,2*hcoreMO[0,0])

def get_W(E,Ebar):
    W=E+Ebar
    delW=Ebar-E
    return W,delW

W,delW=get_W(E,Ebar)

print('W,Delta W', W,delW)

#https://realpython.com/python-scipy-cluster-optimize/
def objective_function(x,W,delW,n):
     sec_term=((n+1)/n)*((1-x**(2*n))/(1-x**(2*n+2)))*x*delW
     return W-sec_term


def tmp(x,W,delW,n):
     sec_term=((n+1)/n)*((1-x**(2*n))/(1-x**(2*n+2)))*x*delW
     return W-sec_term

from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
res = minimize(objective_function,x0=0.7,args=(W,delW,n))
#res = minimize_scalar(objective_function,args=(W,delW,n))
print('res',res)

import numpy as np
pts=np.linspace(0,1,100)
print()
for ii in range(100):
    print(pts[ii],'  ',objective_function(pts[ii],W,delW,n))
