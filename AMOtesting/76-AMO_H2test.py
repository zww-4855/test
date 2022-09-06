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


#eri_J, eri_K=mf.get_jk()

#print('j mat: ', eri_J)

#import sys
#sys.exit()
hcoreMO=orb.T @ hcore @ orb

print('hcoreMO:', hcoreMO)



nelec=mol.nelectron
n=int(nelec/2) # defined 'n' according to J. Chem. Phys. 36, 2247 (1962)
print('number of electrons',nelec)
print('n:',n)


from transformAMO_1ints import *
E,Ebar=get_E_Ebar(hcoreMO,n)
print("check: ",E,2*E,2*hcoreMO[0,0])
W,delW=get_W(E,Ebar)

print('W,Delta W', W,delW)












import sys as sys 
sys.exit()
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


## NOW FIT 2 ELECTRON INTS
from transformAMO_2ints import *
eri=eri8
Utot,U0,U1,U2 = build_U(eri,n,0)
Utotbar,U0bar,U1bar,U2bar = build_U(eri,n,1)
Xtot,X0,X1,X2 =build_X(eri,n,1)
Ytot,Y0,Y0prime,Y1,Y2=build_Y(eri,n, 1)
A=build_A(n, U1,U2,U1bar,U2bar,X1,X2,Xtot,Y0prime,Y1)




#import numpy as np
#pts=np.linspace(0,1,100)
#print()
#for ii in range(100):
#    print(pts[ii],'  ',objective_function(pts[ii],W,delW,n))
