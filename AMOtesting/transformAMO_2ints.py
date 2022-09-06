#!/usr/bin/env python

'''
Scan H2 molecule dissociation curve.
See also 30-scan_pes.py
'''

import numpy
from pyscf import scf
from pyscf import gto
from pyscf import ao2mo
import numpy as np

#U0=sum_k gamma_kk  ie (kk|kk)
def build_U(eri,n,barShift=0):
    U0=U1=U2=Utot=0.0

    for i in range(n):
        k=i+barShift
        U0+=eri[k,k,k,k]

    # j<i
    for i in range(n):
        for j in range(i):
            U1+=eri[j,j,i,i]

    for i in range(n):
        for j in range(i):
            U2+=eri[j,i,i,j]

    U1=U2=0.0 # ONLY VALID FOR H2 MINIMAL BASIS EXAMPLE
    Utot=U0+4.0*U1-2.0*U2

    return Utot,U0,U1,U2



def build_X(eri,n,barShift=1):
    X0=X1=X2=Xtot=0.0
    for i in range(n):
        X0+=eri[i,i,i+barShift,i+barShift]


    for i in range(n):
        for j in range(i):
            X1+=eri[i,i,j,j]+eri[j,j,i,i]

    X1=X1*0.5
    print('x1 is: ', X1) 

    X2=X1=0.0 # ONLY VALID FOR H2 MINIMAL BASIS EXAMPLE


    Xtot=X0+4.0*X1-2.0*X2

    return Xtot,X0,X1,X2


def build_Y(eri,n, barShift=1):
    Y0=Y0prime=Y1=Y2=Ytot=0.0

    for i in range(n):
        k=i+barShift
        Y0+=eri[i,k,i,k]
        Y0prime+=eri[i,k,k,i]

    for i in range(n):
        k=i+barShift
        for j in range(k,i):
            Y1+=eri[i,j,j,i]+eri[j,i,i,j]

    Y1=Y1*0.5
    print('y1 is: ', Y1)
     

    for i in range(n):
        k=i+barShift
        for j in range(k,2*n):  
            Y2+=eri[i,j,i,j]

    Ytot=0.5*Y0+Y2

    return Ytot,Y0,Y0prime,Y1,Y2


#build eqn 48
def build_A(n, U1,U2,U1bar,U2bar,X1,X2,X,Y0prime,Y1):
    firstTerm=U1+U1bar-2.0*X1

    secondTerm=U2+U2bar-2.0*X2

    thirdTerm=0.5*(X-Y0prime-2.0*Y1)  


    A=firstTerm-secondTerm
    A=A/(n-1)
    A=A-thirdTerm

    return A



    

# 'x' is lambda
def objective_function(x,n,A,U,Ubar,Y):
     E12=-2.0*A
     firstTerm=((n+1)/n)*((1-x**(2*n))/(1-x**(2*n+2)))
     secondTerm=(((1+x)/2)**2)*U + (((1-x)/2)**2)*Ubar + (1+x**2)*A - (1-x**2)*Y
     

     E12+=firstTerm*secondTerm

     return E12


