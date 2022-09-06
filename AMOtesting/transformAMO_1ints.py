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

def get_E_Ebar(hcoreMO,n):
    Ebar=0.0
    E=0.0
    for i in range(n):
        E+=hcoreMO[i,i]
        Ebar+=hcoreMO[n+i,n+i]
    return E,Ebar


def get_W(E,Ebar):
    W=E+Ebar
    delW=Ebar-E
    return W,delW


#https://realpython.com/python-scipy-cluster-optimize/
def get_E1(x,W,delW,n):
     sec_term=((n+1)/n)*((1-x**(2*n))/(1-x**(2*n+2)))*x*delW
     return W-sec_term


