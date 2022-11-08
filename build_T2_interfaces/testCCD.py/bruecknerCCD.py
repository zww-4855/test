import pyscf
from pyscf import cc
import numpy
from scipy import linalg as la

from pyscf import lib

from pyscf.cc.bccd import bccd_kernel_
#import 
numpy.set_printoptions(3, linewidth=1000, suppress=True)
mol = pyscf.M(
#    atom = 'H 0 0 0; F 0 0 1.1',
     atom='H 0.00000000    -0.75299460     0.50421485; O -0.00000000     0.00000000    -0.06354025;H -0.00000000     0.75299460     0.50421485',
    basis = 'ccpvdz',
    verbose = 4,
    spin = 0,
)

myhf = mol.HF()
myhf.kernel()
E_ref = myhf.e_tot
rdm1_mf = myhf.make_rdm1()

mycc = cc.CCSD(myhf)
mycc.kernel()

mycc = bccd_kernel_(mycc, diis=True, verbose=4)

print (la.norm(mycc.t1))
assert la.norm(mycc.t1) < 1e-5
