import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.fci


mol = pyscf.gto.Mole()
mol.atom = 'N 0 0 0 ; N 0 0 1.5'
mol.basis = 'STO-6G'
mol.spin = 4
mol.verbose=8
mol.build()

# Hartree-Fock
mf = pyscf.scf.HF(mol)
mf.kernel()
assert mf.converged

# FCI:
fci = pyscf.fci.FCI(mf)
fci.conv_tol = 1e-14
nelec = [sum(mo) for mo in mf.mo_occ]
sz = abs(nelec[0]-nelec[1])/2
ss = (sz)*(sz+1)
fci = pyscf.fci.addons.fix_spin_(fci, ss=1)
fci.kernel()
assert fci.converged

s2, mult = fci.spin_square(fci.ci, fci.norb, fci.nelec)
print("E(FCI)= %.8f  <S^2>= %f  muliplicity= %f" % (fci.e_tot, s2, mult))





