from itertools import product
import pyscf
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermionpyscf import run_pyscf
from pyscf.cc.addons import spatial2spin
import numpy as np
from pyscf import ao2mo

basis = 'cc-pvdz'
mol = pyscf.M(
    atom='H 0 0 0; B 0 0 {}'.format(1.600),
    basis=basis)

mf = mol.RHF().run()
hcore=mf.get_hcore()
hf_C=mf.mo_coeff
hcoreMO=(hf_C.T@hcore)@hf_C


molecule = of.MolecularData(geometry=[['H', (0, 0, 0)], ['B', (0, 0, 1.6)]],
                            basis=basis, charge=0, multiplicity=1)
molecule = run_pyscf(molecule, run_ccsd=True)
oei, tei = molecule.get_integrals()

print(oei[:5,:5],'\n',hcoreMO[:5,:5])
print(np.allclose(oei,hcoreMO))
print(oei==hcoreMO)

print(oei[0,1],hcoreMO[0,1])



twoEints=ao2mo.kernel(mol,mf.mo_coeff)
two_electron_integrals = ao2mo.restore(
        1, # no permutation symmetry
        twoEints, hf_C.shape[0])
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
two_electron_integrals = np.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')


print('tei',two_electron_integrals.shape,hf_C.shape[0])
print('mine',twoEints.shape,two_electron_integrals.shape,np.allclose(two_electron_integrals,twoEints))

## THESE INTEGRALS SHOULD ALREADY BE IN MO FRAMEWORK
