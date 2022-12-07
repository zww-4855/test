from pyscf import fci
import pyscf
from functools import reduce
import numpy
from pyscf import gto, scf, ao2mo, fci
from pyscf.fci import cistring

mol = pyscf.M(
    atom = 'Be 0 0 0',  # in Angstrom
    basis = '6-31g',
    symmetry = True,
)
myhf = mol.RHF().run()

#
# create an FCI solver based on the SCF object
#
cisolver = pyscf.fci.FCI(myhf)
print('E(FCI) = %.12f' % cisolver.kernel()[0])


def kernel(h1e, g2e, norb, nelec, ecore=0):
    def absorb_h1e(h1e, eri, norb, nelec, fac=1):
        '''Modify 2e Hamiltonian to include 1e Hamiltonian contribution.
        '''
        if not isinstance(nelec, (int, numpy.integer)):
            nelec = sum(nelec)
        h2e = ao2mo.restore(1, eri.copy(), norb)
        f1e = h1e - numpy.einsum('jiik->jk', h2e) * .5
        f1e = f1e * (1./(nelec+1e-100))
        for k in range(norb):
            h2e[k,k,:,:] += f1e
            h2e[:,:,k,k] += f1e
        return h2e * fac


    h2e = absorb_h1e(h1e, g2e, norb, nelec, .5)

    na = cistring.num_strings(norb, nelec//2)
    ci0 = numpy.zeros((na,na))
    ci0[0,0] = 1

    # new code here
    neleca = nelecb = nelec//2
    nb = na
    ci_level = 2 # e.g. CISDT; make this kwarg if desired
    alpha_occs = fci.cistring._gen_occslst(range(norb), neleca)
    alpha_excitations = (alpha_occs >= neleca).sum(axis=1)
    beta_occs = fci.cistring._gen_occslst(range(norb), nelecb)
    beta_excitations = (beta_occs >= nelecb).sum(axis=1)
    a_idx, b_idx = numpy.array([
        [a,b] for a in alpha_excitations for b in beta_excitations if a+b > ci_level ]).T

    print('a indx:', a_idx)
    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec)
        # new code here
        hc = hc.reshape(na,nb)
        hc[a_idx,b_idx] = 0
        return hc.reshape(-1)
    hdiag = make_hdiag(h1e, g2e, norb, nelec)
    # new code here
    hdiag = hdiag.reshape(na,nb)
    hdiag[a_idx,b_idx] = 0
    hdiag = hdiag.reshape(-1)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    e, c = lib.davidson(hop, ci0.reshape(-1), precond)
    return e+ecore

c = myhf.mo_coeff
h1e = reduce(numpy.dot, (c.T, myhf.get_hcore(), c))
eri = ao2mo.incore.full(myhf._eri, c)
norb= c.shape[1]
nelec= mol.nelectron
print('norb:',norb,nelec)
cisolver = pyscf.fci.FCI(myhf)
cisolver.kernel=kernel
print('E(FCI) = %.12f' % cisolver.kernel(h1e,eri,norb,nelec)[0])
