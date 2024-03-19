"""
A full working spin-orbital CCSD code generated with pdaggerq

If you want to run the example here you should install pyscf openfermion and openfermion-pyscf
The actual CCSD code (ccsd_energy, singles_residual, doubles_residual, kernel)
do not depend on those packages but you gotta get integrals frome somehwere.

We also check the code by using pyscfs functionality for generating spin-orbital
t-amplitudes from RCCSD.  the main() function is fairly straightforward.
"""
import numpy as np
from numpy import einsum


def ccsd_energy(t1, t2, f, g, o, v):
    """
    < 0 | e(-T) H e(T) | 0> :

    :param f:
    :param g:
    :param t1:
    :param t2:
    :param o:
    :param v:
    :return:
    """
    #	  1.0000 f(i,i)
    energy = 1.0 * einsum('ii', f[o, o])

    #	  1.0000 f(i,a)*t1(a,i)
    energy += 1.0 * einsum('ia,ai', f[o, v], t1)

    #	 -0.5000 <j,i||j,i>
    energy += -0.5 * einsum('jiji', g[o, o, o, o])

    #	  0.2500 <j,i||a,b>*t2(a,b,j,i)
    energy += 0.25 * einsum('jiab,abji', g[o, o, v, v], t2)

    #	 -0.5000 <j,i||a,b>*t1(a,i)*t1(b,j)
    #energy += -0.5 * einsum('jiab,ai,bj', g[o, o, v, v], t1, t1,
    #                        optimize=['einsum_path', (0, 1), (0, 1)])



    t2_dag=t2.transpose(2,3,0,1)
    t2_resid=offset_t2resid_t2dag_w_t2(t2,t2_dag,g,o,v)
    #t2_resid=t2_resid*ccd_kernel.denom["D2aa"]
    print('CCD portion of E:',energy)
    print('UCC portion of E:',np.einsum('jiab,abji',t2_dag,t2_resid))
    totalE=energy-0.5*np.einsum('jiab,abji',t2_dag,t2_resid)
    print('UCC(4) energy is: ', totalE)

    return totalE #energy


def singles_residual(t1, t2, f, g, o, v):
    """
    < 0 | m* e e(-T) H e(T) | 0>
    :param f:
    :param g:
    :param t1:
    :param t2:
    :param o:
    :param v:
    :return:
    """
    #	  1.0000 f(e,m)
    singles_res = 1.0 * einsum('em->em', f[v, o])

    #	 -1.0000 f(i,m)*t1(e,i)
    singles_res += -1.0 * einsum('im,ei->em', f[o, o], t1)

    #	  1.0000 f(e,a)*t1(a,m)
    singles_res += 1.0 * einsum('ea,am->em', f[v, v], t1)

    #	 -1.0000 f(i,a)*t2(a,e,m,i)
    #singles_res += -1.0 * einsum('ia,aemi->em', f[o, v], t2)

    #	 -1.0000 f(i,a)*t1(a,m)*t1(e,i)
    #singles_res += -1.0 * einsum('ia,am,ei->em', f[o, v], t1, t1,
                                 #optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,m>*t1(a,i)
    #singles_res += 1.0 * einsum('ieam,ai->em', g[o, v, v, o], t1)

    #	 -0.5000 <j,i||a,m>*t2(a,e,j,i)
    singles_res += -0.5 * einsum('jiam,aeji->em', g[o, o, v, o], t2)

    #	 -0.5000 <i,e||a,b>*t2(a,b,m,i)
    singles_res += -0.5 * einsum('ieab,abmi->em', g[o, v, v, v], t2)

    #	  1.0000 <j,i||a,m>*t1(a,i)*t1(e,j)
    #singles_res += 1.0 * einsum('jiam,ai,ej->em', g[o, o, v, o], t1, t1,

    #	  1.0000 <i,e||a,b>*t1(a,i)*t1(b,m)
    #singles_res += 1.0 * einsum('ieab,ai,bm->em', g[o, v, v, v], t1, t1,

    #	  1.0000 <j,i||a,b>*t1(a,i)*t2(b,e,m,j)
    #singles_res += 1.0 * einsum('jiab,ai,bemj->em', g[o, o, v, v], t1, t2,

    #	  0.5000 <j,i||a,b>*t1(a,m)*t2(b,e,j,i)
    #singles_res += 0.5 * einsum('jiab,am,beji->em', g[o, o, v, v], t1, t2,

    #	  0.5000 <j,i||a,b>*t1(e,i)*t2(a,b,m,j)
    #singles_res += 0.5 * einsum('jiab,ei,abmj->em', g[o, o, v, v], t1, t2,

    #	  1.0000 <j,i||a,b>*t1(a,i)*t1(b,m)*t1(e,j)
    #singles_res += 1.0 * einsum('jiab,ai,bm,ej->em', g[o, o, v, v], t1, t1, t1,
    return singles_res


def doubles_residual(t1, t2, f, g, o, v):
    """
     < 0 | m* n* f e e(-T) H e(T) | 0>

    :param f:
    :param g:
    :param t1:
    :param t2:
    :param o:
    :param v:
    :return:
    """
    #	 -1.0000 P(m,n)f(i,n)*t2(e,f,m,i)
    contracted_intermediate = -1.0 * einsum('in,efmi->efmn', f[o, o], t2)
    doubles_res = 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate)

    #	  1.0000 P(e,f)f(e,a)*t2(a,f,m,n)
    contracted_intermediate = 1.0 * einsum('ea,afmn->efmn', f[v, v], t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->femn', contracted_intermediate)



    #	  1.0000 <e,f||m,n>
    doubles_res += 1.0 * einsum('efmn->efmn', g[v, v, o, o])

    #	  1.0000 P(e,f)<i,e||m,n>*t1(f,i)
    contracted_intermediate = 1.0 * einsum('iemn,fi->efmn', g[o, v, o, o], t1)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->femn', contracted_intermediate)

    #	  1.0000 P(m,n)<e,f||a,n>*t1(a,m)
    contracted_intermediate = 1.0 * einsum('efan,am->efmn', g[v, v, v, o], t1)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate)

    #	  0.5000 <j,i||m,n>*t2(e,f,j,i)
    doubles_res += 0.5 * einsum('jimn,efji->efmn', g[o, o, o, o], t2)

    #	  1.0000 P(m,n)*P(e,f)<i,e||a,n>*t2(a,f,m,i)
    contracted_intermediate = 1.0 * einsum('iean,afmi->efmn', g[o, v, v, o], t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate) + -1.00000 * einsum('efmn->femn',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'efmn->fenm', contracted_intermediate)

    #	  0.5000 <e,f||a,b>*t2(a,b,m,n)
    doubles_res += 0.5 * einsum('efab,abmn->efmn', g[v, v, v, v], t2)


    #    -0.5000 P(m,n)<j,i||a,b>*t2(a,b,n,i)*t2(e,f,m,j)
    contracted_intermediate = -0.5 * einsum('jiab,abni,efmj->efmn',
                                            g[o, o, v, v], t2, t2,
                                            optimize=['einsum_path', (0, 1),
                                                      (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate)

    #     0.2500 <j,i||a,b>*t2(a,b,m,n)*t2(e,f,j,i)
    doubles_res += 0.25 * einsum('jiab,abmn,efji->efmn', g[o, o, v, v], t2, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #    -0.5000 <j,i||a,b>*t2(a,e,j,i)*t2(b,f,m,n)
    doubles_res += -0.5 * einsum('jiab,aeji,bfmn->efmn', g[o, o, v, v], t2, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #     1.0000 P(m,n)<j,i||a,b>*t2(a,e,n,i)*t2(b,f,m,j)
    contracted_intermediate = 1.0 * einsum('jiab,aeni,bfmj->efmn',
                                           g[o, o, v, v], t2, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate)

    #    -0.5000 <j,i||a,b>*t2(a,e,m,n)*t2(b,f,j,i)
    doubles_res += -0.5 * einsum('jiab,aemn,bfji->efmn', g[o, o, v, v], t2, t2,
                                 optimize=['einsum_path', (0, 2), (0, 1)])

    return doubles_res

def t3_resid(t3,t2,f,g,o,v):
    #        -1.0000 P(j,k)f(l,k)*t3(a,b,c,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('lk,abcijl->abcijk', f[o, o], t3)
    triples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #        -1.0000 f(l,i)*t3(a,b,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('li,abcjkl->abcijk', f[o, o], t3)

    #         1.0000 P(a,b)f(a,d)*t3(d,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('ad,dbcijk->abcijk', f[v, v], t3)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #         1.0000 f(c,d)*t3(d,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('cd,dabijk->abcijk', f[v, v], t3)

    #        -1.0000 P(i,j)*P(a,b)<l,a||j,k>*t2(b,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lajk,bcil->abcijk', g[o, v, o, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #        -1.0000 P(a,b)<l,a||i,j>*t2(b,c,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('laij,bckl->abcijk', g[o, v, o, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #        -1.0000 P(i,j)<l,c||j,k>*t2(a,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lcjk,abil->abcijk', g[o, v, o, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #        -1.0000 <l,c||i,j>*t2(a,b,k,l)
    triples_res += -1.000000000000000 * einsum('lcij,abkl->abcijk', g[o, v, o, o], t2)

    #        -1.0000 P(j,k)*P(b,c)<a,b||d,k>*t2(d,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('abdk,dcij->abcijk', g[v, v, v, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #        -1.0000 P(b,c)<a,b||d,i>*t2(d,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abdi,dcjk->abcijk', g[v, v, v, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #        -1.0000 P(j,k)<b,c||d,k>*t2(d,a,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('bcdk,daij->abcijk', g[v, v, v, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #        -1.0000 <b,c||d,i>*t2(d,a,j,k)
    triples_res += -1.000000000000000 * einsum('bcdi,dajk->abcijk', g[v, v, v, o], t2)
    return triples_res

def offset_t2resid_t2squared(t2,g,o,v):
    #        -0.5000 P(i,j)<l,k||c,d>*t2(c,d,j,k)*t2(a,b,i,l)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,cdjk,abil->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #         0.2500 <l,k||c,d>*t2(c,d,i,j)*t2(a,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdij,ablk->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #        -0.5000 <l,k||c,d>*t2(c,a,l,k)*t2(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #         1.0000 P(i,j)<l,k||c,d>*t2(c,a,j,k)*t2(d,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,cajk,dbil->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #        -0.5000 <l,k||c,d>*t2(c,a,i,j)*t2(d,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,caij,dblk->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])

    return 0.5*doubles_res

def offset_t2resid_t2dag_w_t2(t2,t2_dag,g,o,v):
    #        -0.5000 P(i,j)<l,k||c,d>*t2(c,d,j,k)*t2(a,b,i,l)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,cdjk,abil->abij', t2_dag,g[v,v ,o,o],t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #         0.2500 <l,k||c,d>*t2(c,d,i,j)*t2(a,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdij,ablk->abij', t2_dag, g[v,v,o,o], t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #        -0.5000 <l,k||c,d>*t2(c,a,l,k)*t2(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij->abij', t2_dag, g[v,v,o,o], t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #         1.0000 P(i,j)<l,k||c,d>*t2(c,a,j,k)*t2(d,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,cajk,dbil->abij', t2_dag, g[v,v,o,o], t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #        -0.5000 <l,k||c,d>*t2(c,a,i,j)*t2(d,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,caij,dblk->abij', t2_dag, g[v,v,o,o], t2, optimize=['einsum_path', (0, 2), (0, 1)])

    return doubles_res


def offset_t2resid_T3(t3,g,o,v):
    #t1=t1.transpose(1,0)
    #t3=t3.transpose(3,4,5,0,1,2)
    # Building WnT1 into T2 residual

    ### Add WnT3 into T2 residual eqn:

    #         0.5000 P(i,j)<l,k||c,j>*t3(c,a,b,i,l,k)
    contracted_intermediate =  0.500000000000000 * einsum('lkcj,cabilk->abij', g[o, o, v, o], t3)
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #         0.5000 P(a,b)<k,a||c,d>*t3(c,d,b,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('kacd,cdbijk->abij', g[o, v, v, v], t3)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    return doubles_res

def kernel(t1, t2, t3,fock, g, o, v, e_ai, e_abij,e_t3, max_iter=100, stopping_eps=1.0E-8,
           diis_size=None, diis_start_cycle=4):

    # initialize diis if diis_size is not None
    # else normal scf iterate
    if diis_size is not None:
        from diis import DIIS
        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        t1_dim = t1.size
        t2_dim = t2.size
        old_vec = np.hstack((t1.flatten(), t2.flatten(),t3.flatten()))

    fock_e_ai = np.reciprocal(e_ai)
    fock_e_abij = np.reciprocal(e_abij)
    fock_e_t3=np.reciprocal(e_t3)
    old_energy = ccsd_energy(t1, t2, fock, g, o, v)
    for idx in range(max_iter):

        t2_dag=t2.transpose(2,3,0,1)

        singles_res = 0.0*singles_residual(t1, t2, fock, g, o, v) + 0.0*fock_e_ai * t1
        doubles_res = (doubles_residual(t1, t2, fock, g, o, v)-offset_t2resid_t2squared(t2,g,o,v) + offset_t2resid_t2dag_w_t2(t2,t2_dag,g,o,v)+offset_t2resid_T3(t3,g,o,v)) + fock_e_abij * t2

        triples_res=t3_resid(t3,t2,fock,g,o,v)*0.0+fock_e_t3*t3*0.0

        new_singles = singles_res * e_ai*0.0
        new_doubles = doubles_res * e_abij
        new_triples = triples_res * e_t3 * 0.0
        # diis update
        if diis_size is not None:
            vectorized_iterate = np.hstack(
                (new_singles.flatten(), new_doubles.flatten(),new_triples.flatten()))
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate,
                                                                 error_vec)
            new_singles = new_vectorized_iterate[:t1_dim].reshape(t1.shape)
            new_doubles = new_vectorized_iterate[t1_dim:t1_dim+t2_dim].reshape(t2.shape)
            new_triple=new_vectorized_iterate[t1_dim+t2_dim:].reshape(t3.shape)
            old_vec = new_vectorized_iterate
#        if diis_size is not None:
#            vectorized_iterate = np.hstack(
#                (new_singles.flatten(), new_doubles.flatten()))
#            error_vec = old_vec - vectorized_iterate
#            new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate,
#                                                                 error_vec)
#            new_singles = new_vectorized_iterate[:t1_dim].reshape(t1.shape)
#            new_doubles = new_vectorized_iterate[t1_dim:].reshape(t2.shape)
#            old_vec = new_vectorized_iterate

        current_energy = ccsd_energy(new_singles, new_doubles, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < stopping_eps:
            return new_singles, new_doubles
        else:
            t1 = new_singles*0.0
            t2 = new_doubles
            t3=new_triple*0.0
            old_energy = current_energy
            print("\tIteration {: 5d}\t{: 5.15f}\t{: 5.15f}".format(idx, old_energy, delta_e))
    else:
        print("Did not converge")
        return new_singles, new_doubles


def main():
    from itertools import product
    import pyscf
    import openfermion as of
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from openfermionpyscf import run_pyscf
    from pyscf.cc.addons import spatial2spin
    import numpy as np


    basis = 'STO-6G'
    mol = pyscf.M(
        atom='''
H  0.00000000    -0.75730014     0.52066641
O  0.00000000     0.00000000    -0.06561345
H  0.00000000     0.75730014     0.52066641''',
        basis=basis)

    mf = mol.RHF().run()
    mycc = mf.CCSD().run()
    print('CCSD correlation energy', mycc.e_corr)

    molecule = of.MolecularData(geometry=[['H', (0, -0.75730014,0.52066641)],['O',(0,0,-0.06561345)],['H',(0,0.75730014,0.52066641)]],
                                basis=basis, charge=0, multiplicity=1)
    molecule = run_pyscf(molecule, run_ccsd=True)
    oei, tei = molecule.get_integrals()
    norbs = int(mf.mo_coeff.shape[1])
    occ = mf.mo_occ
    nele = int(sum(occ))
    nocc = nele // 2
    assert np.allclose(np.transpose(mycc.t2, [1, 0, 3, 2]), mycc.t2)

    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)

    # put in physics notation. OpenFermion stores <12|2'1'>
    gtei = astei.transpose(0, 1, 3, 2)

    eps = np.kron(molecule.orbital_energies, np.ones(2))
    n = np.newaxis
    o = slice(None, 2 * nocc)
    v = slice(2 * nocc, None)

    e_t3= 1/(-eps[v,n,n,n,n,n]-eps[n,v,n,n,n,n]-eps[n,n,v,n,n,n]+eps[n,n,n,o,n,n]
            +eps[n,n,n,n,o,n]+eps[n,n,n,n,n,o])

    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    fock = soei + np.einsum('piiq->pq', astei[:, o, o, :])
    hf_energy = 0.5 * np.einsum('ii', (fock + soei)[o, o])
    hf_energy_test = 1.0 * einsum('ii', fock[o, o]) -0.5 * einsum('ijij', gtei[o, o, o, o])
    print(hf_energy_test, hf_energy)
    assert np.isclose(hf_energy + molecule.nuclear_repulsion, molecule.hf_energy)

    g = gtei
    nsvirt = 2 * (norbs - nocc)
    nsocc = 2 * nocc
    #t1f, t2f = kernel(np.zeros((nsvirt, nsocc)), np.zeros((nsvirt, nsvirt, nsocc, nsocc)), fock, g, o, v, e_ai, e_abij)
    #print(ccsd_energy(t1f, t2f, fock, g, o, v) - hf_energy)

    print('starting kernel')
    t3=np.zeros((nsvirt,nsvirt,nsvirt,nsocc,nsocc,nsocc))
    t1f, t2f = kernel(np.zeros((nsvirt, nsocc)), np.zeros((nsvirt, nsvirt, nsocc, nsocc)),t3, fock, g, o, v, e_ai, e_abij,e_t3,
                      diis_size=8, diis_start_cycle=4)
    print('ending kernel')

    import pdagq_T3corr as pdagq_T3corr

    l1=t1f.transpose(1,0)
    l2=t2f.transpose(2,3,0,1)
    t3resid=pdagq_T3corr.build_T3(g,o,v,t2f)
    t3=t3resid*e_t3 #self.denoms["D3aa"]
    trip_energy=pdagq_T3corr.pdagq_T3energy(g,o,v,l1,l2,t3)
    print('trip E:',trip_energy)


    print(ccsd_energy(t1f, t2f, fock, g, o, v) - hf_energy)




if __name__ == "__main__":
    main()
