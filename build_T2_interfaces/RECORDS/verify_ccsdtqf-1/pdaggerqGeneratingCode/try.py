import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms


pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

pq.set_left_operators([['e4(i,j,k,l,d,c,b,a)']])

print('')
print('    < 0 | i* j* b a e(-T) H e(T) | 0> :')
print('')

# Add T3 contribs to CCSDT(Qf)
# Currently, code constructs CCSDTQ-1 correction to doubles residual equations
# To construct CCSDTQ-2 corrections, replace second line with triple commutator of
# T2
pq.add_st_operator(1.0,['f'],['t4'])
pq.add_double_commutator(0.5,['v'],['t2'],['t2'])
pq.add_commutator(1.0,['v'],['t3'])
pq.simplify()

# grab list of fully-contracted strings, then print
doubles_residual_terms = pq.fully_contracted_strings()
for my_term in doubles_residual_terms:
    print(my_term)

doubles_residual_terms = contracted_strings_to_tensor_terms(doubles_residual_terms)
for my_term in doubles_residual_terms:
    print("#\t", my_term)
    print(my_term.einsum_string(update_val='quad_res',
                                output_variables=('a', 'b','c','d', 'i', 'j','k','l')))
    print()

pq.clear()


# Augmented T2 equations with effects of T4 via Wn(T2^2/2 +T3)
# E(Qf)
pq.set_left_operators([['e2(i,j,b,a)']])
#pq.set_left_operators([['l2']])
print('')
print('    < 0 | i* j* b a e(-T) H e(T) | 0> :')
print('')

print('AUGMENTED T2 RESIDUAL EQUATIONS, WITH WN(T2^2) CONTRIBUTION')


pq.add_st_operator(1.0,['f'],['t2'])
pq.add_st_operator(1.0,['v'],['t2'])
pq.add_commutator(1.0,['v'],['t4'])
pq.simplify()

doubles_residual_terms = pq.fully_contracted_strings()
for my_term in doubles_residual_terms:
    print(my_term)

doubles_residual_terms = contracted_strings_to_tensor_terms(doubles_residual_terms)
for my_term in doubles_residual_terms:
    print("#\t", my_term)
    print(my_term.einsum_string(update_val='double_res',
                                output_variables=('a', 'b', 'i', 'j')))
    print()

pq.clear()


