import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms


pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

pq.set_left_operators([['e4(i,j,k,l,d,c,b,a)']])

print('')
print('    < 0 | i* j* b a e(-T) H e(T) | 0> :')
print('')

# Add T3 contribs to CCSDT(Qf)
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

# E(Qf)
pq.set_left_operators([['l2']])

print('')
print('    E(t)')
print('')

pq.add_commutator(1.0,['v'],['t4'])
pq.simplify()

e_t_terms = pq.fully_contracted_strings()
for my_term in e_t_terms:
    print(my_term)


e_t_terms = contracted_strings_to_tensor_terms(e_t_terms)
for my_term in e_t_terms:
    print("#\t", my_term)
    print(my_term.einsum_string(update_val='energy'))
    print()

pq.clear()

