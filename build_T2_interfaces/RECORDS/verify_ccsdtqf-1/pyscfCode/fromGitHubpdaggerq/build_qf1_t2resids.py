import pdaggerq 
from pdaggerq.parser import contracted_strings_to_tensor_terms
# Build T2 residual eqns from T2^t(WT2^2/2)c
pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)
pq.set_left_operators([['e2(i,j,b,a)', 'l2']])
pq.add_double_commutator(0.5, ['v'], ['t2'], ['t2'])
pq.add_commutator(1.0,['v'],['t3'])
pq.simplify()

# grab list of fully-contracted strings, then print
e_t_terms = pq.fully_contracted_strings()
for my_term in e_t_terms:
    print(my_term)

e_t_terms = contracted_strings_to_tensor_terms(e_t_terms)
for my_term in e_t_terms:
    print("#\t", my_term)
    print(my_term.einsum_string(update_val='double_res',
                                output_variables=('a','b','i','j')))
    print()

pq.clear()

