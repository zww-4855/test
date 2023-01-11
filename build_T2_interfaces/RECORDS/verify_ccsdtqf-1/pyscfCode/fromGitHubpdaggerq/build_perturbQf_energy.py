import pdaggerq 
from pdaggerq.parser import contracted_strings_to_tensor_terms

pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)
pq.set_left_operators([['l2', 'l2']])
pq.add_double_commutator(0.5, ['v'], ['t2'], ['t2'])
pq.add_commutator(1.0,['v'],['t3'])
pq.simplify()

# grab list of fully-contracted strings, then print
e_t_terms = pq.fully_contracted_strings()
energy_terms = contracted_strings_to_tensor_terms(e_t_terms)
for my_term in energy_terms:
    print("#\t",my_term)
    print("%s" % (my_term.einsum_string(update_val='energy')))
    print()

pq.clear()

