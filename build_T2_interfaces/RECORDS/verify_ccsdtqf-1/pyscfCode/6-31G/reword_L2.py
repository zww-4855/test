complete_draft=[]
for line in open('qf_energy.py','r'):
    seen=[]
    words=line.rstrip('\n').split()
    fixed_line=[]
    for word in words:
        if word=='l2,' and not word in seen:
            print(word)
            seen.append(word)
            fixed_line.append('t2_dagger,')
        else:
            fixed_line.append(word)

    print('fixed line:', fixed_line)
    complete_draft.append(fixed_line)
    print('seen',seen)



with open("test.txt","w") as file:
    for line in complete_draft:
        file.writelines(" ".join(line))
        file.write('\n')


