import numpy as np
from numpy import einsum

def fourthOrder(o,v,g,l2,t2,t3):
    #        -0.2500 P(i,j)<n,m||k,l>*l2(k,l,d,c)*t2(d,c,j,m)*t2(a,b,i,n)
    contracted_intermediate = -0.250000000000000 * einsum('nmkl,kldc,dcjm,abin->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    double_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.5000 P(i,j)<n,m||k,l>*l2(k,l,d,c)*t2(d,a,j,m)*t2(c,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmkl,kldc,dajm,cbin->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.5000 P(i,j)<n,m||j,l>*l2(k,l,d,c)*t2(d,c,k,m)*t2(a,b,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmjl,kldc,dckm,abin->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.5000 P(i,j)<n,m||j,l>*l2(k,l,d,c)*t2(d,c,i,m)*t2(a,b,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmjl,kldc,dcim,abkn->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -1.0000 P(i,j)<n,m||j,l>*l2(k,l,d,c)*t2(d,a,k,m)*t2(c,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjl,kldc,dakm,cbin->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         1.0000 P(i,j)<n,m||j,l>*l2(k,l,d,c)*t2(d,a,i,m)*t2(c,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmjl,kldc,daim,cbkn->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.5000 <n,m||i,j>*l2(k,l,d,c)*t2(d,c,l,m)*t2(a,b,k,n)
    double_res += -0.500000000000000 * einsum('nmij,kldc,dclm,abkn->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #         0.5000 P(a,b)<n,m||i,j>*l2(k,l,d,c)*t2(d,a,l,m)*t2(c,b,k,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmij,kldc,dalm,cbkn->abij', g[o, o, o, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #         1.0000 P(i,j)<m,d||e,l>*l2(k,l,d,c)*t2(e,c,j,k)*t2(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdel,kldc,ecjk,abim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         1.0000 <m,d||e,l>*l2(k,l,d,c)*t2(e,c,i,j)*t2(a,b,k,m)
    double_res +=  1.000000000000000 * einsum('mdel,kldc,ecij,abkm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #        -1.0000 P(i,j)*P(a,b)<m,d||e,l>*l2(k,l,d,c)*t2(e,a,j,k)*t2(c,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,kldc,eajk,cbim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #        -1.0000 P(a,b)<m,d||e,l>*l2(k,l,d,c)*t2(e,a,i,j)*t2(c,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,kldc,eaij,cbkm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #         0.5000 P(i,j)<m,d||e,j>*l2(k,l,d,c)*t2(e,c,k,l)*t2(a,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mdej,kldc,eckl,abim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -1.0000 P(i,j)<m,d||e,j>*l2(k,l,d,c)*t2(e,c,i,l)*t2(a,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,kldc,ecil,abkm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.5000 P(i,j)*P(a,b)<m,d||e,j>*l2(k,l,d,c)*t2(e,a,k,l)*t2(c,b,i,m)
    contracted_intermediate = -0.500000000000000 * einsum('mdej,kldc,eakl,cbim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         1.0000 P(i,j)*P(a,b)<m,d||e,j>*l2(k,l,d,c)*t2(e,a,i,l)*t2(c,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdej,kldc,eail,cbkm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         1.0000 P(i,j)*P(a,b)<m,a||e,l>*l2(k,l,d,c)*t2(e,d,j,k)*t2(c,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,kldc,edjk,cbim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         1.0000 P(a,b)<m,a||e,l>*l2(k,l,d,c)*t2(e,d,i,j)*t2(c,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,kldc,edij,cbkm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #         0.5000 P(i,j)*P(a,b)<m,a||e,l>*l2(k,l,d,c)*t2(e,b,j,k)*t2(d,c,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mael,kldc,ebjk,dcim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         0.5000 P(a,b)<m,a||e,l>*l2(k,l,d,c)*t2(e,b,i,j)*t2(d,c,k,m)
    contracted_intermediate =  0.500000000000000 * einsum('mael,kldc,ebij,dckm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #         0.5000 P(i,j)*P(a,b)<m,a||e,j>*l2(k,l,d,c)*t2(e,d,k,l)*t2(c,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('maej,kldc,edkl,cbim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #        -1.0000 P(i,j)*P(a,b)<m,a||e,j>*l2(k,l,d,c)*t2(e,d,i,l)*t2(c,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('maej,kldc,edil,cbkm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         0.2500 P(i,j)*P(a,b)<m,a||e,j>*l2(k,l,d,c)*t2(e,b,k,l)*t2(d,c,i,m)
    contracted_intermediate =  0.250000000000000 * einsum('maej,kldc,ebkl,dcim->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #        -0.5000 P(i,j)*P(a,b)<m,a||e,j>*l2(k,l,d,c)*t2(e,b,i,l)*t2(d,c,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('maej,kldc,ebil,dckm->abij', g[o, v, v, o], l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #        -0.2500 <d,c||e,f>*l2(k,l,d,c)*t2(e,a,k,l)*t2(f,b,i,j)
    double_res += -0.250000000000000 * einsum('dcef,kldc,eakl,fbij->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #         0.5000 P(i,j)<d,c||e,f>*l2(k,l,d,c)*t2(e,a,j,l)*t2(f,b,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('dcef,kldc,eajl,fbik->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -0.2500 <d,c||e,f>*l2(k,l,d,c)*t2(e,a,i,j)*t2(f,b,k,l)
    double_res += -0.250000000000000 * einsum('dcef,kldc,eaij,fbkl->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #         0.5000 P(a,b)<d,a||e,f>*l2(k,l,d,c)*t2(e,c,k,l)*t2(f,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('daef,kldc,eckl,fbij->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #        -1.0000 P(i,j)*P(a,b)<d,a||e,f>*l2(k,l,d,c)*t2(e,c,j,l)*t2(f,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('daef,kldc,ecjl,fbik->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         0.5000 P(a,b)<d,a||e,f>*l2(k,l,d,c)*t2(e,c,i,j)*t2(f,b,k,l)
    contracted_intermediate =  0.500000000000000 * einsum('daef,kldc,ecij,fbkl->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #        -0.5000 <a,b||e,f>*l2(k,l,d,c)*t2(e,d,k,l)*t2(f,c,i,j)
    double_res += -0.500000000000000 * einsum('abef,kldc,edkl,fcij->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #         0.5000 P(i,j)<a,b||e,f>*l2(k,l,d,c)*t2(e,d,j,l)*t2(f,c,i,k)
    contracted_intermediate =  0.500000000000000 * einsum('abef,kldc,edjl,fcik->abij', g[v, v, v, v], l2, t2, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
   

#### T3 part
####
    #         0.5000 <m,d||k,l>*l2(k,l,d,c)*t3(c,a,b,i,j,m)
    double_res +=  0.500000000000000 * einsum('mdkl,kldc,cabijm->abij', g[o, v, o, o], l2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #        -1.0000 P(i,j)<m,d||j,l>*l2(k,l,d,c)*t3(c,a,b,i,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdjl,kldc,cabikm->abij', g[o, v, o, o], l2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #         0.5000 <m,d||i,j>*l2(k,l,d,c)*t3(c,a,b,k,l,m)
    double_res +=  0.500000000000000 * einsum('mdij,kldc,cabklm->abij', g[o, v, o, o], l2, t3, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #         0.2500 P(a,b)<m,a||k,l>*l2(k,l,d,c)*t3(d,c,b,i,j,m)
    contracted_intermediate =  0.250000000000000 * einsum('makl,kldc,dcbijm->abij', g[o, v, o, o], l2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #        -0.5000 P(i,j)*P(a,b)<m,a||j,l>*l2(k,l,d,c)*t3(d,c,b,i,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('majl,kldc,dcbikm->abij', g[o, v, o, o], l2, t3, optimize=['einsum_path', (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         0.2500 P(a,b)<m,a||i,j>*l2(k,l,d,c)*t3(d,c,b,k,l,m)
    contracted_intermediate =  0.250000000000000 * einsum('maij,kldc,dcbklm->abij', g[o, v, o, o], l2, t3, optimize=['einsum_path', (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #         0.5000 <d,c||e,l>*l2(k,l,d,c)*t3(e,a,b,i,j,k)
    double_res +=  0.500000000000000 * einsum('dcel,kldc,eabijk->abij', g[v, v, v, o], l2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #         0.2500 P(i,j)<d,c||e,j>*l2(k,l,d,c)*t3(e,a,b,i,k,l)
    contracted_intermediate =  0.250000000000000 * einsum('dcej,kldc,eabikl->abij', g[v, v, v, o], l2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    
    #        -1.0000 P(a,b)<d,a||e,l>*l2(k,l,d,c)*t3(e,c,b,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('dael,kldc,ecbijk->abij', g[v, v, v, o], l2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)
    
    #        -0.5000 P(i,j)*P(a,b)<d,a||e,j>*l2(k,l,d,c)*t3(e,c,b,i,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('daej,kldc,ecbikl->abij', g[v, v, v, o], l2, t3, optimize=['einsum_path', (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)
    
    #         0.5000 <a,b||e,l>*l2(k,l,d,c)*t3(e,d,c,i,j,k)
    double_res +=  0.500000000000000 * einsum('abel,kldc,edcijk->abij', g[v, v, v, o], l2, t3, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #         0.2500 P(i,j)<a,b||e,j>*l2(k,l,d,c)*t3(e,d,c,i,k,l)
    contracted_intermediate =  0.250000000000000 * einsum('abej,kldc,edcikl->abij', g[v, v, v, o], l2, t3, optimize=['einsum_path', (1, 2), (0, 1)])
    double_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)
    return double_res


 
