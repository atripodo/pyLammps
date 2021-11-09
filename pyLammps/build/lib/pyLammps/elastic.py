import numpy as np
from scipy import spatial
from pyLammps import box_manipulations
import pyLammps.pars
from scipy.spatial import KDTree

def check_if_bonded(a,b,nchain=5):
    c=np.mod(a,nchain)
    d=np.mod(b,nchain)
    return (np.abs(a-b)==1)and(np.abs(c-d)!=4)

def is_in(vec,nbox,W): #nbox is a tuple (nx,ny,nz)
    return np.all(np.floor_divide(vec,W)==np.array(nbox),axis=1)

def where_is(vec,boxes,W): #nbox is a tuple (nx,ny,nz)
    return int(np.argwhere(np.all(np.floor_divide(vec,W)==boxes,axis=1)).flatten())

def periodic_distance(r1,r2,L):
    return np.remainder(r1 - r2 + L/2., L) - L/2.

def pair_pot(a,b,rr,prefactor1,prefactor2,boxes,W,L,nchain=5): # qui r è un singolo snapshot

    rab_v=periodic_distance(rr[b],rr[a],L) 
    rab_m=np.linalg.norm(rab_v)
    

    q_ab=weight_q_ab(rr[a],rab_v,W)
    
    if check_if_bonded(a,b,nchain):
        if rab_m>2.5:
            print('\t\t',a,b, 'bonded but at distance',rab_m )
        return prefactor2(rab_m)*np.einsum('i,j,k,l->ijkl',rab_v,rab_v,rab_v,rab_v)*q_ab/(rab_m**2)
    else:
        return prefactor1(rab_m)*np.einsum('i,j,k,l->ijkl',rab_v,rab_v,rab_v,rab_v)*q_ab/(rab_m**2)
    
def prefactor1(rdist,rcut=2.5):#non bondati
    if rdist<rcut:
        return 96*(7-2*rdist**6)/rdist**14
    else:
        return 0
    
def prefactor2(rdist,k=555.5,r0=0.97,rcut=2.5): #bondati
    if rdist<rcut:
        return 96*(7-2*rdist**6)/rdist**14+2*k*r0/rdist
    else:
        raise SystemExit("bonded atom at impossible distance!")
        return 


def weight_q_ab(v_a,v_d,W):# posizione , vettore differenza, dimensione box piccolo
    vec_a=v_a.flatten()
    diff_vec=v_d.flatten()
    
    if all(np.floor_divide(vec_a,W)==np.floor_divide(vec_a+diff_vec,W)):
        return 1.0
    else:  
        dist_a=np.min(np.stack((np.remainder(vec_a,W),W*np.ones(vec_a.shape)-np.remainder(vec_a,W))),axis=0)
        j_sel=np.argmin(np.abs(dist_a/diff_vec))

        x_1=np.abs(dist_a[j_sel])
        x_2=np.abs(diff_vec[j_sel])-x_1

        C=x_1/x_2
        return C/(1+C)
    
def compute_local_elastic_modulus(r, stress,box,n_div,T):
    wr,shiftbox = box_manipulations.wrap_at_boundary(r,box) #wrap at boundary
    
    nc=wr.shape[0] # numero di configurazioni della traiettori
    npa=wr.shape[1] #numero particelle
    
    
    L=shiftbox[0,:,1] #vettori scatola

    V=L[0]*L[1]*L[2] #volume scatola
    
    W=shiftbox[0,1,1]/n_div #dimensione cubetto locale

    boxes=np.array([[[[i,j,k] for k in range(n_div)]for j in range(n_div)]for i in range(n_div)]).reshape(-1,3)
    #creo i cubetti locali
    
    id_sel=[[np.argwhere(is_in(wr[t],m_box,W)).flatten()\
             for m_box in boxes]for t in range(nc)] #faccio la selezione delle particelle per ogni cubetto ad ogni tempo

    time_forest=[KDTree(wr[t].reshape(-1,3),boxsize=L) for t in range(nc)] #ad ogni tempo calcolo l'albero delle distanze 

    pairs=[tree.query_pairs(2.5,output_type='ndarray') for tree in time_forest] #le coppie

    pair_per_box_list=[[[pairs[t][np.argwhere(pairs[t][:,0]==i)].reshape(-1,2) \
                         for i in id_sel[t][m]]for m in range(n_div**3)] for t in range(nc)]
    #le coppie per box
    
    
    
    C_B_m_t=np.array([[np.sum(np.array([pair_pot(a,b,wr[t],prefactor1,prefactor2,boxes,W,L)/W**3 \
                              for a,b in np.concatenate(pair_per_box_list[t][i][:])]),axis=0) \
             for i in range(n_div**3)]for t in range(nc)]) #calcolo componente non-affine (di Born)
    
    #creo il tensore degli stress
    
    sigma=np.zeros((nc,npa,3,3))
    sigma[:,:,0,0]=stress[:,:,0]
    sigma[:,:,1,1]=stress[:,:,1]
    sigma[:,:,2,2]=stress[:,:,2]
    sigma[:,:,0,1]=stress[:,:,3]
    sigma[:,:,1,0]=stress[:,:,3]
    sigma[:,:,0,2]=stress[:,:,4]
    sigma[:,:,2,0]=stress[:,:,4]
    sigma[:,:,1,2]=stress[:,:,5]
    sigma[:,:,2,1]=stress[:,:,5]
    
    #tensore stress blocchetto m
    sigma_m=np.array([[np.sum(sigma[t,np.where(is_in(wr[t],m_box,W)),:,:].reshape(-1,3,3),axis=0)/W**3 \
                       for t in range(nc)]for m_box in boxes])
    #tensore stress intera scatola
    sigma_tot=np.sum(sigma_m,axis=0)*W**3/(V) 

    C_N_m=(V/T)*(np.einsum('mtij,tkl->mijkl',sigma_m,sigma_tot)/nc- \
             np.einsum('mij,kl->mijkl',np.mean(sigma_m,axis=1),np.mean(sigma_tot,axis=0)))
    
    C=np.mean(C_B_m_t,axis=0)-C_N_m
    
    #moduli di taglio
    G3=C[:,0,1,0,1]/2
    G4=C[:,0,2,0,2]/2
    G5=C[:,1,2,1,2]/2
    shear=(G3+G4+G3)/3

    print(shear.shape)
    
    pos0=[where_is(wr[0,i],boxes,W) for i in range(wr.shape[1])]
    local_em=[shear[i] for i in pos0]
    

    return C , local_em