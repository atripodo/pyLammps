#!/usr/bin/python3

import re
from glob import glob
import numpy as np
import sympy as smp
from scipy import spatial
from sympy.physics.vector import ReferenceFrame
from sympy.physics.vector import gradient
from timeit import default_timer as timer
from multiprocessing import Pool

def timestep(string):
	return list(map(int,re.findall(r'\d+',string)))[-1]


def initialize_txt(file):
    X = np.loadtxt(file,skiprows=9)
    Lxmin = 1*np.loadtxt(file,usecols=(0),skiprows=5,max_rows=1)
    LXmax =  1*np.loadtxt(file,usecols=(1),skiprows=5,max_rows=1)
    LYmin =  1*np.loadtxt(file,usecols=(0),skiprows=6,max_rows=1)
    LYmax =  1*np.loadtxt(file,usecols=(1),skiprows=6,max_rows=1)
    LZmin =  1*np.loadtxt(file,usecols=(0),skiprows=7,max_rows=1)
    LZmax =  1*np.loadtxt(file,usecols=(1),skiprows=7,max_rows=1)

    step = 1*np.loadtxt(file,usecols=(0),skiprows=1,max_rows=1)
    box = np.array([step,LXmin,LXmax,LYmin,LYmax,LZmin,LZmax])

    return X, box


def initialize_bin(file):
    step, Npar, tricl,boundary, box,Nfield,Nproc = zip(*np.fromfile(file,dtype=np.dtype('i8,i8,i4,6i4,6f8,i4,i4'), count =1))
    step=step[0]
    Npar = Npar[0]
    boundary = boundary[0]
    box = box[0]
    Nfield = Nfield[0]
    Nproc = Nproc[0]
    X = np.fromfile(file,dtype=np.dtype(f'i4,{int(Nfield*Npar/Nproc)}f8'), count = Nproc,offset=100)
    X = np.array([X[i][1].reshape(int(Npar/Nproc),Nfield) for i in range(Nproc)])

    X = X.reshape(X.shape[0]*X.shape[1],X.shape[2] )

    return X,box,boundary,step

def read_lammps(filename,n_proc,binary=True):
    path = glob(filename)
    
    if binary ==False:     
        with Pool(n_proc) as pool:
            X,box = zip(*pool.map(initialize_txt,path) )
        
        box = np.array(box)
        t = np.argsort(box[:,0])
        box = box[t]
        X = np.array(X)[t]
        return X,box
    
    else:
        with Pool(n_proc) as pool:
            X,box,boundary,step = zip(*pool.map(initialize_bin,path) )
        t = np.argsort(np.array(step))
        X = np.array(X)[t]
        box = np.array(box)[t]
        boundary = boundary[0]
        
        return X,box,boundary
        
  
#classe particelle
class Particles:
    def __init__(self,path,n_proc) :
        self._X, self._box, self._boundary = read_lammps(path,n_proc)

    @property
    def X(self):
        return self._X
    @property
    def box(self):
        return self._box
    @property
    def boundary(self):
        return self._boundary

    def wrap(self):
        if (self.boundary[0]==0):
            Lx = (self.box[:,1] -self.box[:,0])[:,None]
            self._X[:,:,0] = self._X[:,:,0] - Lx*np.around(self._X[:,:,0]/Lx) + Lx/2   
        if (self.boundary[2]==0):
            Ly = (self.box[:,3] -self.box[:,2])[:,None]
            self._X[:,:,1] = self._X[:,:,1] - Ly*np.around(self._X[:,:,1]/Ly) + Ly/2   
        if (self.boundary[4]==0):
            Lz = (self.box[:,5] -self.box[:,4])[:,None]
            self._X[:,:,2] = self._X[:,:,2] - Lz*np.around(self._X[:,:,2]/Lz) + Lz/2 
    
 
