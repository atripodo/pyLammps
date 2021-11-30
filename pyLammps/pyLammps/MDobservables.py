import numpy as np
from scipy import spatial
from pyLammps.box_manipulations import *
import pyLammps.pars
from scipy.spatial import KDTree,ConvexHull
from scipy.interpolate import interp1d
import random
from freud import box,order,locality

# ----------------------------------------------------------------------------------------
# wr    coordinate wrapped
# npa   numero di particelle
# nc    numero di configurazioni


def local_density_correlation_function(wr, shifted_box):
    nc = wr.shape[0]
    npa = wr.shape[1]
    cf = []
    for conf in range(nc):
        b1 = wr[conf] < shifted_box[conf, :, 0]
        b2 = wr[conf] > shifted_box[conf, :, 1]
        assert not (b1.any() or b2.any()), "coordinates must be wrapped at boundaries"
        replica = replicate(wr[conf], shifted_box[conf])
        tree = spatial.KDTree(replica)
        n = tree.query_ball_point(replica[:npa], 2.4, return_length=True)
        if conf == 0:
            rho_0 = n
            norm = np.mean(n ** 2) - np.mean(n) ** 2
        v_temp = np.mean(rho_0 * n) - np.mean(rho_0) * np.mean(n)
        cf.append(v_temp / norm)
    return np.array(cf)


def first_and_second_shell_cordination_num(
    wr, shifted_box, conf, dist1=1.44, dist2=2.3
):
    nc = wr.shape[0]
    npa = wr.shape[1]

    b1 = wr[conf] < shifted_box[conf, :, 0]
    b2 = wr[conf] > shifted_box[conf, :, 1]
    assert not (b1.any() or b2.any()), "coordinates must be wrapped at boundaries"

    tree = KDTree(wr[conf], boxsize=shifted_box[conf, :, 1])

    n1 = []
    n2 = []
    for i in range(npa):
        l1 = tree.query_ball_point(wr[conf, i], dist1)
        l2 = tree.query_ball_point(wr[conf, i], dist2)
        n1 += [len(l1)]
        n2 += [len(set(l2) - set(l1))]
    return np.array(n1).reshape(-1, 1), np.array(n2).reshape(-1, 1)


def radial_distrib_func(wr, shifted_box, sample_size, dr=0.1):

    nc = wr.shape[0]
    npa = wr.shape[1]
    assert sample_size < nc, "sample size must be contained in the configurations set"
    sample = random.choices(np.arange(nc), k=sample_size)

    Lmax = np.min(shifted_box[:, :, 1])
    bins = np.arange(0, Lmax / 2, dr)

    gr_tot = []
    for conf in sample:
        rho = npa / (
            shifted_box[conf, 0, 1] * shifted_box[conf, 1, 1] * shifted_box[conf, 2, 1]
        )
        norm = 4 * rho * np.pi * dr * bins ** 2
        tree = KDTree(wr[conf], boxsize=shifted_box[conf, :, 1])
        neighb = tree.count_neighbors(tree, bins, cumulative=False) / npa
        gr_tot.append(neighb[1:] / norm[1:])
    gr_tot = np.array(gr_tot)
    return bins[1:], np.mean(gr_tot, axis=0)

def voro_analysis(wr,shiftbox,t):
    wr0=wr[t]
    npa=wr0.shape[0]
    
    box = freud.box.Box.from_box(shiftbox[t, :, 1] - shiftbox[t, :, 0])
    voro = freud.locality.Voronoi()
    cells = voro.compute((box, wr0)).polytopes

    
    volumes=voro.volumes
    surfaces=[ConvexHull(np.array(cells[jj])).area for jj in range(npa)]
    asphericity=np.power(surfaces,3)/(np.power(volumes,2)*36*np.pi)-1

    return np.array(volumes), np.array(surfaces), asphericity


def per_atom_steinhardt_OP(box, r, t, l, rmax):
    box = freud.box.Box.from_box(box[t, :, 1] - box[t, :, 0])
    ql = freud.order.Steinhardt(l)
    ql.compute((box, r[0]), {"r_max": rmax})
    return ql.particle_order

def displacement_analysis(r,t,compute_u2=True):
    nc = r.shape[0]
    npa = r.shape[1]

    dr=r-r[0]
    dr2=np.linalg.norm(dr,axis=-1)**2
    
    if compute_u2:
        u2=np.array([interp1d(t,dr2[:,i])(1) for i in range(npa)])
        return np.mean(dr2,axis=1),u2
    else:
        return np.mean(dr2,axis=1)


# Intermediate scattering function


def createK(box, kval):
    a = np.mean(box, axis=0)
    L = a[:, 1] - a[:, 0]
    dk = np.pi / L
    tol = (0.5 * dk).max()
    nbar = (np.ones(3) * kval / dk).astype(int)
    alln = []
    for nx in range(-nbar[0], nbar[0]):
        for ny in range(-nbar[1], nbar[1]):
            for nz in range(-nbar[2], nbar[2]):
                alln.append((nx, ny, nz))
    alln = np.array(alln)
    allk = alln * dk
    mod = np.linalg.norm(allk, axis=1)
    allk = allk[np.argwhere((mod > (kval - tol)) & (mod < kval + tol))].reshape(-1, 3)
    return np.array(allk)


def self_intermediate_scattering_function(r, box, kval):
    nc = r.shape[0]
    npa = r.shape[1]
    isf = []
    allk = createK(box, kval)
    for conf in range(1, nc):
        displ = r[conf] - r[0]
        isf.append(np.mean(np.cos(np.dot(allk, np.transpose(displ)))))
    return np.array(isf)
