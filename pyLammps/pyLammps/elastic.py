import numpy as np
from scipy import spatial
from pyLammps import box_manipulations
import pyLammps.pars
from scipy.spatial import KDTree


def check_if_bonded(a, b, nchain=5):
    c = np.mod(a, nchain)
    d = np.mod(b, nchain)
    return (np.abs(a - b) == 1) and (np.abs(c - d) != 4)


def is_in(vec, nbox, W):  # nbox is a tuple (nx,ny,nz)
    return np.all(np.floor_divide(vec, W) == np.array(nbox), axis=1)


def where_is(vec, boxes, W):  # nbox is a tuple (nx,ny,nz)
    return int(np.argwhere(np.all(np.floor_divide(vec, W) == boxes, axis=1)).flatten())


def periodic_distance(r1, r2, L):
    return np.remainder(r1 - r2 + L / 2.0, L) - L / 2.0


def born_term_addend(
    x, rr, boxes, W, plength, k=555.5, r0=0.97, nchain=5
):  # qui r è un singolo snapshot
    a, b = x
    rab_v = periodic_distance(rr[b], rr[a], plength)
    rab_m = np.linalg.norm(rab_v)

    if rab_m > 2.5:
        print(a, b)

    q_ab = weight_q_ab(rr[a], rab_v, W)

    if check_if_bonded(a, b, nchain):
        return (
            ((96 * (7 - 2 * rab_m ** 6) / rab_m ** 14) + (2 * k * r0 / rab_m))
            * np.einsum("i,j,k,l->ijkl", rab_v, rab_v, rab_v, rab_v)
            * q_ab
            / (rab_m ** 2)
        )
    else:
        return (
            (96 * (7 - 2 * rab_m ** 6) / rab_m ** 14)
            * np.einsum("i,j,k,l->ijkl", rab_v, rab_v, rab_v, rab_v)
            * q_ab
            / (rab_m ** 2)
        )


def weight_q_ab(v_a, v_d, W):  # posizione , vettore differenza, dimensione box piccolo
    vec_a = v_a.flatten()
    diff_vec = v_d.flatten()

    if all(np.floor_divide(vec_a, W) == np.floor_divide(vec_a + diff_vec, W)):
        return 0.5
    else:
        sign = np.sign(diff_vec)
        prod = np.zeros((2, 3))
        prod[0, sign == -1] = 1
        prod[1, sign >= 0] = 1
        dist_a = np.stack(
            (np.remainder(vec_a, W), W * np.ones(vec_a.shape) - np.remainder(vec_a, W))
        )[np.where(prod)]
        out = np.nan_to_num(np.abs(dist_a / diff_vec), nan=np.inf)

        j_sel = np.argmin(out)

        x_1 = np.abs(dist_a[j_sel])
        x_2 = np.abs(diff_vec[j_sel]) - x_1

        C = x_1 / x_2
        return C / (1 + C)


def compute_local_elastic_modulus(r, stress, box, n_div, T):
    wr, shiftbox = box_manipulations.wrap_at_boundary(r, box)  # wrap at boundary

    nc = wr.shape[0]  # numero di configurazioni della traiettori
    npa = wr.shape[1]  # numero particelle

    L = shiftbox[0, :, 1]  # vettori scatola

    V = L[0] * L[1] * L[2]  # volume scatola

    W = shiftbox[0, 1, 1] / n_div  # dimensione cubetto locale

    boxes = np.array(
        [
            [[[i, j, k] for k in range(n_div)] for j in range(n_div)]
            for i in range(n_div)
        ]
    ).reshape(-1, 3)
    # creo i cubetti locali

    id_sel = [
        [np.argwhere(is_in(wr[t], m_box, W)).flatten() for m_box in boxes]
        for t in range(nc)
    ]  # faccio la selezione delle particelle per ogni cubetto ad ogni tempo

    time_forest = [
        KDTree(wr[t].reshape(-1, 3), boxsize=L) for t in range(nc)
    ]  # ad ogni tempo calcolo l'albero delle distanze

    # le coppie per box
    pairs_per_box = np.array(
        [
            [
                np.concatenate(
                    [
                        [
                            [i, x]
                            for x in time_forest[t].query_ball_point(wr[t, i], 2.5)
                            if x != i
                        ]
                        for i in id_sel[t][m]
                    ]
                )
                for t in range(nc)
            ]
            for m in range(len(boxes))
        ],
        dtype=object,
    )

    C_B_m = np.array(
        [
            np.mean(
                list(
                    map(
                        lambda j: (1 / W ** 3)
                        * np.sum(
                            np.apply_along_axis(
                                born_term_addend,
                                1,
                                pairs_per_box[m][j],
                                rr=wr[j],
                                boxes=boxes,
                                W=W,
                                plength=L,
                            ),
                            axis=0,
                        ),
                        range(nc),
                    )
                ),
                axis=0,
            )
            for m in range(n_div ** 3)
        ]
    )  # calcolo componente non-affine (di Born)

    # creo il tensore degli stress

    sigma = np.zeros((nc, npa, 3, 3))
    sigma[:, :, 0, 0] = stress[:, :, 0]
    sigma[:, :, 1, 1] = stress[:, :, 1]
    sigma[:, :, 2, 2] = stress[:, :, 2]
    sigma[:, :, 0, 1] = stress[:, :, 3]
    sigma[:, :, 1, 0] = stress[:, :, 3]
    sigma[:, :, 0, 2] = stress[:, :, 4]
    sigma[:, :, 2, 0] = stress[:, :, 4]
    sigma[:, :, 1, 2] = stress[:, :, 5]
    sigma[:, :, 2, 1] = stress[:, :, 5]

    # tensore stress blocchetto m
    sigma_m = np.array(
        [
            [
                np.sum(
                    sigma[t, np.where(is_in(wr[t], m_box, W)), :, :].reshape(-1, 3, 3),
                    axis=0,
                )
                / W ** 3
                for t in range(nc)
            ]
            for m_box in boxes
        ]
    )
    # tensore stress intera scatola
    sigma_tot = np.mean(sigma_m, axis=0)

    C_N_m = (V / T) * (
        np.einsum("mtij,tkl->mijkl", sigma_m, sigma_tot) / nc
        - np.einsum(
            "mij,kl->mijkl", np.mean(sigma_m, axis=1), np.mean(sigma_tot, axis=0)
        )
    )

    C = C_B_m - C_N_m

    # moduli di taglio
    G3 = np.copy(C[:, 0, 1, 0, 1]) / 2
    G4 = np.copy(C[:, 0, 2, 0, 2]) / 2
    G5 = np.copy(C[:, 1, 2, 1, 2]) / 2
    shear = (G3 + G4 + G3) / 3

    pos0 = [where_is(wr[0, i], boxes, W) for i in range(npa)]
    local_em = [shear[i] for i in pos0]

    return C, local_em
