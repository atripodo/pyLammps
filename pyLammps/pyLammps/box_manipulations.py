import numpy as np
import pyLammps.pars as pars

# ----------------------------------------------------------------------------------------


def replicate(r, box):
    assert (
        len(r.shape) == 2
    ), "replicate acts only on single snapshot. Specify the snapshot"
    r0 = np.copy(r)
    L = box[:, 1]

    rplus = np.copy(r[(r[:, 0] < L[0] / 2)])
    rmin = np.copy(r[(r[:, 0] > L[0] / 2)])
    rplus[:, 0] += L[0]
    rmin[:, 0] -= L[0]
    stecca = np.concatenate((r0, rplus, rmin), axis=0)

    steccaplus = np.copy(stecca[(stecca[:, 1] < L[1] / 2)])
    steccamin = np.copy(stecca[(stecca[:, 1] > L[1] / 2)])
    steccaplus[:, 1] += L[1]
    steccamin[:, 1] -= L[1]
    piano = np.concatenate((stecca, steccaplus, steccamin), axis=0)

    pianoplus = np.copy(piano[(piano[:, 2] < L[2] / 2)])
    pianomin = np.copy(piano[(piano[:, 2] > L[2] / 2)])
    pianoplus[:, 2] += L[2]
    pianomin[:, 2] -= L[2]
    vol = np.concatenate((piano, pianoplus, pianomin), axis=0)

    return vol


def wrap_at_boundary(r, box):
    o_shape = r.shape
    nc = o_shape[0]
    npa = o_shape[1]
    shift = np.copy(box)
    shift[:, :, 1] = box[:, :, 0]
    shifted_box = box - shift
    shifted_r = r - np.swapaxes(
        np.repeat(shift[:, :, -1, np.newaxis], npa, axis=2), 1, 2
    )
    for conf in range(nc):
        L = shifted_box[0, :, 1]
        b1 = shifted_r[conf] < shifted_box[conf, :, 0]
        b2 = shifted_r[conf] > shifted_box[conf, :, 1]
        while b1.any() or b2.any():
            shifted_r[conf] += b1 * L - b2 * L
            b1 = shifted_r[conf] < shifted_box[conf, :, 0]
            b2 = shifted_r[conf] > shifted_box[conf, :, 1]
    return shifted_r, shifted_box
