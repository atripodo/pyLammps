import numpy as np
import pyvoro
from scipy.spatial import ConvexHull


def polygon_area(poly):
    # shape (N, 3)
    if isinstance(poly, list):
        poly = np.array(poly)
    # all edges
    edges = poly[1:] - poly[0:1]
    # row wise cross product
    cross_product = np.cross(edges[:-1], edges[1:], axis=1)
    # area of all triangles
    area = np.linalg.norm(cross_product, axis=1) / 2
    return sum(area)


def voro_analysis(wr0, shiftbox0):
    assert (
        len(wr0.shape) == 2
    ), "function voro_analysis needs a sigle snapshot wrapped at boundaries"
    npa = wr0.shape[0]
    voro_tessellation = pyvoro.compute_voronoi(
        wr0, shiftbox0, 1.26, periodic=[True] * 3
    )
    volumes = [voro_tes[jj]["volume"] for jj in range(npa)]
    surfaces = [
        ConvexHull(np.array(voro_tessellation[jj]["vertices"])).area
        for jj in range(npa)
    ]
    asphericity = np.power(surfaces, 3) / (np.power(volumes, 2) * 36 * np.pi) - 1
    return np.array(volumes), np.array(surfaces), asphericity
