import pandas as pd
import numpy as np
import os
import pyLammps.pars as pars
import glob
import re

# ----------------------------------------------------------------------------------------


def timestep(string):
    return list(map(int, re.findall(r"\d+", string)))[-1]


def gather_config_trajectory(nrun):
    pars.read_pars()
    pars.print_pars()
    all_f = []
    end = -0.1
    for window in pars.window_list:
        dirpath = "./" + window + "/" + pars.prefix + str(nrun) + "/" + pars.postfix
        print(dirpath)
        files = glob.glob(dirpath + "/Conf*")
        files.sort(key=timestep)
        add = [f for f in files if timestep(f) > end]
        if add:
            all_f += add
            end = timestep(add[-1])
    all_f.sort(key=timestep)
    return all_f


def load_data(files):
    nc = len(files)
    trjct_data = []
    time_data = []
    box_data = []
    for f in files:
        with open(f, "r") as dtf:
            lines = [dtf.readline() for i in range(9)]
            t = int(lines[1])
            lx = list(map(float, lines[5].split(" ")))
            ly = list(map(float, lines[6].split(" ")))
            lz = list(map(float, lines[7].split(" ")))
        box_data.append(np.array([lx, ly, lz]))
        trjct_data.append(np.loadtxt(f, skiprows=9))
        time_data.append(t)
    trjct_data = np.array(trjct_data)
    time_data = np.array(time_data) * pars.dt
    box = np.array(box_data)
    r = trjct_data[:, :, 2:5]  # conf #indice_part #componenti
    v = trjct_data[:, :, 5:]
    return time_data, r, v, box


def stress_load_data(files):
    nc = len(files)
    trjct_data = []
    time_data = []
    box_data = []
    for f in files:
        with open(f, "r") as dtf:
            lines = [dtf.readline() for i in range(9)]
            t = int(lines[1])
            lx = list(map(float, lines[5].split(" ")))
            ly = list(map(float, lines[6].split(" ")))
            lz = list(map(float, lines[7].split(" ")))
        box_data.append(np.array([lx, ly, lz]))
        trjct_data.append(np.loadtxt(f, skiprows=9))
        time_data.append(t)
    trjct_data = np.array(trjct_data)
    time_data = np.array(time_data) * pars.dt
    box = np.array(box_data)
    r = trjct_data[:, :, 2:5]  # conf #indice_part #componenti
    stress = trjct_data[:, :, 5:]
    return time_data, r, stress, box
