import numpy as np
import pyLammps.readLammps as reader
import pyLammps.pars as pars

trj_files=reader.gather_config_trajectory(1)#è la lista dei file della run "1"
                                            #se non è stato compilato il file setup verrà creato e andrà compilato
t,r,v,box=reader.load_data(trj_files) # t istanti di tempo alle quali sono stati presi gli snapshot
                                # r array posizione di ogni particella per ogni istante
                                # v array velocità di ogni particella per ogni istante
                                # posizione bordi del box di simulazione per ogni istante
