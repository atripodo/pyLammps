import pandas as pd
import os
import glob

dt=None
window_list=None
nruns=None
prefix=None
postfix=None


def read_pars():
    global dt
    global window_list
    global nruns
    global prefix
    global postfix
    if not(os.path.isfile('setup_pars.txt')):
        f = open("setup_pars.txt", "a")
        f.write("timestep,windows,runs_number,run_prefix,run_postfix")
        f.write("\n\n\n #########################################################\n")
        f.write('il path per il file sarà creato come window/run_prefix+$(numero della run)+/run_postfix\n')
        f.write('quindi nel caso della prima finestra in questione sarà T038vshort/run1/lmp_data\n')
        f.write('il programma cercherà in automatico i file "Conf*"')
        f.close()
        print("compile the setup_file")
    else:
        with open('setup_pars.txt',"r") as dtf:
            lines=[dtf.readline() for i in range(2)]
        assert lines[1]!='\n',"setup file not compiled"
        df=pd.read_csv('setup_pars.txt').fillna('')
        dt=float(df.at[0,'timestep'])
        window_list=(df.at[0,'windows']).split()
        nruns=int(df.at[0,'runs_number'])
        prefix=df.at[0,'run_prefix']
        postfix=df.at[0,'run_postfix']
    return

def print_pars():
    global dt
    global window_list
    global nruns
    global prefix
    global postfix
    print('dt=',dt)
    print('temporal windows=',window_list)
    print('number of runs=',nruns)
    print('folder prefix=',prefix)
    print('folder postfix=',postfix)
    return