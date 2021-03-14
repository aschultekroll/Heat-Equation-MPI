import seaborn as sb
from time import time
from mpi4py import MPI
import numpy as np
import os
import pandas as pd
from heatmap import heatmap
from stepsolver import stepsolver
from stepsolver2 import stepsolver2
import math


def init(n,initial,boundary):
    #create grid with size n*n 
    grid = np.empty((n, n))
    #fill with initial condition
    grid.fill(initial)

    #fill grid with boundary condition
    for i in range(n):
        grid[i][0] = boundary
        grid[i][n - 1] = -1* boundary
        grid[0][i] = boundary
        grid[n-1][i] = -1 * boundary
    
    return grid

def create_directory(N, iterations, ghostcells,im_dir,s_dir,cwd):
    
    if not os.path.exists(s_dir):
        #create data directory
        os.makedirs(s_dir)
    if not os.path.exists(im_dir):
        #create image directory
        os.makedirs(im_dir) 
    return im_dir, s_dir


def heat_parallel(s_dir,cwd,sqrt, sqrt_proc,initial,boundary,rank, iterations, number_processes, gridpart_length, ghostcells, N, rest, comm, stencil, output):
    # array for time measurements
    time_eval = []

    # initialize grid
    grid = None
    im_dir = str(cwd + "/visualizations/{}_{}_{}/".format(N, ghostcells, iterations))
    
    
    

    if rank == 0:
        # call init function to initialize a grid od N*N
        grid = init(N,initial,boundary)
        
        create_directory(N, iterations, ghostcells,im_dir,s_dir,cwd)
        
        # start global time measurement
        start_time_global = time()

    #solve all timesteps 
    if sqrt:
        stepsolver2(sqrt_proc,time_eval,grid,rank, iterations, number_processes, gridpart_length, ghostcells, N, rest, comm, stencil,output,im_dir)
     
    else: 
        stepsolver(im_dir, time_eval,grid,rank, iterations, number_processes, gridpart_length, ghostcells, N, rest, comm, stencil,output)
    
    if rank == 0:
        #end global time measurement
        end_time_global = time()
        time2 = end_time_global - start_time_global
       
        return [N, number_processes,ghostcells, iterations,sqrt, time2, sum(time_eval),output]
        




# initialize stencil kernel
def def_stencil(alpha, dt, h):
    stencil = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])
    #calculate u_k+1
    stencil = alpha * dt/(h * h)  * stencil 
    return stencil

def time_measurement(sqrt, sqrt_proc,initial,boundary,rank, iterations, number_processes, gridpart_length, ghostcells, N, rest, comm, stencil, output,alpha):
    #time dataframe
    cwd = os.getcwd()
    s_dir = str(cwd +"/statistics/")
    
    time_df = pd.DataFrame(columns=["N","Processes","Geisterzellenbreite","Stepsolver2","Iterations","Overall Time", "Time","Output"])
    #time measurement file (set range higher to get more evaluations)
    for i in range(0, 1):
        if rank == 0:
            time_dfs = heat_parallel(s_dir,cwd,sqrt, sqrt_proc,initial,boundary,rank, iterations, number_processes, gridpart_length, ghostcells, N, rest, comm, stencil, output)
            #MPI barrier
            comm.Barrier()
            time_df.loc[i] = time_dfs
        else:
            heat_parallel(s_dir,cwd,sqrt, sqrt_proc,initial,boundary,rank,iterations, number_processes, gridpart_length, ghostcells, N, rest, comm, stencil, output)
            comm.Barrier()

    if rank == 0:
        time_df.to_csv('time5.csv', mode='a', header=False)
        time_df.to_csv(
            path_or_buf=s_dir + "time_measures_{}_{}_{}_{}.csv".format(N, alpha, ghostcells,
                                                                                    iterations))

