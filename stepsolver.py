import numpy as np
from time import time
from scipy import signal
from mpi4py import MPI
import pandas as pd
from heatmap import heatmap

def stepsolver (im_dir, time_eval, grid,rank, iterations, number_processes, gridpart_length, ghostcells, N, rest, comm, stencil,output):
    
    for iteration in range(0, iterations):   
        if rank == 0:

            ############# Send from Rank 0 to other Ranks  ##################################

            for i in range(0, number_processes):
                tag = iteration * number_processes *2 + i
                # separate the grid into overlapping chunks
                if i == 0:
                    continue
                # grid for "last process"
                elif i == number_processes-1:
                    gridpart = grid[gridpart_length * i - ghostcells : N, ] 
                    
                # grid for all prcesses except the master and last
                else:
                    gridpart = grid[(gridpart_length * i) - ghostcells : gridpart_length * (i + 1) + ghostcells, ]
                   
                #send all gridparts to the processes exept the one for process0/master
                comm.Send(gridpart, dest=i, tag=tag)


        if rank == 0:
            gridpart = grid[0 : gridpart_length + ghostcells, ]


        ############ Receive from Rank 0 ################################

        elif rank == number_processes-1:
            tag = iteration * number_processes *2 + rank
            # define empty container for expected data from master
            gridpart = np.empty((gridpart_length + ghostcells + rest, N), dtype=np.float64)
            # receive data from process i and save to empty container
            comm.Recv(gridpart, source=0, tag=tag)

        else:
            tag = iteration * number_processes *2 + rank
            # define empty container for expected data from master
            gridpart = np.empty((gridpart_length + ghostcells * 2, N), dtype=np.float64)

            # receive data from process i and save to empty container
            comm.Recv(gridpart, source=0, tag=tag)



        ############ Calculation ################################

        # start time mesaurement (only of calculation on one node)
        start_time = time()

        for i in range(0, ghostcells):
            #convolve two 2-dimensional arrays using signal convolve 2d from scipy
            gridpart = gridpart + signal.convolve2d(gridpart, stencil, boundary='fill', mode='same')



        # time evaluation
        end_time =time()
        time_diff = end_time - start_time
        time_eval.append(time_diff)


        ########### Send to Rank 0 #################################

        if rank != 0:
            tag = iteration * number_processes *2 + number_processes + rank
            comm.Send(gridpart, dest=0, tag=tag)

        ########## Recollect data ###################################

        if rank == 0:

            grid = np.empty((N, N), dtype=np.float64)

            for i in range(0, number_processes):
                if i == 0:
                    # cut overlap from convolved data
                    return_values = gridpart[0:gridpart_length, ]

                    # insert data from rank 0 into original grid
                    grid[0:gridpart_length,] = return_values

                elif i == number_processes-1:
                    tag = iteration * number_processes *2   + number_processes + i

                    # define empty container for expected data from process i
                    return_values = np.zeros((gridpart_length + ghostcells + rest, N), dtype=np.float64)

                    # receive data from process i and save to empty container
                    comm.Recv(return_values, source=i, tag = tag)
                    # cut overlap from received data
                    return_values = return_values[ghostcells:, ]

                    # insert received data into original grid
                    grid[i * gridpart_length:N] = return_values

                else:
                    tag = iteration * number_processes *2   + number_processes + i

                    # define empty container for expected data from process i
                    return_values = np.empty((gridpart_length + 2 * ghostcells, N), dtype=np.float64)

                    # receive data from process i and save to empty container
                    comm.Recv(return_values, source = i, tag = tag)
                    # cut overlap from received data
                    return_values = return_values[ghostcells:len(return_values) - ghostcells]

                    # insert received data into original grid
                    grid[i * gridpart_length:(i + 1) * gridpart_length, ] = return_values

            #save visualizations
            #Achtung: Für jede Iteration wird ein Bild erstellt, ggf. auskommentieren
            if output:
                heatmap(grid, iteration, ghostcells, im_dir,number_processes)
    if rank==0:
        if output:
            grid=np.matrix(grid)
            grid_df = pd.DataFrame(data=grid.astype(float))
            grid_df.to_csv('final_matrix.csv', sep=' ', header=False, float_format='%.6f', index=False)       
    return(time_eval)
