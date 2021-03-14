import numpy as np
from time import time
from scipy import signal
from mpi4py import MPI
import pandas as pd
from heatmap import heatmap
import math



def stepsolver2 (sqrt_proc,time_eval, grid,rank, iterations, number_processes, gridpart_length, ghostcells, N, rest, comm, stencil,output,im_dir):
   
    grid_ghost = gridpart_length + ghostcells
    grid_ghost_rest = grid_ghost + rest
    grid_ghost2 = gridpart_length + ghostcells*2
    sqrt_proc = int(sqrt_proc)
    for iteration in range(0, iterations):   
        if rank == 0:
                    
            ###### Comment this in to set boundary condition as "constant" ####################
            # for i in range(N):
            #     grid[i][0] = 20
            #     grid[i][N - 1] = -20
            #     grid[0][i] = 2
            #     grid[N-1][i] = -20
            #      #print(grid)
            #########################################################################################

            ############# Send Data from Rank 0 to other Ranks  ##################################
            j=0
            for i in range(0, number_processes):
                grid_i_ghost = (gridpart_length * i) - ghostcells
                tag = iteration * number_processes *2 + i
                
                # separate the grid into overlapping chunks
                if i == 0:
                    continue
                if i%sqrt_proc==0:
                    j+=1
                    grid_j_ghost = (gridpart_length * j) - ghostcells
                    grid_j1_ghost = gridpart_length * (j+1) + ghostcells
                # letzte reihe
                if i >= (number_processes-sqrt_proc):
                    #letzte reihe erste spalte
                    if i%sqrt_proc==0:
                        gridpart = np.ascontiguousarray(grid[grid_j_ghost : N , (gridpart_length * (i-j*(sqrt_proc))) : (gridpart_length) * (i-j*(sqrt_proc)+1)+ghostcells ])
                    elif i == number_processes-1:
                        gridpart = np.ascontiguousarray(grid[grid_j_ghost : N , (gridpart_length * (i-j*(sqrt_proc))) - ghostcells : ])
                    #letzte reihe mitte
                    else:
                        gridpart = np.ascontiguousarray(grid[grid_j_ghost : N , (gridpart_length * (i-j*(sqrt_proc))) - ghostcells : (gridpart_length) * (i-j*(sqrt_proc)+1)+ghostcells ])
                       
                   
                else:
                    #erste reihe 
                    if(gridpart_length*i<N):
                        #erste reihe ende
                        if(gridpart_length*(i+1)==N-rest):
                            gridpart = np.ascontiguousarray(grid[ : grid_ghost, grid_i_ghost : N ])
                            
                        #erste reihe mitte
                        else:
                            gridpart = np.ascontiguousarray(grid[ : grid_ghost, grid_i_ghost : (gridpart_length) * (i + 1) + ghostcells ])
                            
                    #mittlere reihe
                    if(gridpart_length*i>=(N-rest)):

                        if (i%sqrt_proc)==0:
                            gridpart = np.ascontiguousarray(grid[grid_j_ghost : grid_j1_ghost, 0 : grid_ghost])
                        elif (i+1)%sqrt_proc==0:
                            gridpart = np.ascontiguousarray(grid[grid_j_ghost : grid_j1_ghost, (gridpart_length * (i-j*sqrt_proc)) - ghostcells : ])

                        else:
                            gridpart = np.ascontiguousarray(grid[grid_j_ghost : grid_j1_ghost, (gridpart_length * (i-j*sqrt_proc)) - ghostcells : (gridpart_length) * (i-j*sqrt_proc+1)+ghostcells])
                            
                
                comm.Send(gridpart, dest=i, tag=tag)
            #####################################################################################################################################################################################

        
        if rank == 0:
            gridpart = grid[0 : grid_ghost, 0 : grid_ghost]
        
                
       

        ############ Receive Data from Rank 0 ###############################################

        #letzter prozess,letzte reihe
        elif rank == number_processes-1:
            tag = iteration * number_processes *2 + rank
            
            # define empty container for expected data from master
            gridpart = np.empty((grid_ghost_rest, grid_ghost_rest), dtype=np.float64)
            
            # receive data from process i and save to empty container
            comm.Recv(gridpart, source=0, tag=tag)
        #letzte reihe, erster
        elif rank == number_processes-sqrt_proc:
            tag = iteration * number_processes *2 + rank
            
            # define empty container for expected data from master
            gridpart = np.empty((grid_ghost_rest, grid_ghost), dtype=np.float64)
            
            # receive data from process i and save to empty container
            comm.Recv(gridpart, source=0, tag=tag)
        
        #letzte reihe mitte
        elif rank >= number_processes-sqrt_proc:
            tag = iteration * number_processes *2 + rank
            
            # define empty container for expected data from master
            gridpart = np.empty((grid_ghost_rest, grid_ghost2), dtype=np.float64)
            
            # receive data from process i and save to empty container
            comm.Recv(gridpart, source=0, tag=tag)
        
        #erste reihe mittlerer
        elif rank < sqrt_proc-1:
            tag = iteration * number_processes *2 + rank
            
            # define empty container for expected data from master
            gridpart = np.empty((grid_ghost , grid_ghost2), dtype=np.float64)
            
            # receive data from process i and save to empty container
            comm.Recv(gridpart, source=0, tag=tag)
        #erste reihe letzter
        elif rank < sqrt_proc:
            tag = iteration * number_processes *2 + rank

            # define empty container for expected data from master
            
            gridpart = np.empty((grid_ghost , grid_ghost+rest), dtype=np.float64)
            
            # receive data from process i and save to empty container
            comm.Recv(gridpart, source=0, tag=tag)
        #mittlere reihe erster
        elif rank%sqrt_proc==0:
            tag = iteration * number_processes *2 + rank

            # define empty container for expected data from master
            gridpart = np.empty((grid_ghost2 , grid_ghost ), dtype=np.float64)
            
            # receive data from process i and save to empty container
            comm.Recv(gridpart, source=0, tag=tag)
            
        #mittlere reihe letzter
        elif (rank+1)%sqrt_proc==0:
            tag = iteration * number_processes *2 + rank
            
            # define empty container for expected data from master
            gridpart = np.empty((grid_ghost2 , grid_ghost+rest ), dtype=np.float64)
             
            # receive data from process i and save to empty container
            comm.Recv(gridpart, source=0, tag=tag)
        #mittlere reihe mitte
        else:
            tag = iteration * number_processes *2 + rank
            # define empty container for expected data from master
            gridpart = np.empty((grid_ghost2 , grid_ghost2 ), dtype=np.float64)
            
            # receive data from process i and save to empty container
            comm.Recv(gridpart, source=0, tag=tag)

        ####################################################################################################################




        ############ Calculation ######################################################################################

        # start local time mesaurement of the time that the process needs for the calculation
        start_time = time()

        for i in range(0, ghostcells):
            #convolve two 2-dimensional arrays using signal convolve 2d from scipy
            gridpart = gridpart + signal.convolve2d(gridpart, stencil, boundary='fill', mode='same')



        # end local time measurement 
        end_time =time()
        time_diff = end_time - start_time
        time_eval.append(time_diff)
        ##################################################################################################################




        ########### Send data to Rank 0 #########################################

        if rank != 0:
            tag = iteration * number_processes *2 + number_processes + rank
            comm.Send(gridpart, dest=0, tag=tag)

        ####################################################################




        ########## Recollect data #############################################################

        if rank == 0:

            #introduce parameter k for row management
            k=0

            grid = np.zeros((N, N), dtype=np.float64)

            for i in range(0, number_processes):
                if i == 0:
                    # cut overlap from convolved data
                    return_values = gridpart[0:gridpart_length,0:gridpart_length]

                    grid[0:gridpart_length,0:gridpart_length] = return_values

                    continue

                #increase k if "process row" increases
                if (i%sqrt_proc==0):
                    k+=1
                    g_k = gridpart_length * k
                    k_s = k*sqrt_proc
                    k1 = k+1
                #letzte reihe
                if i >= number_processes-sqrt_proc:
                    if ((i+1)%sqrt_proc==0):
                        tag = iteration * number_processes *2 + number_processes + i
                        # define empty container for expected data from process i
                        return_values = np.zeros((grid_ghost_rest, grid_ghost_rest), dtype=np.float64)

                        # receive data from process i and save to empty container
                        comm.Recv(return_values, source=i, tag = tag)
                        # cut overlap from received data
                        return_values = np.ascontiguousarray(return_values[ghostcells: ,ghostcells: ])
                        
                        grid[g_k: N, gridpart_length * (i-k_s) : N]=return_values
                        
                    #letzte reihe anfang        
                    elif(((i)%sqrt_proc==0)):
                        tag = iteration * number_processes *2 + number_processes + i

                        # define empty container for expected data from process i
                        return_values = np.zeros((grid_ghost_rest, grid_ghost), dtype=np.float64)

                        # receive data from process i and save to empty container
                        comm.Recv(return_values, source=i, tag = tag)
                
                        # cut overlap from received data
                        return_values = np.ascontiguousarray(return_values[ghostcells: , :gridpart_length ])
                         
                        grid[g_k: N,  : gridpart_length ]=return_values

                    #letzte reihe mitte      
                    elif((i%sqrt_proc>0)):
                        tag = iteration * number_processes *2 + number_processes + i

                        # define empty container for expected data from process i
                        return_values = np.zeros((grid_ghost +rest, grid_ghost2), dtype=np.float64)
                        
                        # receive data from process i and save to empty container
                        comm.Recv(return_values, source=i, tag = tag)
                        
                        # cut overlap from received data
                        return_values = np.ascontiguousarray(return_values[ghostcells: ,ghostcells:grid_ghost])
                        
                        
                        grid[g_k: N, gridpart_length * (i-k_s) : gridpart_length * (i-k_s+1) ]=return_values
                        
                    
                #mittlere reihe anfang
                elif (k>0):
                    if (i%sqrt_proc==0):
                        tag = iteration * number_processes *2 + number_processes + i

                        # define empty container for expected data from process i
                        return_values = np.zeros((grid_ghost2 , grid_ghost), dtype=np.float64)
                        
                        # receive data from process i and save to empty container
                        comm.Recv(return_values, source = i, tag = tag)
                        
                        # cut overlap from received data
                        return_values = np.ascontiguousarray(return_values[ghostcells:grid_ghost,:gridpart_length])
                    
                        grid[k * gridpart_length:(k1)* gridpart_length, :gridpart_length] = return_values
                        
                    #mittlere reihe ende
                    elif((i+1)%sqrt_proc==0):
                        tag = iteration * number_processes *2 + number_processes + i
                        
                        # define empty container for expected data from process i
                        return_values = np.zeros((grid_ghost2 , grid_ghost+rest), dtype=np.float64)
                        
                        # receive data from process i and save to empty container
                        comm.Recv(return_values, source = i, tag = tag)
                       
                        # cut overlap from received data
                        return_values = np.ascontiguousarray(return_values[ghostcells:grid_ghost, ghostcells:])
                    
                        grid[k * gridpart_length:(k1)* gridpart_length, (i-k_s) * gridpart_length:] = return_values
                        
                    #mittlere reihe mitte
                    elif(i%sqrt_proc>0):
                        tag = iteration * number_processes *2 + number_processes + i

                        # define empty container for expected data from process i
                        return_values = np.zeros((grid_ghost2 , grid_ghost2), dtype=np.float64)
                        
                        # receive data from process i and save to empty container
                        comm.Recv(return_values, source = i, tag = tag)
                    
                        # cut overlap from received data
                        return_values = np.ascontiguousarray(return_values[ghostcells:grid_ghost,ghostcells:grid_ghost])
                    
                        grid[k * gridpart_length:(k1)* gridpart_length, (i-k_s) * gridpart_length: (i-k_s+1) * gridpart_length] = return_values
                        
               
                    
                
                #erste reihe ende
                elif(k==0):
                    if (i+1 == sqrt_proc):
                        tag = iteration * number_processes *2 + number_processes + i

                        # define empty container for expected data from process i
                        return_values = np.zeros((grid_ghost , grid_ghost+ rest), dtype=np.float64)
                        
                        # receive data from process i and save to empty container
                        comm.Recv(return_values, source = i, tag = tag)

                        # cut overlap from received data
                        return_values = np.ascontiguousarray(return_values[0:gridpart_length,ghostcells:])
                    
                        grid[:gridpart_length, i * gridpart_length:] = return_values
                        
                #erste reihe mitte
                    elif(i!=0):
                        tag = iteration * number_processes *2 + number_processes + i

                        # define empty container for expected data from process i
                        return_values = np.zeros((grid_ghost , grid_ghost2), dtype=np.float64)
                        
                        # receive data from process i and save to empty container
                        comm.Recv(return_values, source = i, tag = tag)
                    
                        # cut overlap from received data
                        return_values = np.ascontiguousarray(return_values[0:gridpart_length,ghostcells:grid_ghost])
                        
                        grid[  :gridpart_length, i * gridpart_length:(i + 1) * gridpart_length] = return_values
                
                
              
        
        ###############################################################################################################################                
                
                
                
        ###### Save Visualizations#########################################################################
            #Achtung: Für jede Iteration wird ein Bild erstellt, ggf. auskommentieren
            if output: 
                heatmap(grid, iteration, ghostcells, im_dir, number_processes)
                
    if rank==0:
        if output:
            grid=np.matrix(grid)
            grid_df = pd.DataFrame(data=grid.astype(float))
            grid_df.to_csv('final_matrix.csv', sep=' ', header=False, float_format='%.6f', index=False)
        #######################################################################################################
        
    return(time_eval)
