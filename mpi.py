from mpi4py import MPI
import numpy as np
import sys
from mpi_util import heat_parallel, create_directory, def_stencil, time_measurement
import math

######################################################################################################################
    # Parralel Computing Projektarbeit im Wintersemester 2020/2021 von Anne Schulte-Kroll (Matrikelnummer: 192904)
######################################################################################################################

if __name__ == "__main__":
    # Input error exception
    if len(sys.argv) != 6:
        raise Exception("please input the necessary parameters as follows:\n"
                        "Usage: mpirun -n [num-proc] [-hostfile hosts.txt] python3 (1) (2) (3) (4) (5) (6) \n"
                        "Example: mpirun python3 mpi.py 1000000 1e-6 10 100 True \n"
                        "1:\t 'filename.py'\n"
                        "2:\t N: Matric dimension \n"
                        "3:\t alpha: 0 < thermal diffusitvity < 1e-6\n"
                        "4:\t Ghostcells: overlapping parts of grid \n"
                        "5:\t iterations/communatations: Number of communication steps\n (Effective number of Iterations = Ghostcells*iterations)"
                        "6:\t True/False: True if you want to generate output, False if not \n")

    # define some input values
    N = int(sys.argv[1]) # to define Matrix dimension (Matrix dimension: N*N)
    alpha = float(sys.argv[2]) # thermal diffusitivity
    ghostcells = int(sys.argv[3]) # ghostcells
    iterations = int(sys.argv[4]) # number of timesteps (iterations)
    output = bool(sys.argv[5] == "True") #output or no output

    # N=10000
    # alpha=float(1e-6)
    # ghostcells=500
    # iterations=2
    # output=bool(False)

    boundary = 20.0
    initial = 0.0
    
    ################################## initialize Variables #############################################

    comm = MPI.COMM_WORLD # intracommunicator instance
    rank = comm.Get_rank() # The process rank (Example: 0,1 for 2 processes)
    number_processes = comm.Get_size() # The number of processes in the communicator 
   
    h = 1 / (N + 1) # The stepsize

    # calculate gridpart size and rest

    sqrt_proc=math.sqrt(number_processes)

    if ((int(sqrt_proc)-sqrt_proc)==0): 
        sqrt = bool(True)
    else:
        sqrt = bool(False)
    
    if sqrt:
        gridpart_length = int(N / sqrt_proc)
        rest = N % gridpart_length
        
    else:
        gridpart_length = int(N / number_processes)
        rest = N % number_processes
        

    dt = h * h / (4 * alpha) 
    

    ###################################################################################################################


    ################################### call Functions  ###############################################################

    #create the stencil
    stencil = def_stencil(alpha, dt, h)


    #create file with time measurements and start calculation
    time_measurement(sqrt, sqrt_proc, initial, boundary, rank, iterations, number_processes, gridpart_length, ghostcells, N, rest, comm, stencil, output, alpha)


    ###################################################################################################################



