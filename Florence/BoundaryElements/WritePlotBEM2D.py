from __future__ import print_function, division
import numpy as np
import imp

# Pr = imp.load_source('Problem_Arch','BoundaryElements/Problem_Rectangle/ProblemDataBEM_1.py')
# Pr = imp.load_source('Problem_Arch','BoundaryElements/Problem_Arch/ProblemDataBEM_2.py')
# Pr = imp.load_source('Problem_Arch','BoundaryElements/Problem_Rectangle_LM/ProblemDataBEM_LM_1.py')
# Pr = imp.load_source('Problem_Arch','BoundaryElements/Problem_Arch_LM/ProblemDataBEM_LM_2.py')

def WritePlotBEM2D(sol,total_sol,POT,FLUX1,FLUX2,LHS2LHS,LHS2RHS,mesh,printt=1,plot=1,plotopt=1,write=0):

    # Separate potential and flux
    pot = sol[np.array(LHS2LHS,dtype=int)]
    dpot = sol[np.array(LHS2RHS,dtype=int)]

    if write==1:
        np.savetxt('Results.txt',total_sol)
        print('The BE solution at boundary point is written in text file "Results.txt" - both potential and flux')
        np.savetxt('POT.txt',total_sol)
        print('The potential values at internal points are written in text file "POT.txt"')
        np.savetxt('FLUX1.txt',total_sol)
        print('The X-direction flux values at internal points are written in text file "FLUX1.txt"')
        np.savetxt('FLUX2.txt',total_sol)
        print('The X-direction flux values at internal points are written in text file "FLUX2.txt"')

    if plot==1:
        # Call the plot function from the problem data file
        Pr.PlotFunc(mesh,POT,FLUX1,FLUX2,plotopt)

    if printt ==1:
        print(POT)
        # print(FLUX1)
        # print(FLUX2)