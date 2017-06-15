
if Sys.CPU_CORES == 4
    prepender = "/media/MATLAB"
else
    prepender = "/home/roman/Downloads"
end

# include(prepender*"/Julia/MUMPS.jl/src/MUMPS.jl")
include(prepender*"/MUMPS.jl/src/MUMPS.jl")

using MUMPS
using MAT

function CallMUMPS()
    
    # READ I,J & V ARRAYS FROM FILES
    filename = @__FILE__
    dir = filename[1:end-13]

    file = matopen(dir*"JuliaDict.mat")
    rowIndA = read(file,"rowIndA")
    colPtrA = read(file,"colPtrA")
    valuesA = read(file,"valuesA")
    shapeA = read(file,"shapeA")
    b = read(file,"rhs")
    close(file)

    m = shapeA[1]
    n = shapeA[2]

    # BUILD THE SPARSE MATRIX
    A = sparse(rowIndA[1,:],colPtrA[1,:],valuesA[1,:],m,n)
    b = b'
    # MUMPS SOLVER
    t_solve = time()
    sol = solveMUMPS(A,b[:,1])
    println("MUMPS solver time is: ", time() - t_solve)

    # WRITE BACK THE RESULTS
    # writedlm(dir*"/solution",sol)
    file = matopen(dir*"/solution.mat", "w")
    write(file, "solution", sol)
    close(file)
end

CallMUMPS()