
if CPU_CORES == 4
    prepender = "/media/MATLAB"
else
    prepender = "/home/roman"
end

include(prepender*"/Julia/MUMPS.jl/src/MUMPS.jl")

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
    # MUMPS SOLVER
    sol = solveMUMPS(A,b[:,1])

    # WRITE BACK THE RESULTS
    writedlm(dir*"/solution",sol)
end

CallMUMPS()