
include("/home/roman/Julia/MUMPS.jl/src/MUMPS.jl")
using MUMPS

function CallMUMPS()
	
	# dir = pwd()
	dir = "/home/roman/Dropbox/Florence/Core/FiniteElements/Solvers"
	# READ I,J & V ARRAYS FROM FILES
	rowIndA = convert(Array{Int64,2},readdlm(dir*"/rowIndA"))
	colPtrA = convert(Array{Int64,2},readdlm(dir*"/colPtrA"))
	valuesA = convert(Array{Float64,2},readdlm(dir*"/valuesA"))
	shapeA = convert(Array{Int64,2},readdlm(dir*"/shapeA"))

	b = convert(Array{Float64,2},readdlm(dir*"/rhs"))
	b = b[:,1]

	m = shapeA[1]
	n = shapeA[2]

	# BUILD THE SPARSE MATRIX
	A = sparse(rowIndA[:,1],colPtrA[:,1],valuesA[:,1],m,n)
	# MUMPS SOLVER
	sol = solveMUMPS(A,b)
	# print(sol[])

	# WRITE BACK THE RESULTS
	writedlm(dir*"/solution",sol)
end

CallMUMPS()