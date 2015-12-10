
include("/home/roman/Julia/MUMPS.jl/src/MUMPS.jl")
using MUMPS
using MAT

function CallMUMPS()
	
	# dir = pwd()
	dir = "/home/roman/Dropbox/Florence/Core/FiniteElements/Solvers"
	# READ I,J & V ARRAYS FROM FILES
	file = matopen(dir*"/JuliaDict.mat")
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
	# fileb = matopen("JuliaDict.mat", "w")
	# write(file, "sol", sol)
	# close(fileb)
end

CallMUMPS()

# function CallMUMPS()
	
# 	# dir = pwd()
# 	dir = "/home/roman/Dropbox/Florence/Core/FiniteElements/Solvers"
# 	# READ I,J & V ARRAYS FROM FILES
# 	rowIndA = convert(Array{Int64,2},readdlm(dir*"/rowIndA"))
# 	colPtrA = convert(Array{Int64,2},readdlm(dir*"/colPtrA"))
# 	valuesA = convert(Array{Float64,2},readdlm(dir*"/valuesA"))
# 	shapeA = convert(Array{Int64,2},readdlm(dir*"/shapeA"))

# 	b = convert(Array{Float64,2},readdlm(dir*"/rhs"))
# 	b = b[:,1]

# 	m = shapeA[1]
# 	n = shapeA[2]

# 	# BUILD THE SPARSE MATRIX
# 	A = sparse(rowIndA[:,1],colPtrA[:,1],valuesA[:,1],m,n)
# 	# MUMPS SOLVER
# 	sol = solveMUMPS(A,b)

# 	# WRITE BACK THE RESULTS
# 	writedlm(dir*"/solution",sol)
# end

# CallMUMPS()