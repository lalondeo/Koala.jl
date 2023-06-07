using COSMO
using SparseArrays
using LinearAlgebra
using Suppressor

# For NPA hierarchies, we bypass the JuMP interface as the problems encountered tend to be very large and very sparse, and JuMP does not handle such problems well


# Encodes that sum_{a in coeffs} a_3 X[a_1, a_2] ? b_val, where ? is = or >= 
# For every (i,j) pair, there must be at most one tuple a in coeffs such that either a[1] = i, a[2] = j or a[1] = j, a[2] = i
struct Constraint 
	coeffs::Array{Tuple{Int64, Int64, Float64}}
	b_val::Float64
end

# Encodes a SDP of the form  
#			min tr(PX)
#			tr(A_iX) >= b_i, for every nonneg constraints
#			tr(A_iX) = b_i,  for every eq constraint
#			X is positive semidefinite

mutable struct SDP_Model
	N::Int64 # For a NxN SDP matrix
	
	constraints_nonneg::Array{Constraint}
	constraints_eq::Array{Constraint} 
	objective::Array{Tuple{Int64, Int64, Float64}}
	
	# Internal data
	A_t::SparseMatrixCSC{Float64, Int64} 
	b::SparseVector{Float64, Int64}
	P::SparseVector{Float64, Int64}
	
	SDP_Model(N) = new(N, [], [], [])
end


function conv_2d_to_triangle(i,j) 
	if(i > j)
		return conv_2d_to_triangle(j,i)
	else
		return i + div(j * (j-1), 2)
	end
end

function compile_constraints!(model::SDP_Model)
	l1 = length(model.constraints_nonneg)
	l2 = length(model.constraints_eq)
	# A
	I::Array{Int64} = []
	J::Array{Int64} = []
	V::Array{Float64} = []
	
	# b
	b::Array{Float64} = []
	
	sqrt2 = sqrt(2)
	for i=1:(l1 + l2)
		constraint = (i <= l1 ? model.constraints_nonneg[i] : model.constraints_eq[i - l1])
		for tup in constraint.coeffs
			push!(I, conv_2d_to_triangle(tup[1], tup[2]))
			push!(J, i)
			push!(V, tup[1] == tup[2] ? tup[3] : tup[3] / sqrt2)
		end
		push!(b, constraint.b_val)
	end
	
	model.A_t = sparse(I, J, V, div(model.N * (model.N+1), 2), l1 + l2)
	model.b = sparse(b)
		
end

function compile_pseudo_objective!(model::SDP_Model; offset = 1e-05)
	I::Array{Int64} = []
	V::Array{Float64} = []
	sqrt2 = sqrt(2)
	
	diagonal_elements_hit = [false for i=1:model.N]
	
	for tup in model.objective
		push!(I, conv_2d_to_triangle(tup[1], tup[2]))
		if(tup[1] == tup[2])
			push!(V, tup[3] - offset)
			diagonal_elements_hit[tup[1]] = true
		else
			push!(V,  tup[3] / sqrt2)
		end
	end
	
	for i=1:model.N
		 if(!(diagonal_elements_hit[i]))
			push!(I, conv_2d_to_triangle(i,i))
			push!(V, -offset)
		 end
	 end
	
	model.P = sparsevec(I, V, div(model.N * (model.N+1), 2))
end

function triangle_to_matrix_conversion(v, n)
	v = sparse(v)
	pos, vals = findnz(v)
	N = length(v)

	M = zeros(n,n)
	
	i = 1
	j = 1
	index = 1
	for k=1:length(v)
		if(index <= length(pos) && pos[index] == k)
			if(i == j)
				M[i,j] = vals[index]
			else
				M[i,j] = M[j,i] = vals[index] / sqrt(2)
			end
			index += 1
		end
		
		if(i==j)
			j += 1
			i = 1
		else
			i += 1
		end
	end
	return M
end

struct found_y
	y::Array{Float64}
end

function find_dual_solution(model::SDP_Model, target_val::Float64 = Inf, iterations::Int64 = 5000; verbose = true)
	real_model = COSMO.Model()	
	N_t, M = size(model.A_t)
	l_nn = length(model.constraints_nonneg)
	
	constraints = [COSMO.Constraint(-model.A_t, model.P, COSMO.PsdConeTriangle)]
	if(target_val != Inf) 
		push!(constraints, COSMO.Constraint(sparse(transpose(model.b)), [-target_val], COSMO.ZeroSet))
	end
			
	if(l_nn != 0)
		I_nonneg = sparse([i for i=1:l_nn], [i for i=1:l_nn], [1.0 for i=1:l_nn], l_nn, M)
		push!(constraints, COSMO.Constraint(I_nonneg, sparsevec([0.0 for i=1:l_nn]), COSMO.Nonnegatives)) # y_i must be nonnegative for 1 <= y_i <= l_nn
	end
	assemble!(real_model, spzeros(M, M), -model.b, constraints, settings = COSMO.Settings(verbose=true, adaptive_rho = true, rho = 0.1, alpha = 1.0, accelerator = AndersonAccelerator{Float64, Type1, RollingMemory, NoRegularizer}, check_infeasibility = 1000, verbose_timing = true, check_termination = 20, eps_abs = 1e-12, sigma = 1e-07, eps_rel = 1e-12, decompose = false, eps_prim_inf = 1e-12, max_iter = iterations))
	
	try
		if(verbose)
			res = COSMO.optimize!(real_model)
			return res.x
		else
			@suppress res = COSMO.optimize!(real_model)
			return res.x
		end
	catch e
		return e.y
	end
	
end


function validate_dual_solution(model::SDP_Model, y::Array{Float64})
	N = model.N
	R = zeros(N, N) # This will contain P - A_t y
	@assert length(model.constraints_nonneg) + length(model.constraints_eq) == length(y)
	
	for tup in model.objective
		if(tup[1] == tup[2])
			R[tup[1], tup[2]] += tup[3]
		
		else
			# Because the objective matrix is symmetric, off-diagonal elements are counted twice, so we divide by two to balance this out
			R[tup[1], tup[2]] += tup[3]/2
			R[tup[2], tup[1]] += tup[3]/2
		end
	end

	l_nn = length(model.constraints_nonneg)
	val_obj::Float64 = 0
	
	for i=1:length(y)
		if(i <= l_nn && y[i] < 0) return -Inf end # The first l_nn dual variables correspond to inequality constraints, and must therefore be nonnegative
		constraint = (i <= l_nn ? model.constraints_nonneg[i] : model.constraints_eq[i - l_nn])
		for tup in constraint.coeffs
			if(tup[1] == tup[2])
				R[tup[1],tup[2]] -= y[i] * tup[3]
			else
				# Same as before
				R[tup[1],tup[2]] -= y[i] * tup[3] / 2 
				R[tup[2],tup[1]] -= y[i] * tup[3] / 2
			end
		end
		val_obj += y[i] * constraint.b_val
	end

	k = eigmin(R)
	# Prove that R is positive definite
	if(k > 1e-08 && isposdef(R-diagm([k/2 for i=1:N])))
		return val_obj
	else
		return -Inf
	end
	
end

