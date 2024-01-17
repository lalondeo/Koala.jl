
### Code for solving the dual of a sparse semidefinite program

# Encodes that sum_{a in coeffs} a_3 X[a_1, a_2] ? b_val, where ? can be = or >= 
# For every (i,j) pair, there must be at most one tuple a in coeffs such that either a[1] = i, a[2] = j or a[1] = j, a[2] = i
mutable struct Constraint 
	coeffs::Vector{Tuple{Int64, Int64, Float64}}
	b_val::Float64
end


function matrify(cons::Constraint, n::Int)
	A = zeros(n,n)
	for x in cons.coeffs
		if(x[1] == x[2])
			A[x[1],x[1]] = x[3]
		else
			A[x[1],x[2]] = A[x[2],x[1]] = x[3]/2
		end
	end
	return A
end

# Encodes a SDP of the form  
#			min tr(PX)
#			tr(A_iX) >= b_i, for every inequality constraint
#			tr(A_iX) = b_i,  for every equality constraint
#			X is positive semidefinite

# What we will actually be solving is the dual SDP
#			max b * y
#			C - transpose(y) * A is positive definite
#			y_i >= 0, for every inequality constraint

mutable struct SDP_Model
	N::Int64 # The size of the matrix variable
	
	constraints_nonneg::Vector{Constraint}
	constraints_eq::Vector{Constraint} 
	objective::Vector{Tuple{Int64, Int64, Float64}}
	
	# Internal data, compiled so that the dual can be solved repeatedly without having to recompute invariant components of the model (such as A_t) each time
	A_t::SparseMatrixCSC{Float64, Int64} 
	b::SparseVector{Float64, Int64}
	P::SparseVector{Float64, Int64}
	
	SDP_Model(N) = new(N, [], [], [])
end


### Internals ###
function conv_2d_to_triangle(i,j) 
	if(i > j)
		return conv_2d_to_triangle(j,i)
	else
		return i + div(j * (j-1), 2)
	end
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


### Compilation functions ###
"""
	compile_constraints!(model::SDP_Model)

Given a model, builds the internal objects A_t and b
"""
function compile_constraints!(model::SDP_Model)
	l1 = length(model.constraints_nonneg)
	l2 = length(model.constraints_eq)
	# A
	I::Vector{Int64} = []
	J::Vector{Int64} = []
	V::Vector{Float64} = []
	
	# b
	b::Vector{Float64} = []
	
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

"""
	compile_objective!(model::SDP_Model; offset = 1e-05)

Given a model, builds the internal objective. offset is subtracted from the diagonal of the objective matrix for 
"""
function compile_objective!(model::SDP_Model; offset = 1e-05)
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


### Functions related to the optimization of the model and the validation of the result ###

function callback(model::SDP_Model, target_val::Float64, ws, n::Int; epsilon = 1e-05)
	if(n % 5 == 0)
		_x = ws.sm.D * ws.vars.x
		k = length(model.constraints_nonneg) + length(model.constraints_eq)
		return abs(validate_dual_solution(model::SDP_Model, _x[1:k]; formal = false) - target_val) < epsilon
	end
	
	return false
end

"""
	find_dual_solution(model::SDP_Model; target_val::Float64 = Inf, kwargs...)

Given a model, optimizes its dual and returns the corresponding dual variable. If target_val is not set to Inf, we add the additional constraint that the objective value be equal to target_val. 
The other keyword arguments are passed to COSMO. 
"""
function optimize_dual(model::SDP_Model; target_val::Float64 = Inf, kwargs...)
	kwargs = Dict{Symbol, Any}(kwargs)
	@assert !haskey(kwargs, :scaling) || kwargs[:scaling] != 0
	if(!haskey(kwargs, :eps_abs))
		kwargs[:eps_abs] = 1e-08
	end
	
	if(!haskey(kwargs, :eps_rel))
		kwargs[:eps_rel] = 1e-08
	end
	
	if(target_val != Inf && !haskey(kwargs, :callback))
		kwargs[:callback] = (ws, n) -> callback(model, target_val, ws, n)
	end
	
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

	assemble!(real_model, spzeros(M, M), -model.b, constraints, settings = COSMO.Settings(;kwargs...))
	
	res = COSMO.optimize!(real_model)
	return res.x
end

"""
	validate_dual_solution(model::SDP_Model, y::Vector{Float64}; formal::Bool = false)::Float64

Given a SDP model and a dual solution y for the model, returns the objective value corresponding to y if y could be proven to be dual feasible and -Inf otherwise. 
If formal = false, isposdef is used, and otherwise we work harder to obtain a proof in exact arithmetic that y is dual feasible.
"""
function validate_dual_solution(model::SDP_Model, y::Vector{Float64}; formal::Bool = false)::Float64
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
		if(i <= l_nn) # Nonnegativity constraint
			y[i] = max(y[i], 0.0)
		end

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
	return val_obj

	# Prove that R is positive definite
	if((formal && prove_positive_definiteness(R)) || (!formal && isposdef(R)))
		return val_obj
	else
		return -Inf
	end
	
end

"""
	prove_positive_definiteness(M::Matrix{Float64}, n_digits = 30)

Given a matrix, attempts to prove in exact arithmetic that it is positive semidefinite. n_digits is the number of digits of precision used in the course of the proof. 
"""
function prove_positive_definiteness(M::Matrix{Float64}, n_digits = 30)
	@assert n_digits % 2 == 0
	B = Int128(10)^n_digits
	
	# Make sure that the computation of M_ won't overflow
	if(BigFloat(maximum(abs, M)) * B > BigFloat(2.0)^126)
		return false
	end
	# This corresponds to rounding everything in M to n_digits and multiplying the result by B to obtain a matrix with integer entries
	M_ = broadcast((x)->Int128(round(x*B)), M)
	
	n = size(M)[1]
	min_val = eigmin(M)
	if(min_val < 0) return false end

	k = min_val/2;
	k_ = Int128(ceil(k*B));
	
	I = diagm([1.0 for i=1:n]);
	I_ = diagm([Int128(1) for i=1:n]);

	B_L = Int128(10)^div(n_digits, 2)
	L_ = broadcast((x)->Int128(round(x*B_L)), Matrix(cholesky(M - k * I).L))
	
	# Make sure that the computation of difference below won't overflow
	if(BigInt(maximum(abs, L_))^2 * n + k_ + maximum(abs, M_) > BigInt(2)^126)
	   return false
	end	
	
	# This is the difference between M_ and a matrix whose smallest eigenvalue is clearly greater or equal to k_
	# By showing that the Frobenius norm of this is strictly smaller than k_, we've shown that M_ is positive definite too, 
	# as the Frobenius norm upper bounds the spectral norm
	difference = M_ - k_ * I_ - L_ * transpose(L_) 
	
	# Make sure that the computation of norm(difference) won't overflow
	if(BigInt(maximum(abs, difference))^2 * n^2 > BigInt(2)^126)
		return false
	end
	
	# ||BM - k_ * I_ - L_ * transpose(L_)|| <= norm(difference) + n^2 by the triangle inequality
	return norm(difference) + n^2 < k_
end

