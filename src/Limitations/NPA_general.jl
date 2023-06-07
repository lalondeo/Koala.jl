using COSMO
using JuMP

using SparseArrays
using LinearAlgebra
using Suppressor

include("NPA_utils.jl")

### Monomial arithmetic ###
struct Monomial
	alice_part::Array{Projector}
	bob_part::Array{Projector}
end

function eta(m::Monomial)::Monomial
	if(isempty(m.alice_part))
		return Monomial([], eta(m.bob_part))
	elseif(isempty(m.bob_part))
		return Monomial(eta(m.alice_part), [])
	else
		return m
	end
end


Base.copy(m::Monomial) = Monomial(copy(m.alice_part), copy(m.bob_part))

function (*)(m1::Monomial, m2::Monomial)
	return Monomial(m1.alice_part*m2.alice_part, m1.bob_part*m2.bob_part)

end

function (==)(m1::Monomial, m2::Monomial)
	return (m1.alice_part == m2.alice_part) && (m1.bob_part == m2.bob_part)
end



Base.isequal(m1::Monomial, m2::Monomial) = (m1 == m2)
Base.hash(m::Monomial) = hash(m.alice_part) + hash(m.bob_part)
Base.length(m::Monomial) = length(m.alice_part) + length(m.bob_part)

function reverse(m::Monomial)::Monomial
	m = copy(m)
	reverse!(m.alice_part)
	reverse!(m.bob_part)
	return m
end



### Actual hierarchy ###


struct NPAGeneral <: NPA
	n_X::Int64
	n_Y::Int64
	n_A::Int64
	n_B::Int64
	level::Int64
	
	correlation_components::Dict{Tuple{Int,Int,Int,Int}, Tuple{Int,Int}}
	model::SDP_Model

	function NPAGeneral(n_X::Int64, n_Y::Int64, n_A::Int64, n_B::Int64, level::Int64; filtering::Filtering = full, impose_maximally_entangled = false)
		### Building atomic monomials from the atomic monomials of Alice and Bob
		atomic_monomials_alice = build_atomic_monomials(n_X, n_A, level, filtering)
		atomic_monomials_bob = build_atomic_monomials(n_Y, n_B, level, filtering)
		
		atomic_monomials::Array{Monomial} = []
		for m1 in atomic_monomials_alice
			for m2 in atomic_monomials_bob
				if(length(m1) + length(m2) > level)
					break 
				end
				push!(atomic_monomials, Monomial(m1,m2))
			end
		end
		
		N = length(atomic_monomials)
		model = SDP_Model(N)
		push!(model.constraints_eq, Constraint([(1, 1, 1.0)], 1.0)) # Normalization constraint
		

		
		### Building monomials and enforcing zero constraints
		monomials::Dict{Monomial, Tuple{Int,Int}} = Dict()

		for i=1:N
			for j=i:N
				try
					m = reverse(atomic_monomials[i]) * atomic_monomials[j]
					if(impose_maximally_entangled)
						m = eta(m)
					end
				
					if(haskey(monomials, m))
						i2, j2 = monomials[m]
						push!(model.constraints_eq, Constraint([(i, j, 1.0), (i2, j2, -1.0)], 0.0))
					else
						monomials[m] = (i,j)
					end
					
				catch ZeroException
					push!(model.constraints_eq, Constraint([(i, j, 1.0)], 0.0)) # Force the component to zero
				end
			end
		end
		
		### Forcing values corresponding to probabilities to be nonnegative
		for x=1:n_X
			for y=1:n_Y
				for a=1:n_A
					for b=1:n_B
						index = monomials[Monomial([Projector(x,a)], [Projector(y,b)])]
						push!(model.constraints_nonneg, Constraint([(index[1], index[2], 1.0)], 0.0))
					end
				end
			end
		end
		
		### Adding some POVM constraints
		for monomial in keys(monomials)
			if(length(monomial) < 2*level)
				alice_part = monomial.alice_part
				bob_part = monomial.bob_part
				
				# Adding POVM constraints corresponding to Alice's projectors here
				for x=1:n_X
					try
						coeffs = [(monomials[Monomial(alice_part[1:m] * [Projector(x,a)] * alice_part[m+1:end], bob_part)]..., 1.0) for a=1:n_A]
						if(length(Set(coeffs)) != n_A || (monomials[monomial]..., 1.0) in coeffs) 
							continue
						end
						push!(model.constraints_eq, Constraint([coeffs; (monomials[monomial]..., -1.0)], 0.0))
					catch # Either we hit a KeyError or a ZeroException. In either case, the constraint can't be added and we move on.
					end
				end
				
				# Adding POVM constraints corresponding to Bob's projectors here, the same as before
				for y=1:n_Y
					try
						coeffs = [(monomials[Monomial(alice_part, bob_part[1:m] * [Projector(y,b)] * bob_part[m+1:end])]..., 1.0) for b=1:n_B]
						if(length(Set(coeffs)) != n_B || (monomials[monomial]..., 1.0) in coeffs) 
							continue
						end
						push!(model.constraints_eq, Constraint([coeffs; (monomials[monomial]..., -1.0)], 0.0))
					catch
					end
				end
			end
		end
		compile_constraints!(model)
		
		correlation_components = Dict()
		for x=1:n_X
			for y=1:n_Y
				for a=1:n_A
					for b=1:n_B
						correlation_components[(x,y,a,b)] = monomials[Monomial([Projector(x,a)],[Projector(y,b)])]
					end
				end
			end
		end
		new(n_X, n_Y, n_A, n_B, level, correlation_components, model)
	end
end			


	


# function UB_pire_cas(n_X, n_Y, n_A, n_B, f, promesse, niveau)
	# info = NPA_Info_general(n_X, n_Y, n_A, n_B, niveau, full)
	# info.model.N += 1 # we take X[N+1, N+1] to be maximum probability of error over all inputs
	# push!(info.model.objective, (info.model.N, info.model.N, -1.0))
	# for x=1:n_X
		# for y=1:n_Y
			# if(promesse(x,y))
				# coeffs = [(info.polynomials[XY_Polynomial([Monomial(x,a)],[Monomial(y,b)])]..., f(x,y,a,b)) for a=1:n_A for b=1:n_B]
				# push!(info.model.constraints_nonneg, Constraint([coeffs; (info.model.N, info.model.N, -1.0)], 0.0))
			# end
		# end
	# end
	# compile_constraints!(info.model)
	# compile_pseudo_objective!(info.model)
	# y = find_dual_solution(info.model, Inf, 8000)
	# return -dot(info.model.b, y)
# end

# # Given the game G, updates G.model's objective value
# function build_objective!(G::Game, info::NPA_Info_general, distribution; offset = 1e-05)
	# @assert G.n_X == info.n_X
	# @assert G.n_Y == info.n_X
	# @assert G.n_A == info.n_A
	# @assert G.n_B == info.n_B
	# empty!(info.model.objective)
	# for x=1:G.n_X
		# for y=1:G.n_Y
			# for a=1:G.n_A
				# for b=1:G.n_B
					# if(!((x,y,a,b) in G.R))
						# push!(info.model.objective, (info.polynomials[XY_Polynomial([Monomial(x,a)],[Monomial(y,b)])]..., distribution[x,y]))
					# end
				# end
			# end
		# end
	# end
	# compile_pseudo_objective!(info.model; offset = offset)
# end	

# function build_objective!(G::Game, info::NPA_Info_general, distribution; offset = 1e-05)
	# @assert G.n_X == info.n_X
	# @assert G.n_Y == info.n_X
	# @assert G.n_A == info.n_A
	# @assert G.n_B == info.n_B
	# empty!(info.model.objective)
	# for x=1:G.n_X
		# for y=1:G.n_Y
			# for a=1:G.n_A
				# for b=1:G.n_B
					# if(!((x,y,a,b) in G.R))
						# push!(info.model.objective, (info.polynomials[XY_Polynomial([Monomial(x,a)],[Monomial(y,b)])]..., distribution[x,y]))
					# end
				# end
			# end
		# end
	# end
	# compile_pseudo_objective!(info.model; offset = offset)
# end					



# function check_games(inputfile, outputfile, info_)
	# infos = [deepcopy(info_) for i=1:Threads.nthreads()]
	# BLAS.set_num_threads(1)
	# out = open(outputfile, "w")
	# u = Threads.SpinLock();
	# v = Threads.SpinLock();
	# counter = 0;
	
	# A_t, b = compile_constraints(info_.model)
	# @Threads.threads for line in readlines(inputfile)
		# info = infos[Threads.threadid()]
		# G = decode_game(line, info.n_X, info.n_Y, info.n_A, info.n_B)
		# build_objective!(G, info)
		# P = compile_pseudo_objective(info.model)
		# y = find_dual_solution(P, A_t, b, info.model.N, 0, 0.01, 100)
		# if(validate_dual_solution(info.model, y) < 1e-03)
			# lock(u) do
				# write(out, line * "\n")
			# end
		# end
		
		# lock(v) do
			# counter+=1;
			# if(counter & (2^14-1) == 0)
				# println(counter)
			# end
		# end
	# end
	# close(out)
# end


# function CC_lower_bound(X, Y, C, f, level, filtering)
	# info = NPA_Info_general(X, Y*C, C, 2, level, filtering)
	# info.model.N += 1 # we take X[N+1, N+1] to be maximum probability of error over all inputs
	# push!(info.model.objective, (info.model.N, info.model.N, 1.0))
	# for x=1:X
		# for y=1:Y
			# if(f[(x,y)] != 2)
				# b = 1 - f[(x,y)]
				# coeffs = [(info.polynomials[XY_Polynomial([Monomial(x,c)],[Monomial((y-1)*C+c,b+1)])]..., -1.0) for c=1:C]
				# push!(info.model.constraints_nonneg, Constraint([coeffs; (info.model.N, info.model.N, 1.0)], 0.0))
			# end
		# end
	# end
	# compile_constraints!(info.model)
	# compile_pseudo_objective!(info.model)
	# y = find_dual_solution(info.model, Inf, 8000)
	# return 1 - dot(info.model.b, y), y, info.polynomials, info.model.constraints_eq, info.model.constraints_nonneg
# end

# function CC_lower_bound_dist(X, Y, C, f, level, filtering, dist)
	# info = NPA_Info_general(X, Y*C, C, 2, level, filtering)
	# for x=1:X
		# for y=1:Y
			# if(f[(x,y)] != 2)
				# b = 1 - f[(x,y)]
				# coeffs = [info.polynomials[XY_Polynomial([Monomial(x,c)],[Monomial((y-1)*C+c,b+1)])] for c=1:C]
				# for coeff in coeffs
					# push!(info.model.objective, (coeff..., dist[(x,y)]))
				# end
			# end
		# end
	# end
	# compile_constraints!(info.model)
	# compile_pseudo_objective!(info.model)
	# y = find_dual_solution(info.model, Inf, 1000)
	# return 1 - dot(info.model.b, y), y, info
# end

# function gamma_2(M)
	# model = JuMP.Model(COSMO.Optimizer);
	
	# m,n = size(M);
	
	# @variable(model, Z[1:(n+m), 1:(n+m)], PSD)
	# @variable(model, c)
	# @objective(model, Min, c)
	
	# for i=1:n+m
		# @constraint(model, c >= Z[i,i])
	# end

	# for i=1:m
		# for j=1:n
			# @constraint(model, Z[i, m+j] == M[i,j])
		# end
	# end
	
	# @suppress status = JuMP.optimize!(model)
	# return JuMP.objective_value(model)
# end

# function gamma_2_alpha(R, alpha)
	# model = JuMP.Model(COSMO.Optimizer);
	
	# m,n = size(R);
	
	# @variable(model, M[1:m, 1:n])
	# @variable(model, Z[1:(n+m), 1:(n+m)], PSD)
	# @variable(model, c)
	# @objective(model, Min, c)
	
	# for i=1:n+m
		# @constraint(model, c >= Z[i,i])
	# end

	# for i=1:m
		# for j=1:n
			# @constraint(model, Z[i, m+j] == M[i,j])
			# @constraint(model, 1 <= R[i,j] * M[i,j])
			# @constraint(model, R[i,j] * M[i,j] <= alpha)
		# end
	# end
	
	# @suppress status = JuMP.optimize!(model)
	# return JuMP.objective_value(model)
# end

# function norme_tr(A)
	# return sum(svdvals(A))
# end

# function test_norme_trace_1(A, epsilon)
	# min_val = 1000
	# n = size(A)[1]
	# for i=1:1000000
		# P = rand(n, n) * 2 * epsilon .- epsilon
		# min_val = min(min_val, norme_tr(A+P))
	# end
	# return min_val
# end

# function test_norme_trace_2(A, epsilon)
	# min_val = 1000
	# alpha = 1 / (1-2*epsilon)
	# n = size(A)[1]
	# for i=1:1000000
		# B = zeros(n,n)
		# for i=1:n
			# for j=1:n
				# B[i,j] = (rand() * (alpha-1) + 1)/A[i,j]
			# end
		# end
		# min_val = min(min_val, norme_tr(B))
	# end
	# return min_val
# end
	