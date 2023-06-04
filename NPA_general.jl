using COSMO
using JuMP
import Base.*
import Base.==
import Base.isequal
import Base.length
import Base.copy
using SparseArrays
using LinearAlgebra
using Suppressor
include("utils.jl")
include("sdp.jl")

### Polynomial arithmetic ###
mutable struct XY_Polynomial
	alice_part::Array{Monomial}
	bob_part::Array{Monomial}
end


Base.copy(p::XY_Polynomial) = XY_Polynomial(copy(p.alice_part), copy(p.bob_part))

function (*)(p1::XY_Polynomial, p2::XY_Polynomial)
	return XY_Polynomial(p1.alice_part*p2.alice_part, p1.bob_part*p2.bob_part)

end

function (==)(p1::XY_Polynomial, p2::XY_Polynomial)
	return p1.alice_part == p2.alice_part && p1.bob_part == p2.bob_part
end



Base.isequal(p1::XY_Polynomial, p2::XY_Polynomial) = (p1 == p2)
Base.hash(p::XY_Polynomial) = hash(p.alice_part) + hash(p.bob_part)
Base.length(p::XY_Polynomial) = length(p.alice_part) + length(p.bob_part)

function reverse(p::XY_Polynomial)::XY_Polynomial
	p = copy(p)
	reverse!(p.alice_part)
	reverse!(p.bob_part)
	return p
end



### Actual hierarchy ###


mutable struct NPA_Info_general
	n_X::Int64
	n_Y::Int64
	n_A::Int64
	n_B::Int64
	level::Int64
	polynomials::Dict{XY_Polynomial, Tuple{Int,Int}}
	model::SDP_Model

	function NPA_Info_general(n_X::Int64, n_Y::Int64, n_A::Int64, n_B::Int64, level::Int64, filtering::Filtering = same_output; filtrage_plus = (x) -> true)
		### Building atomic polynomials from the atomic polynomials of Alice and Bob
		atomic_polynomials_alice = build_atomic_polynomials(n_X, n_A, level, filtering)
		atomic_polynomials_bob = build_atomic_polynomials(n_Y, n_B, level, filtering)
		
		atomic_polynomials::Array{XY_Polynomial} = []
		for p in atomic_polynomials_alice
			for q in atomic_polynomials_bob
				if(length(p) + length(q) > level)
					break # As both lists are sorted with respect to size, all other q's will also be such that XY_Polynomial(p, q) is too large
				end
				poly = XY_Polynomial(p,q)
				if(filtrage_plus(poly))
					push!(atomic_polynomials, poly)
				end
			end
		end
		
		N = length(atomic_polynomials)
		model = SDP_Model(N)
		push!(model.constraints_eq, Constraint([(1, 1, 1.0)], 1.0)) # Normalization constraint
		

		
		### Building polynomials and enforcing zero constraints
		polynomials::Dict{XY_Polynomial, Tuple{Int,Int}} = Dict()

		for i=1:N
			for j=i:N
				try
					p = reverse(atomic_polynomials[i]) * atomic_polynomials[j]
					if(haskey(polynomials, p))
						i2, j2 = polynomials[p]
						push!(model.constraints_eq, Constraint([(i, j, 1.0), (i2, j2, -1.0)], 0.0))
					else
						polynomials[p] = (i,j)
					end
					
				catch ZeroException
					push!(model.constraints_eq, Constraint([(i, j, 1.0)], 0.0))
				end
			end
		end
		
		for x=1:n_X
			for y=1:n_Y
				for a=1:n_A
					for b=1:n_B
						index = polynomials[XY_Polynomial([Monomial(x,a)], [Monomial(y,b)])]
						push!(model.constraints_nonneg, Constraint([(index[1], index[2], 1.0)], 0.0))
					end
				end
			end
		end
		
		### Adding some POVM constraints
		for polynomial in keys(polynomials)
			if(length(polynomial) < 2*level)
				alice_part = polynomial.alice_part
				bob_part = polynomial.bob_part
				
				# Adding POVM constraints corresponding to Alice's projectors here
				for x=1:n_X
					for m=0:length(alice_part)
						try
							coeffs = [(polynomials[XY_Polynomial(alice_part[1:m] * [Monomial(x,a)] * alice_part[m+1:end], bob_part)]..., 1.0) for a=1:n_A]
							if(length(Set(coeffs)) != n_A || (polynomials[polynomial]..., 1.0) in coeffs) 
								continue
							end
							push!(model.constraints_eq, Constraint([coeffs; (polynomials[polynomial]..., -1.0)], 0.0))
						catch # Either we hit a KeyError or a ZeroException. In either case, the constraint can't be added and we move on.
						end
					end
				end
				
				# Adding POVM constraints corresponding to Bob's projectors here
				for y=1:n_Y
					for m=0:length(bob_part)
						try
							coeffs = [(polynomials[XY_Polynomial(alice_part, bob_part[1:m] * [Monomial(y,b)] * bob_part[m+1:end])]..., 1.0) for b=1:n_B]
							if(length(Set(coeffs)) != n_B || (polynomials[polynomial]..., 1.0) in coeffs) 
								continue
							end
							push!(model.constraints_eq, Constraint([coeffs; (polynomials[polynomial]..., -1.0)], 0.0))
						catch
						end
					end
				end
			end
		end
		compile_constraints!(model)
		new(n_X, n_Y, n_A, n_B, level, polynomials, model)
	end
end			


function UB_pire_cas(n_X, n_Y, n_A, n_B, f, promesse, niveau)
	info = NPA_Info_general(n_X, n_Y, n_A, n_B, niveau, full)
	info.model.N += 1 # we take X[N+1, N+1] to be maximum probability of error over all inputs
	push!(info.model.objective, (info.model.N, info.model.N, -1.0))
	for x=1:n_X
		for y=1:n_Y
			if(promesse(x,y))
				coeffs = [(info.polynomials[XY_Polynomial([Monomial(x,a)],[Monomial(y,b)])]..., f(x,y,a,b)) for a=1:n_A for b=1:n_B]
				push!(info.model.constraints_nonneg, Constraint([coeffs; (info.model.N, info.model.N, -1.0)], 0.0))
			end
		end
	end
	compile_constraints!(info.model)
	compile_pseudo_objective!(info.model)
	y = find_dual_solution(info.model, Inf, 8000)
	return -dot(info.model.b, y)
end

# Given the game G, updates G.model's objective value
function build_objective!(G::Game, info::NPA_Info_general, distribution; offset = 1e-05)
	@assert G.n_X == info.n_X
	@assert G.n_Y == info.n_X
	@assert G.n_A == info.n_A
	@assert G.n_B == info.n_B
	empty!(info.model.objective)
	for x=1:G.n_X
		for y=1:G.n_Y
			for a=1:G.n_A
				for b=1:G.n_B
					if(!((x,y,a,b) in G.R))
						push!(info.model.objective, (info.polynomials[XY_Polynomial([Monomial(x,a)],[Monomial(y,b)])]..., distribution[x,y]))
					end
				end
			end
		end
	end
	compile_pseudo_objective!(info.model; offset = offset)
end	

function build_objective!(G::Game, info::NPA_Info_general, distribution; offset = 1e-05)
	@assert G.n_X == info.n_X
	@assert G.n_Y == info.n_X
	@assert G.n_A == info.n_A
	@assert G.n_B == info.n_B
	empty!(info.model.objective)
	for x=1:G.n_X
		for y=1:G.n_Y
			for a=1:G.n_A
				for b=1:G.n_B
					if(!((x,y,a,b) in G.R))
						push!(info.model.objective, (info.polynomials[XY_Polynomial([Monomial(x,a)],[Monomial(y,b)])]..., distribution[x,y]))
					end
				end
			end
		end
	end
	compile_pseudo_objective!(info.model; offset = offset)
end					



function check_games(inputfile, outputfile, info_)
	infos = [deepcopy(info_) for i=1:Threads.nthreads()]
	BLAS.set_num_threads(1)
	out = open(outputfile, "w")
	u = Threads.SpinLock();
	v = Threads.SpinLock();
	counter = 0;
	
	A_t, b = compile_constraints(info_.model)
	@Threads.threads for line in readlines(inputfile)
		info = infos[Threads.threadid()]
		G = decode_game(line, info.n_X, info.n_Y, info.n_A, info.n_B)
		build_objective!(G, info)
		P = compile_pseudo_objective(info.model)
		y = find_dual_solution(P, A_t, b, info.model.N, 0, 0.01, 100)
		if(validate_dual_solution(info.model, y) < 1e-03)
			lock(u) do
				write(out, line * "\n")
			end
		end
		
		lock(v) do
			counter+=1;
			if(counter & (2^14-1) == 0)
				println(counter)
			end
		end
	end
	close(out)
end


function CC_lower_bound(X, Y, C, f, level, filtering)
	info = NPA_Info_general(X, Y*C, C, 2, level, filtering)
	info.model.N += 1 # we take X[N+1, N+1] to be maximum probability of error over all inputs
	push!(info.model.objective, (info.model.N, info.model.N, 1.0))
	for x=1:X
		for y=1:Y
			if(f[(x,y)] != 2)
				b = 1 - f[(x,y)]
				coeffs = [(info.polynomials[XY_Polynomial([Monomial(x,c)],[Monomial((y-1)*C+c,b+1)])]..., -1.0) for c=1:C]
				push!(info.model.constraints_nonneg, Constraint([coeffs; (info.model.N, info.model.N, 1.0)], 0.0))
			end
		end
	end
	compile_constraints!(info.model)
	compile_pseudo_objective!(info.model)
	y = find_dual_solution(info.model, Inf, 8000)
	return 1 - dot(info.model.b, y), y, info.polynomials, info.model.constraints_eq, info.model.constraints_nonneg
end

function CC_lower_bound_dist(X, Y, C, f, level, filtering, dist)
	info = NPA_Info_general(X, Y*C, C, 2, level, filtering)
	for x=1:X
		for y=1:Y
			if(f[(x,y)] != 2)
				b = 1 - f[(x,y)]
				coeffs = [info.polynomials[XY_Polynomial([Monomial(x,c)],[Monomial((y-1)*C+c,b+1)])] for c=1:C]
				for coeff in coeffs
					push!(info.model.objective, (coeff..., dist[(x,y)]))
				end
			end
		end
	end
	compile_constraints!(info.model)
	compile_pseudo_objective!(info.model)
	y = find_dual_solution(info.model, Inf, 1000)
	return 1 - dot(info.model.b, y), y, info
end

function gamma_2(M)
	model = JuMP.Model(COSMO.Optimizer);
	
	m,n = size(M);
	
	@variable(model, Z[1:(n+m), 1:(n+m)], PSD)
	@variable(model, c)
	@objective(model, Min, c)
	
	for i=1:n+m
		@constraint(model, c >= Z[i,i])
	end

	for i=1:m
		for j=1:n
			@constraint(model, Z[i, m+j] == M[i,j])
		end
	end
	
	@suppress status = JuMP.optimize!(model)
	return JuMP.objective_value(model)
end

function gamma_2_alpha(R, alpha)
	model = JuMP.Model(COSMO.Optimizer);
	
	m,n = size(R);
	
	@variable(model, M[1:m, 1:n])
	@variable(model, Z[1:(n+m), 1:(n+m)], PSD)
	@variable(model, c)
	@objective(model, Min, c)
	
	for i=1:n+m
		@constraint(model, c >= Z[i,i])
	end

	for i=1:m
		for j=1:n
			@constraint(model, Z[i, m+j] == M[i,j])
			@constraint(model, 1 <= R[i,j] * M[i,j])
			@constraint(model, R[i,j] * M[i,j] <= alpha)
		end
	end
	
	@suppress status = JuMP.optimize!(model)
	return JuMP.objective_value(model)
end

function norme_tr(A)
	return sum(svdvals(A))
end

function test_norme_trace_1(A, epsilon)
	min_val = 1000
	n = size(A)[1]
	for i=1:1000000
		P = rand(n, n) * 2 * epsilon .- epsilon
		min_val = min(min_val, norme_tr(A+P))
	end
	return min_val
end

function test_norme_trace_2(A, epsilon)
	min_val = 1000
	alpha = 1 / (1-2*epsilon)
	n = size(A)[1]
	for i=1:1000000
		B = zeros(n,n)
		for i=1:n
			for j=1:n
				B[i,j] = (rand() * (alpha-1) + 1)/A[i,j]
			end
		end
		min_val = min(min_val, norme_tr(B))
	end
	return min_val
end
	