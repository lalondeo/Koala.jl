using COSMO
using JuMP
import Base.*
import Base.==
import Base.isequal
import Base.length
import Base.copy
using SparseArrays
using LinearAlgebra
using Combinatorics
export NPAGeneral
import IterTools

struct ClassicalMonomial
	alice_part::Vector{Int}
	bob_part::Vector{Int}
end

function (*)(p1::Vector{Int}, p2::Vector{Int})::Union{Vector{Int},Nothing}
	n = length(p1)
	@assert n == length(p2)
	res = zeros(Int, n)
	for i=1:n
		if(p1[i] != 0)
			if(p2[i] != 0 && p1[i] != p2[i])
				return nothing
			else
				res[i] = p1[i]
			end
		else	
			res[i] = p2[i]
		end
	end
	return res
end
	

function (*)(m1::ClassicalMonomial, m2::ClassicalMonomial)::Union{ClassicalMonomial, Nothing}
	
	prod1 = m1.alice_part*m2.alice_part
	if(prod1 == nothing)
		return nothing
	end
	
	prod2 = m1.bob_part*m2.bob_part
	if(prod2 == nothing)
		return nothing
	end
	return ClassicalMonomial(prod1, prod2)

end

function (==)(m1::ClassicalMonomial, m2::ClassicalMonomial)
	return (m1.alice_part == m2.alice_part) && (m1.bob_part == m2.bob_part)
end

function build_classical_atomic_monomials(n_X::Int, n_A::Int, level::Int)::Vector{Vector{Int}}
	monomials::Vector{Vector{Int}} = [];
	
	for k=0:level
		for subset in combinations(1:n_X, k)
			for outputs in IterTools.product([collect(1:n_A) for i=1:k]...)
				new_monomial = zeros(Int, n_X)
				new_monomial[subset] .= outputs
				push!(monomials, new_monomial)
			end
		end
	end
	return monomials
end
				
	
Base.isequal(m1::ClassicalMonomial, m2::ClassicalMonomial) = (m1.alice_part == m2.alice_part && m1.bob_part == m2.bob_part)
Base.hash(m::ClassicalMonomial) = hash(m.alice_part) + hash(m.bob_part)
Base.length(m::ClassicalMonomial) = count((i)->i != 0, m.alice_part) + count((i)->i != 0, m.bob_part)


### Actual hierarchy ###

struct NPAClassical <: NPA
	n_X::Int64
	n_Y::Int64
	n_A::Int64
	n_B::Int64
	level::Int64
	
	correlation_components::Dict{Tuple{Int,Int,Int,Int}, Tuple{Int,Int}}
	model::SDP_Model

	function NPAClassical(n_X::Int64, n_Y::Int64, n_A::Int64, n_B::Int64, level::Int64)
		### Building atomic monomials from the atomic monomials of Alice and Bob
		atomic_monomials_alice = build_classical_atomic_monomials(n_X, n_A, level)
		atomic_monomials_bob = build_classical_atomic_monomials(n_Y, n_B, level)
		correlation_components::Dict{Tuple{Int,Int,Int,Int}, Tuple{Int,Int}} = Dict()

		atomic_monomials::Vector{ClassicalMonomial} = []
		for m1 in atomic_monomials_alice
			for m2 in atomic_monomials_bob
				if(count((i)->i != 0, m1) + count((i)->i != 0, m2) <= level)
					push!(atomic_monomials, ClassicalMonomial(m1,m2))
				end
			end
		end
		N = length(atomic_monomials)
		model = SDP_Model(N)
		push!(model.constraints_eq, Constraint([(1, 1, 1.0)], 1.0)) # Normalization constraint
		
		### Building monomials and enforcing zero constraints
		monomial_registry::Dict{ClassicalMonomial, Tuple{Int,Int}} = Dict()

		for i=1:N
			for j=i:N
				m = atomic_monomials[i] * atomic_monomials[j]

				if(m == nothing)
					push!(model.constraints_eq, Constraint([(i, j, 1.0)], 0.0))
					
				elseif(haskey(monomial_registry, m)) # Already known, enforce the equality
					i2, j2 = monomial_registry[m]
					push!(model.constraints_eq, Constraint([(i, j, 1.0), (i2, j2, -1.0)], 0.0))
					if(i == j)
						monomial_registry[m] = (i,j)
					end
					
				else
					monomial_registry[m] = (i,j)
					if(count((i)->i != 0, m.alice_part) == 1 && count((i)->i != 0, m.bob_part) == 1)
						push!(model.constraints_nonneg, Constraint([(i,j,1.0)], 0.0)) # This is a probability and therefore must be nonnegative
					end
				end
	
			end
		end
		
		monomial = ClassicalMonomial(zeros(Int, n_X), zeros(Int, n_Y))
		for x=1:n_X
			for a=1:n_A
				monomial.alice_part[x] = a
				for y=1:n_Y
					for b=1:n_B
						monomial.bob_part[y] = b
						correlation_components[(x,y,a,b)] = monomial_registry[monomial]
					end
					monomial.bob_part[y] = 0
				end
			end
			monomial.alice_part[x] = 0
		end
		
		### Adding some POVM constraints
		for monomial in keys(monomial_registry)
			if(length(monomial) < 2*level)
				alice_part = monomial.alice_part
				bob_part = monomial.bob_part
				
				###### Alice part ######
			
				for x=1:n_X
					if(alice_part[x] == 0)
						constraint_ok = true
						constraint_coeffs = [(monomial_registry[monomial]..., -1.0)]	
						for a=1:n_A
							monomial.alice_part[x] = a
							if(haskey(monomial_registry, monomial))
								push!(constraint_coeffs, (monomial_registry[monomial]..., 1.0))
							else
								constraint_ok = false
								break
							end
						end
						monomial.alice_part[x] = 0
						if(constraint_ok)
							push!(model.constraints_eq, Constraint(constraint_coeffs, 0.0))
						end
					end
				end
				
				###### Bob part ######
			
				for y=1:n_Y
					if(bob_part[y] == 0)
						constraint_ok = true
						constraint_coeffs = [(monomial_registry[monomial]..., -1.0)]	
						for b=1:n_B
							monomial.bob_part[y] = b
							if(haskey(monomial_registry, monomial))
								push!(constraint_coeffs, (monomial_registry[monomial]..., 1.0))
							else
								constraint_ok = false
								break
							end
						end
						monomial.bob_part[y] = 0
						if(constraint_ok)
							push!(model.constraints_eq, Constraint(constraint_coeffs, 0.0))
						end
					end
				end
			end
		end
		compile_constraints!(model)
		new(n_X, n_Y, n_A, n_B, level, correlation_components, model)
	end
	
	function NPAClassical(game::Game, level::Int64)
		NPAClassical(game.n_X, game.n_Y, game.n_A, game.n_B, level)
	end
	
	function NPAClassical(problem::OneWayCommunicationProblem, level::Int)
		NPAClassical(problem.n_X, problem.n_Y * problem.C, problem.C, problem.n_Z, level; filtering = filtering, impose_maximally_entangled = impose_maximally_entangled)
	end
end			

	
