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
	function NPAGeneral(game::Problems.Game, level::Int64; filtering::Filtering = full, impose_maximally_entangled = false)
		NPAGeneral(game.n_X, game.n_Y, game.n_A, game.n_B, level; filtering = filtering, impose_maximally_entangled = impose_maximally_entangled)
	end
	
	function NPAGeneral(problem::Problems.OneWayCommunicationProblem, level::Int; filtering::Filtering = full, impose_maximally_entangled = false)
		NPAGeneral(problem.n_X, problem.n_Y * problem.C, problem.C, 2, level; filtering = filtering, impose_maximally_entangled = impose_maximally_entangled)
	end
end			

	