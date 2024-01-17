using COSMO
using JuMP
import Base.*
import Base.==
import Base.isequal
import Base.length
import Base.copy
using SparseArrays
using LinearAlgebra
export NPAGeneral

### Monomial arithmetic ###
struct Monomial
	alice_part::Vector{Projector}
	bob_part::Vector{Projector}
end

function eta(m::Monomial)::Union{Monomial, Nothing}
	if(isempty(m.alice_part))
		res_eta = eta(m.bob_part)
		if(res_eta == nothing)
			return nothing
		else
			return Monomial([], res_eta)
		end
	elseif(isempty(m.bob_part))
		res_eta = eta(m.alice_part)
		if(res_eta == nothing)
			return nothing
		else
			return Monomial(res_eta, [])
		end
	else
		return m
	end
end


Base.copy(m::Monomial) = Monomial(copy(m.alice_part), copy(m.bob_part))

function (*)(m1::Monomial, m2::Monomial)::Union{Monomial, Nothing}
	prod1 = m1.alice_part*m2.alice_part
	if(prod1 == nothing)
		return nothing
	end
	
	prod2 = m1.bob_part*m2.bob_part
	if(prod2 == nothing)
		return nothing
	end
	return Monomial(prod1, prod2)

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

# The original NPA hierarchy of Navascues, Pironio and Acin (2008) 
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
		correlation_components::Dict{Tuple{Int,Int,Int,Int}, Tuple{Int,Int}} = Dict()

		atomic_monomials::Vector{Monomial} = []
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
		monomial_registry::Dict{Monomial, Tuple{Int,Int}} = Dict()

		for i=1:N
			for j=i:N
				m = reverse(atomic_monomials[i]) * atomic_monomials[j]
				if(impose_maximally_entangled)
					m = eta(m)
				end
				
				if(m == nothing)
					push!(model.constraints_eq, Constraint([(i, j, 1.0)], 0.0))
					
				
				elseif(haskey(monomial_registry, m)) # Already known, enforce the equality
					i2, j2 = monomial_registry[m]
					push!(model.constraints_eq, Constraint([(i, j, 1.0), (i2, j2, -1.0)], 0.0))
				else
					monomial_registry[m] = (i,j)
					if(length(m.alice_part) == 1 && length(m.bob_part) == 1)
						push!(model.constraints_nonneg, Constraint([(i,j,1.0)], 0.0)) # This is a probability and therefore must be nonnegative
					end
				end
	
			end
		end
		
		for x=1:n_X
			for y=1:n_Y
				for a=1:n_A
					for b=1:n_B
						correlation_components[(x,y,a,b)] = monomial_registry[Monomial([Projector(x,a)],[Projector(y,b)])]
					end
				end
			end
		end
		
		### Adding some POVM constraints
		for monomial in keys(monomial_registry)
			if(length(monomial) < 2*level)
				alice_part = monomial.alice_part
				bob_part = monomial.bob_part
				
				###### Alice part ######
			
				for k=0:length(alice_part)
					for x=1:n_X
						if(k == 0 && length(alice_part) > 0 && ( x == alice_part[1].i || (impose_maximally_entangled && !isempty(bob_part) && x == alice_part[end].i))) 
							continue
						end
						
						if(k > 0 && k != length(alice_part) && (x == alice_part[k].i || x == alice_part[k+1].i))
							continue
						end
						
						if(k == length(alice_part) && length(alice_part) > 0 && (x == alice_part[end].i || (impose_maximally_entangled && !isempty(bob_part)))) 
							continue
						end
						
						constraint_ok = true
						constraint_coeffs = [(monomial_registry[monomial]..., -1.0)]
						# Run through all the tau(monomial[1:k] * P^x_a monomial[k+1]:end). If one is not in the registry, we give up, and we add the constraint otherwise.
						for a=1:n_A
							_monomial = Monomial([alice_part[1:k]; [Projector(x, a)]; alice_part[k+1:end]], bob_part)
							if(impose_maximally_entangled)
								_monomial = eta(_monomial)
							end
								
							if(haskey(monomial_registry, _monomial))
								push!(constraint_coeffs, (monomial_registry[_monomial]..., 1.0))
							else
								constraint_ok = false
								break
							end
						end
						if(constraint_ok)
							push!(model.constraints_eq, Constraint(constraint_coeffs, 0.0))
						end
						
					end
				end
				
				###### Bob part ######
			
				for k=0:length(bob_part)
					for y=1:n_Y
						if(k == 0 && length(bob_part) > 0 && (y == bob_part[1].i || (impose_maximally_entangled && !isempty(alice_part) && y == bob_part[end].i))) 
							continue
						end
						
						if(k > 0 && k != length(bob_part) && (y == bob_part[k].i || y == bob_part[k+1].i))
							continue
						end
						
						if(k == length(bob_part) && length(bob_part) > 0 && (y == bob_part[end].i || (impose_maximally_entangled && !isempty(alice_part)))) 
							continue
						end
						
						constraint_ok = true
						constraint_coeffs = [(monomial_registry[monomial]..., -1.0)]
						# Run through all the tau(monomial[1:k] * P^x_a monomial[k+1]:end). If one is not in the registry, we give up, and we add the constraint otherwise.
						for b=1:n_B
							_monomial = Monomial(alice_part, [bob_part[1:k]; [Projector(y, b)]; bob_part[k+1:end]])
							if(impose_maximally_entangled)
								_monomial = eta(_monomial)
							end
								
							if(haskey(monomial_registry, _monomial))
								push!(constraint_coeffs, (monomial_registry[_monomial]..., 1.0))
							else
								constraint_ok = false
								break
							end
						end
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
	function NPAGeneral(game::Game, level::Int64; filtering::Filtering = full, impose_maximally_entangled = false)
		NPAGeneral(game.n_X, game.n_Y, game.n_A, game.n_B, level; filtering = filtering, impose_maximally_entangled = impose_maximally_entangled)
	end
	
	function NPAGeneral(problem::OneWayCommunicationProblem, level::Int; filtering::Filtering = full, impose_maximally_entangled = false)
		NPAGeneral(problem.n_X, problem.n_Y * problem.C, problem.C, problem.n_Z, level; filtering = filtering, impose_maximally_entangled = impose_maximally_entangled)
	end
end			

	
