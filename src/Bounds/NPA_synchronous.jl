export NPASynchronous
import Base.reverse

# A NPA hierarchy specialized to synchronous correlations, i.e. the input sets are the same, the output sets are the same and the outputs are equal when the inputs are
# Basically the NPA hierarchy given in "A synchronous NPA hierarchy with applications" by Russell (2021)

struct NPASynchronous <: NPA
	n_X::Int64 
	n_A::Int64 
	correlation_components::Dict{Tuple{Int,Int,Int,Int}, Tuple{Int,Int}}
	model::SDP_Model 

	function NPASynchronous(n_X::Int, n_A::Int, level::Int; filtering::Filtering = same_output)
		
		atomic_monomials = build_atomic_monomials(n_X, n_A, level, filtering)
		
		N = length(atomic_monomials)
		model = SDP_Model(N)
		push!(model.constraints_eq, Constraint([(1, 1, 1.0)], 1.0)) # Normalization constraint: the first monomial in atomic_monomials is always the empty monomial

		monomial_registry::Dict{Vector{Projector}, Tuple{Int,Int}} = Dict() # Maps known monomials to indices of M corresponding to the evaluation of that monomial
		correlation_components::Dict{Tuple{Int,Int,Int,Int}, Tuple{Int,Int}} = Dict()
		
		### Enforcing zero constraints as well as forcing equivalent monomials to have equal values ###
		known_zero_value = (-1,-1);
		foo = 0
		for i=1:N
			for j=i:N
				monomial = eta(Base.reverse(atomic_monomials[i]) * atomic_monomials[j])
				
				if(monomial == nothing) # Monomial is equal to zero, enforce this in the SDP
					push!(model.constraints_eq, Constraint([(i,j,1.0)], 0.0)) 
					known_zero_value = (i,j)
				
				elseif(haskey(monomial_registry, monomial)) # Monomial is already known, enforce the corresponding equality constraint 
					push!(model.constraints_eq, Constraint([(i, j, 1.0), (monomial_registry[monomial]..., -1.0)], 0.0)) 
					
				else # New monomial, add to the registry
					monomial_registry[monomial] = (i,j)
					if(length(monomial) <= 2)
						push!(model.constraints_nonneg, Constraint([(i,j,1.0)], 0.0)) # This is a probability and therefore must be nonnegative
					end
				end
						
			end
		end
		
		### Building correlation_components ###
		for x=1:n_X
			for y=1:n_X
				for a=1:n_A
					for b=1:n_A
						if(x == y)
							if(a == b)
								correlation_components[(x,y,a,b)] = monomial_registry[[Projector(x,a)]]
							else
								correlation_components[(x,y,a,b)] = known_zero_value
							end
						else
							correlation_components[(x,y,a,b)] = monomial_registry[eta([Projector(x,a), Projector(y,b)])]
						end
					end
				end
			end
		end
		
		### POVM constraints ###
		for monomial in keys(monomial_registry)
			if(length(monomial) < 2*level) # Otherwise, the constraint adding below will necessarily fail because _monomial will always be of size 2*level + 1
				for k=0:length(monomial)-1
					for x=1:n_X
						# We attempt to add the constraint sum_a tau(monomial[1:k] * P^x_a monomial[k+1]:end ) = tau(monomial)
						# This checks that _monomial below does not contain adjacent projectors with repeated terms
						if((k > 0 || (x != monomial[1].i && x != monomial[end].i)) && (k == 0 || (x != monomial[k].i && x != monomial[k+1].i))) 
							constraint_ok = true
							constraint_coeffs = [(monomial_registry[monomial]..., -1.0)]
							# Run through all the tau(monomial[1:k] * P^x_a monomial[k+1]:end). If one is not in the registry, we give up, and we add the constraint otherwise.
							for a=1:n_A
								_monomial = eta([monomial[1:k]; [Projector(x,a)]; monomial[k+1:end]])
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
		end
							
		compile_constraints!(model)
		new(n_X, n_A, correlation_components, model)
	end
	
	function NPASynchronous(game::Game, level::Int; filtering::Filtering = same_output)
		@assert game.n_X == game.n_Y
		@assert game.n_A == game.n_B
		NPASynchronous(game.n_X, game.n_A, level; filtering = filtering)
	end
end		


	
