export NPASynchronous
import Base.reverse

# A NPA hierarchy specialized to synchronous correlations, i.e. the input sets are the same, the output sets are the same and the outputs are the same when the inputs are the same
# Of interest most specifically for coloring games
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
		push!(model.constraints_eq, Constraint([(1, 1, 1.0)], 1.0)) # Normalization constraint

		monomials::Dict{Vector{Projector}, Tuple{Int,Int}} = Dict()
		correlation_components::Dict{Tuple{Int,Int,Int,Int}, Tuple{Int,Int}} = Dict()
		
		### Enforcing zero constraints as well as forcing equivalent monomials to have equal values
		known_zero_value = (-1,-1);
		for i=1:N
			for j=i:N
				try
					monomial = eta(Base.reverse(atomic_monomials[i]) * atomic_monomials[j])
					
					if(haskey(monomials, monomial))
						i2, j2 = monomials[monomial]
						# This is a constraint of the form tau(p1 p2) - tau(q1 q2) = 0, where p1p2 ~ q1q2 
						push!(model.constraints_eq, Constraint([(i, j, 1.0), (i2, j2, -1.0)], 0.0)) 
					else
						monomials[monomial] = (i,j)
					end
						
				catch ZeroException
					push!(model.constraints_eq, Constraint([(i,j,1.0)], 0.0)) # monomial was identically zero	
					known_zero_value = (i,j)
				end	
			end
		end
		for x=1:n_X
			for y=1:n_X
				for a=1:n_A
					for b=1:n_A
						if(x == y)
							if(a == b)
								correlation_components[(x,y,a,b)] = monomials[[Projector(x,a)]]
							else
								correlation_components[(x,y,a,b)] = known_zero_value
							end
						else
							correlation_components[(x,y,a,b)] = monomials[eta([Projector(x,a), Projector(y,b)])]
						end
					end
				end
			end
		end
		
		
		### Adding some POVM constraints
		for monomial in keys(monomials)
			if(length(monomial) < 2*level)
				for v=1:n_X 
					# We will attempt to add the constraint tau(E^v_1 monomial) + tau(E^v_2 monomial) + ... - tau(monomial) = 0
					try
						# We set a coefficient of 1 to every poly of the form "(E^v_c) monomial" for every c
						coeffs = [(monomials[eta([Monomial(v,c)] * monomial)]..., 1.0) for c=1:n_A]
						if(length(Set(coeffs)) != n_A) # Ensure that there are no repetitions
							continue
						end
						push!(model.constraints_eq, Constraint([coeffs; (monomials[monomial]..., -1.0)], 0.0)) # We set a coefficient of -1 to monomial
					catch 
					end
				end
			end
		end
		compile_constraints!(model)
		new(n_X, n_A, correlation_components, model)
	end
	
end		


	