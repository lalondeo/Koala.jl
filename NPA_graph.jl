include("sdp.jl")
include("utils.jl")
import Base.reverse


mutable struct NPA_Info_graph
	n::Int64 # Size of the graph
	k::Int64 # Size of the coloring
	polynomials::Dict{Array{Monomial}, Tuple{Int64, Int64}} # Map from a polynomial p to an index (i,j) of the SDP matrix M corresponding to tau(p)
	model::SDP_Model # Corresponding SDP model. 
	# Note that the constructor builds the constraints, which are instance-independent, but not the objective, which isn't. To build the objective for a given graph, 
	# use build_objective! below.
	
	function NPA_Info_graph(n::Int, k::Int, level::Int, filtering::Filtering = same_output)
		Random.seed!(1)
		
		### Building atomic polys and basic setup
		atomic_polynomials = build_atomic_polynomials(n, k, level, filtering)
		
		N = length(atomic_polynomials)
		model = SDP_Model(N)
		push!(model.constraints_eq, Constraint([(1, 1, 1.0)], 1.0)) # Normalization constraint

		polynomials::Dict{Array{Monomial}, Tuple{Int64, Int64}} = Dict()
		
		### Enforcing zero constraints as well as forcing equivalent polynomials to have equal values
		for i=1:N
			for j=i:N
				try
					polynomial = eta(reverse(atomic_polynomials[i]) * atomic_polynomials[j])
					if(haskey(polynomials, polynomial))
						i2, j2 = polynomials[polynomial]
						# This is a constraint of the form tau(p1 p2) - tau(q1 q2) = 0, where p1p2 ~ q1q2 
						push!(model.constraints_eq, Constraint([(i, j, 1.0), (i2, j2, -1.0)], 0.0)) 
					else
						
						polynomials[polynomial] = (i,j)
					end
						
				catch ZeroException
					push!(model.constraints_eq, Constraint([(i,j,1.0)], 0.0)) # polynomial was identically zero	
				end	
			end
		end
		
		### Adding some POVM constraints
		for polynomial in keys(polynomials)
			if(length(polynomial) < 2*level)
				for v=1:n 
					# We will attempt to add the constraint tau(E^v_1 polynomial) + tau(E^v_2 polynomial) + ... - tau(polynomial) = 0
					try
						# We set a coefficient of 1 to every poly of the form "(E^v_c) polynomial" for every c
						coeffs = [(polynomials[eta([Monomial(v,c)] * polynomial)]..., 1.0) for c=1:k]
						if(length(Set(coeffs)) != k) # Ensure that there are no repetitions
							continue
						end
						push!(model.constraints_eq, Constraint([coeffs; (polynomials[polynomial]..., -1.0)], 0.0)) # We set a coefficient of -1 to polynomial
					catch 
					end
				end
			end
		end
		compile_constraints!(model)
		new(n, k, polynomials, model)
	end
	
end		

# Given the graph G, updates info.model's objective
function build_objective!(G::Matrix{Bool}, info::NPA_Info_graph)
	@assert size(G)[1] == info.n
	empty!(info.model.objective)
	for v1=1:info.n
		for v2=v1+1:info.n
			if(G[v1,v2])
				for c=1:info.k
					index = info.polynomials[eta([Monomial(v1,c), Monomial(v2,c)])]
					push!(info.model.objective, (index..., 1.0))
				end
					
			end
		end
	end
	compile_pseudo_objective!(info.model)
end

	