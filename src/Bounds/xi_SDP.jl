export xi_SDP
"""
   xi_SDP(G::Matrix{Bool}; target_val::Int = -1, formal::Bool = false, kwargs...)::Int64

Given a graph G, returns a lower bound on the ceil of the value of xi_SDP, as defined in Paulsen, Severini, Stahlke, Todorov and Winter (2014). If nothing is given as a target, 
the actual value is computed, while if a target is given, we attempt to show that the target given is indeed a lower bound. formal specifies whether we should prove in exact arithmetic
that the returned value is indeed a correct lower bound. The other kwargs get passed to the SDP solver.
"""
function xi_SDP(G::Matrix{Bool}; target_val::Int64 = -1, formal::Bool = false, kwargs...)#::Int64

	cliques = list_maximal_cliques(G)
	# The clique number of G lower bounds xi_SDP, so target_val is indeed a lower bound on \ceil{xi_SDP} if G contains a clique of that size
	if(target_val != -1 && maximum((x)->length(x), cliques) >= target_val)
		return target_val
	end
	
	n = size(G)[1]
	model = SDP_Model(n+1)
	push!(model.objective, (1, 1, 1.0))

	for i=1:n
		for j=i+1:n
			if(G[i,j])
				push!(model.constraints_eq, Constraint([(i+1, j+1, 1.0)], 0.0))
			else
				push!(model.constraints_nonneg, Constraint([(i+1, j+1, 1.0)], 0.0))
			end
		end
	end
	
	for i=2:n+1
		push!(model.constraints_eq, Constraint([(i, i, 1.0)], 1.0))
		push!(model.constraints_eq, Constraint([(i, 1, 1.0)], 1.0))
	end
	
	for clique in cliques
		for w=2:n+1
			coeffs = []
			for v in clique
				push!(coeffs, (w, v+1, -1.0))
			end
			push!(model.constraints_nonneg, Constraint(coeffs, -1.0))
		end
	end
	
	for i=1:length(cliques)
		for j=i+1:length(cliques)
			coeffs = [(1, 1, 1.0)]
			for v1 in cliques[i]
				for v2 in cliques[j]
					push!(coeffs, (v1+1, v2+1, 1.0))
				end
			end
			push!(model.constraints_nonneg, Constraint(coeffs, Float64(length(cliques[i]) + length(cliques[j]))))
		end
	end
	
	compile_constraints!(model)
	compile_objective!(model)
	z = validate_dual_solution(model, optimize_dual(model; target_val = (target_val == -1 ? Inf : target_val - 0.999), kwargs...); formal = formal)
	if(z == -Inf)
		return -1
	else
		return Int(ceil(z - 1e-08))
	end
end
