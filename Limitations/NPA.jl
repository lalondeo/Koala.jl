function upper_bound_dist(game::Game, distribution::Matrix{Float64}, info::NPAGeneral)
	empty!(info.model.objective)
	for (x,y,a,b) in game.R
		push!(info.model.objective, (info.correlation_components[(x,y,a,b)]..., -distribution[x,y]))
	end
	
	compile_pseudo_objective!(info.model; offset = 1e-05)	
	y = find_dual_solution(info.model, Inf, 800; verbose = false)
	if(validate_dual_solution(info.model, y) < 1e-03)
		return -dot(info.model.b, y)
	else
		return Inf
	end
end

function adapt_to_worst_case(info::T) where T <: NPA
	info.model.N += 1 # Extend the size of the matrix by one: the value in the lower right corner corresponds to the minimum over all inputs of the probability of winning
	info.model.objective = [(info.model.N, info.model.N, -1.0)]
end

function upper_bound_worst_case(game::Game, info::T) where T <: NPA
	n_nonneg_constraints = length(info.model.constraints_nonneg)
	for x=1:game.n_X
		for y=1:game.n_Y
			constraint = [(info.model.N, info.model.N, -1.0)]
			for a=1:game.n_A
				for b=1:game.n_B
					if(((x,y,a,b) in game.R))
						push!(constraint, (info.correlation_components[(x,y,a,b)]..., 1.0))
					end
				end
			end
			push!(info.model.constraints_nonneg, Constraint(constraint, 0.0))
		end
	end
	compile_pseudo_objective!(info.model; offset = 1e-05)	
	compile_constraints!(info.model)
	y = find_dual_solution(info.model, Inf, 8000; verbose = true)
	res = Inf
	if(validate_dual_solution(info.model, y) < 1e-03)
		res = -dot(info.model.b, y)
	end
	
	info.model.constraints_nonneg = info.model.constraints_nonneg[1:n_nonneg_constraints]
	return res
end	

function upper_bound_worst_case(problem::OneWayCommunicationProblem, info::
	
	
