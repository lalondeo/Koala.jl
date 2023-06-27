export upper_bound_dist, adapt_to_worst_case, upper_bound_worst_case 


"""
	function upper_bound_dist(game::Problems.Game, distribution::Matrix{Float64}, info::T) where T <: NPA

Given a game, a distribution on the inputs and a NPA object, calculates an upper bound on the commuting operators value of the game. """
function upper_bound_dist(game::Problems.Game, distribution::Matrix{Float64}, info::T; offset = 1e-05, target = Inf, kwargs...) where T <: NPA
	empty!(info.model.objective)
	for x=1:game.n_X	
		for y=1:game.n_Y
			for a=1:game.n_A
				for b=1:game.n_B
					if(game.R[x,y,a,b])
						push!(info.model.objective, (info.correlation_components[(x,y,a,b)]..., -distribution[x,y]))
					end
				end
			end
		end
	end
	
	compile_pseudo_objective!(info.model; offset = offset)	
	y = find_dual_solution(info.model; target_val = (target == Inf ? Inf : -target), kwargs...)
	if(validate_dual_solution(info.model, y) < 1e-03)
		return -dot(info.model.b, y)
	else
		return Inf
	end
end





"""
	upper_bound_worst_case(problem::Problems.OneWayCommunicationProblem, info::T) where T <: NPA

Given a communication problem and a NPA object, calculates an upper bound on the commuting operators value of the game in the worst case.
"""
function upper_bound_worst_case(problem::Problems.OneWayCommunicationProblem, info::T; offset = 1e-05, target = Inf, kwargs...) where T <: NPA
	info.model.N += 1
	info.model.objective = [(info.model.N, info.model.N, -1.0)]
	compile_pseudo_objective!(info.model; offset = 1e-05)		

	n_nonneg_constraints = length(info.model.constraints_nonneg)
	for x=1:problem.n_X
		y = 1
		for _y=1:problem.n_Y
			if(problem.promise[x,_y])
				constraint = [(info.model.N, info.model.N, -1.0)]
				for c=1:problem.C
					push!(constraint, (info.correlation_components[(x,y,c,problem.f[x,_y] ? 2 : 1)]..., 1.0))
					y += 1
				end
				
				push!(info.model.constraints_nonneg, Constraint(constraint, 0.0))
			else
				y += problem.C
			end
		end
	end	
	compile_constraints!(info.model)
	y = find_dual_solution(info.model; target_val = (target == Inf ? Inf : -target), kwargs...)
	res = Inf
	if(validate_dual_solution(info.model, y) < 1e-03)
		res = -dot(info.model.b, y)
	end
	
	info.model.constraints_nonneg = info.model.constraints_nonneg[1:n_nonneg_constraints]
	info.model.N -= 1
	info.model.objective = []
	
	return res
end
