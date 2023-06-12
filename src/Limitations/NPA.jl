export upper_bound_dist, adapt_to_worst_case, upper_bound_worst_case 


"""
	function upper_bound_dist(game::Problems.Game, distribution::Matrix{Float64}, info::T) where T <: NPA

Given a game, a distribution on the inputs and a NPA object, calculates an upper bound on the commuting operators value of the game. """
function upper_bound_dist(game::Problems.Game, distribution::Matrix{Float64}, info::T; verbose = false, iter = 8000, offset = 1e-05, target = Inf) where T <: NPA
	empty!(info.model.objective)
	for (x,y,a,b) in game.R
		push!(info.model.objective, (info.correlation_components[(x,y,a,b)]..., -distribution[x,y]))
	end
	
	compile_pseudo_objective!(info.model; offset = offset)	
	y = find_dual_solution(info.model, target, iter; verbose = verbose)
	if(validate_dual_solution(info.model, y) < 1e-03)
		return -dot(info.model.b, y)
	else
		return Inf
	end
end


function adapt_to_worst_case(info::T) where T <: NPA
	info.model.N += 1 # Extend the size of the matrix by one: the value in the lower right corner corresponds to the minimum over all inputs of the probability of winning
	info.model.objective = [(info.model.N, info.model.N, -1.0)]
	compile_pseudo_objective!(info.model; offset = 1e-05)	

end

"""
	function upper_bound_worst_case(game::Problems.Game, info::T) where T <: NPA

Given a game and a NPA object, calculates an upper bound on the commuting operators value of the game in the worst case. 
adapt_to_worst_case needs to have been called on the info object beforehand. """
function upper_bound_worst_case(game::Problems.Game, info::T; verbose = false, iter = 8000, target = Inf) where T <: NPA
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
	compile_constraints!(info.model)
	y = find_dual_solution(info.model, target, iter; verbose = verbose)
	res = Inf
	if(validate_dual_solution(info.model, y) < 1e-03)
		res = -dot(info.model.b, y)
	end
	
	info.model.constraints_nonneg = info.model.constraints_nonneg[1:n_nonneg_constraints]
	return res
end	

"""
	upper_bound_worst_case(problem::Problems.OneWayCommunicationProblem, info::T) where T <: NPA

Given a communication problem and a NPA object, calculates an upper bound on the commuting operators value of the game in the worst case. 
adapt_to_worst_case needs to have been called on the info object beforehand. """
function upper_bound_worst_case(problem::Problems.OneWayCommunicationProblem, info::T; verbose = false, iter = 8000, offset = 1e-05, target = Inf) where T <: NPA
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
			end
		end
	end	
	compile_constraints!(info.model)
	y = find_dual_solution(info.model, target, iter; verbose = verbose)
	res = Inf
	if(validate_dual_solution(info.model, y) < 1e-03)
		res = -dot(info.model.b, y)
	end
	
	info.model.constraints_nonneg = info.model.constraints_nonneg[1:n_nonneg_constraints]
	return res
end
