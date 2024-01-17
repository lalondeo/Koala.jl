export upper_bound_game, upper_bound_CC 


"""
	function upper_bound_game(game::Game, distribution::Matrix{Float64}, info::T) where T <: NPA

Given a game, a distribution on the inputs and a NPA object, calculates an upper bound on the commuting operators value of the game based on the NPA hierarchy info. If target is set to something else than Inf, 
we instead attempt to prove that target is indeed an upper bound on the value of the game. This will generally run faster. The kwargs are passed to the semidefinite solver. """
function upper_bound_game(game::Game, distribution::Matrix{Float64}, info::T; formal = false, offset = 1e-05, target = Inf, kwargs...) where T <: NPA
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
	
	compile_objective!(info.model; offset = offset)	
	y = optimize_dual(info.model; target_val = (target == Inf ? Inf : -target), kwargs...)
	return -validate_dual_solution(info.model, y; formal = formal)
end

"""
	upper_bound_CC(problem::OneWayCommunicationProblem, info::T; offset = 1e-05, target = Inf, kwargs...) where T <: NPA

Given a communication problem and a NPA object, calculates an upper bound on the commuting operators value of the game in the worst case. Inf will be returned in case something went wrong
and the solver's solution is infeasible. If target is set to a non-infinity value, we will instead attempt to prove that target is indeed an upper bound on the commuting operators value of the game, 
return target if this worked and Inf otherwise. offset is a constant offset added to the objective function for numerical purposes. The other keyword arguments will be 
passed on to COSMO. Parameters that may be of interest are verbose, for having a readout of the solution process, max_iter, for bounding the number of iterations that the solver 
may perform, and eps_abs, which controls the accuracy with which the upper bound is approximated. 
"""
function upper_bound_CC(problem::OneWayCommunicationProblem, info::NPAGeneral; offset = 1e-05, target = Inf, kwargs...)::Float64
	info.model.N += 1
	info.model.objective = [(info.model.N, info.model.N, -1.0)]
	compile_pseudo_objective!(info.model; offset = offset)		

	n_nonneg_constraints = length(info.model.constraints_nonneg)
	for x=1:problem.n_X
		y = 1
		for _y=1:problem.n_Y
			if(problem.promise[x,_y])
				constraint = [(info.model.N, info.model.N, -1.0)]
				for c=1:problem.C					
					push!(constraint, (info.correlation_components[(x,y,c,problem.f[x,_y])]..., 1.0))
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
	println(validate_dual_solution(info.model, y))
	if(validate_dual_solution(info.model, y) > -2)
		res = -dot(info.model.b, y)
	end
	
	info.model.constraints_nonneg = info.model.constraints_nonneg[1:n_nonneg_constraints]
	info.model.N -= 1
	info.model.objective = []
	
	return res
end
