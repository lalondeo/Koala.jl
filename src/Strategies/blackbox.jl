using JuMP
using LinearAlgebra
using RandomMatrices
import Base.copyto!



abstract type StrategyType end
abstract type InternalSolverDataType end

export evaluate_success_probabilities!, evaluate_success_probabilities, evaluate_success_probability, optimize_strategy!


######################### Black-box algorithm for local search #########################



"""
	evaluate_success_probabilities!(problem::P, strategy::S, success_probabilities::Matrix{Float64}) where P <: Problems.ProblemType where S <: StrategyType
	
Given a problem, a strategy for said problem and a matrix of success probabilities, fills the matrix so that the entry (x,y) equals the success probability on inputs x and y. """
function evaluate_success_probabilities!(problem::P, strategy::S, success_probabilities::Matrix{Float64}) where P <: Problems.ProblemType where S <: StrategyType
	error("evaluate_success_probabilities! not implemented")
end

"""
	evaluate_success_probabilities(problem::P, strategy::S)::Matrix{Float64} where P <: Problems.ProblemType where S <: StrategyType

Given a problem and a strategy for said problem, returns a n_X x n_Y matrix containing the success probabilities for every input pair. """
function evaluate_success_probabilities(problem::P, strategy::S)::Matrix{Float64} where P <: Problems.ProblemType where S <: StrategyType
	success_probabilities = zeros(problem.n_X, problem.n_Y)
	evaluate_success_probabilities!(problem, strategy, success_probabilities)
	return success_probabilities
end


"""
	copyto!(strat1::S, strat2::S) where S <: StrategyType

Copies strat2 into strat1. """
function copyto!(strat1::S, strat2::S) where S <: StrategyType
	error("copyto! not implemented")
end

"""
	evaluate_success_probability(problem::P, strategy::S, distribution::Matrix{Float64})::Float64 where P <: Problems.ProblemType where S <: StrategyType

Given a problem, a strategy for said problem and a distribution on the inputs, returns the average success probability under the distribution.
"""
function evaluate_success_probability(problem::P, strategy::S, distribution::Matrix{Float64})::Float64 where P <: Problems.ProblemType where S <: StrategyType
	@warn "evaluate_success_probability not overriden: forced to compute it using evaluate_success_probabilities, which corresponds to an unnecessary allocation. "
	return dot(distribution, evaluate_success_probabilities(problem, strategy))
end

"""
	scramble_strategy!(strategy::S, problem::P) where S <: StrategyType where P <: Problems.ProblemType

Given a strategy for a given problem, fiddles with the strategy. This is used in the context of a local search to reboot the search.  """
function scramble_strategy!(strategy::S, problem::P) where S <: StrategyType where P <: Problems.ProblemType
	error("scramble_strategy! not implemented")
end


"""
	improve_strategy!(problem::P, strategy::S, distribution::Matrix{Float64}, data::I) where P <: Problems.ProblemType where S <: StrategyType where I <: InternalSolverDataType

Give a problem, a strategy for said problem, a distribution on the inputs and a solver data object, attemps to replace the strategy with a better one. """
function improve_strategy!(problem::P, strategy::S, distribution::Matrix{Float64}, data::I) where P <: Problems.ProblemType where S <: StrategyType where I <: InternalSolverDataType
	error("improve_strategy! not implemented")
end


function optimize_strategy!(problem::P, strategy::S, test_strategy::S, distribution::Matrix{Float64}, solver_data::I; 
				max_iter::Int=50, epsilon::Float64 = 1e-04, stop_at_local_maximum::Bool = false) where P <: Problems.ProblemType where S <: StrategyType where I <: InternalSolverDataType
				
	old_value = 0
	best_value = 0

	copyto!(test_strategy, strategy)
	copy_strategy = false # Controls whether the strategy in test_strategy is the best strategy known so far and should be copied to strategy in the event that the iteration limit were to be hit
	for i=1:max_iter
		copy_strategy = false
		improve_strategy!(problem, test_strategy, distribution, solver_data)
		value = evaluate_success_probability(problem, test_strategy, distribution)
		if(abs(value - old_value) < epsilon || value > 1 - epsilon)
			
			if(value > best_value)
				best_value = value
				copyto!(strategy, test_strategy)

			end
			
			if(stop_at_local_maximum || value > 1 - epsilon)
				break
			end
			old_value = 0
			scramble_strategy!(test_strategy, problem)

		else
			if(value > best_value)
				copy_strategy = true
			end
	
			old_value = value
		end
	end
	
	if(copy_strategy)
		copyto!(strategy, test_strategy)
	end
end

"""
optimize_strategy!(problem::P, strategy::S, distribution::Matrix{Float64}, solver_data::I
				max_iter::Int=50, epsilon::Float64 = 1e-04, stop_at_local_maximum::Bool = false) where P <: Problem where S <: Strategy where I <: InternalSolverData
				
	Given a problem, an initial strategy for said problem, a distribution on the inputs and a InternalSolverData object solver\\_data, attempts to repeatedly improve the strategy by means of 
	improve_strategy! until a local maximum is reached to within an accuracy of epsilon. When this happens, the solver either stops or scrambles the strategy and keeps on trying to improve it, 
	depending on the value of stop_at_local_maximum. At the end of the process, the contents of strategy is the best-scoring strategy found so far.
"""
function optimize_strategy!(problem::P, strategy::S, distribution::Matrix{Float64}, solver_data::I; 
				max_iter::Int=50, epsilon::Float64 = 1e-04, stop_at_local_maximum::Bool = false) where P <: Problems.ProblemType where S <: StrategyType where I <: InternalSolverDataType
	return optimize_strategy!(problem, strategy, deepcopy(strategy), distribution, solver_data; max_iter=max_iter, epsilon = epsilon, stop_at_local_maximum = stop_at_local_maximum)
end


######################### Blackbox algorithm for finding a hard distribution #########################

"""
	generate_hard_distribution(N, oracle; max_iter = 1000, alpha = 0.003, M, = 10000 min_stabilization_iter = 50, trust_oracle = false, epsilon = 1e-03, time_until_suppression = 20, additional_constraints = (model, D) -> nothing)

Given a distribution size N and an oracle which, given a distribution D on N elements, returns an array of N probabilities indicating the success probabilities for every index
of whatever task is being studied, so that the average success probability under distribution D is maximal. 

Benders decomposition (i.e. the dual of column generation) is run to generate a hard distribution, i.e. one so that for the task at hand, the success probability is minimal for any strategy. A simple stabilization
scheme in L1 norm is employed to facilitate convergence, in the style of the paper 'Stabilized Column Generation' by du Merle, Villeneuve, Desrosiers and Hansen (1997). Namely, at every iteration,
a free gap of alpha between the old and the new distribution in allowed in infinity norm, any further gap being penalized with respect to M. Setting M to zero disables stabilization. 
Stabilization kicks in at iteration min\\_stabilization. We wait until applying stabilization because there's no trouble with convergence initially. trust\\_oracle corresponds to whether the oracle solves the optimization problem exactly or only approximately. If set to true, 
a halting condition will be checked (namely, if the gap between the primal and dual values is no greater than epsilon), while if not, the full max\\_iter iterations will be run. Success probabilities 
with corresponding dual variable equal to zero for time\\_until\\_suppression iterations are thrown out. Additionally, additional_constraints is a function that is given the model and D and is permitted to add constraints on the allowed distributions."""
function generate_hard_distribution(n_X::Int64, n_Y::Int64, oracle!::Function; max_iter=1000, alpha = 0.003, M = 100, min_stabilization = 50, trust_oracle = false, epsilon = 1e-03,
									time_until_suppression = 20, additional_constraints = (model, D) -> nothing, verbose = true)
	model = Model(LP_solver())
	set_silent(model)
	
	
	@variable(model, D[1:n_X, 1:n_Y]; lower_bound = 0.0)
	@constraint(model, sum(D) == 1) 
	additional_constraints(model, D)
	
	@variable(model, z >= 0) # Best winning probability in the worst case so far
	
	### Stabilization variables
	@variable(model, gap_plus[1:n_X, 1:n_Y]; lower_bound = 0.0)
	@variable(model, gap_minus[1:n_X, 1:n_Y]; lower_bound = 0.0)
	@variable(model, allowed_gap[1:n_X, 1:n_Y])
	@constraint(model, allowed_gap .<= alpha)
	@constraint(model, allowed_gap .>= -alpha)
	@objective(model, Min, z + M * sum(gap_plus) + M * sum(gap_minus))

	### Timing
	time_model = 0
	time_oracle = 0
	time_total = 0
	
	time_init = time()
	
	general_constraints = Dict() # Constraints to how long since the constraint had a nonzero reduced cost: constraints whose reduced cost has been zero for too long are deleted
	stabilizing_constraints = [] 
	current_z = 0;
	
	best_distribution = zeros(n_X, n_Y)
	current_distribution = zeros(n_X, n_Y)
	success_probabilities = zeros(n_X, n_Y)
	least_avg_success_probability = 1.0
	new_best_distribution = false

	for iter=1:max_iter
		if(verbose)
			println("---$(iter)---")
			println("TOTAL TIME: ", time() - time_init, ", TIME MODEL: ", time_model, ", TIME ORACLE: ", time_oracle)
			println("CURRENT Z: $(current_z)")
		end
		
		time_model += @elapsed optimize!(model)
		current_distribution .= JuMP.value.(D)

		current_z = objective_value(model) - M * sum(JuMP.value.(gap_plus)) - M * sum(JuMP.value.(gap_minus))
		if(current_z > 0.5 + 1e-04)
			duals = []
			for constraint in keys(general_constraints)
				push!(duals, JuMP.dual(constraint))
			end
			i_constraint = 1
			for constraint in keys(general_constraints)
				if(duals[i_constraint] < 1e-05)
					general_constraints[constraint] += 1
					if(general_constraints[constraint] > time_until_suppression)
						delete!(general_constraints, constraint)
						delete(model, constraint)
					end
				else
					general_constraints[constraint] = 0
				end
				i_constraint += 1
			end
		end
		time_oracle += @elapsed oracle!(current_distribution, success_probabilities)
		avg_success_probability = dot(success_probabilities, current_distribution)
	
		if(trust_oracle)
			if(abs(avg_success_probability - current_z) < epsilon)
				break
			end
			
			if(avg_success_probability < least_avg_success_probability)
				least_avg_success_probability = avg_success_probability
				best_distribution .= current_distribution
				new_best_distribution = true
			end
		else
			best_distribution .= current_distribution
			new_best_distribution = true
		end
		
		c = @constraint(model, z >= sum(success_probabilities[x,y] * D[x,y] for x=1:n_X for y=1:n_Y))
		general_constraints[c] = 0
		
		if(iter >= min_stabilization && new_best_distribution)
			for c in stabilizing_constraints
				delete(model, c)
			end
			stabilizing_constraints = []
			if(iter % 10 != 0)
				for c in @constraint(model, D .- best_distribution .== gap_plus - gap_minus + allowed_gap)
					push!(stabilizing_constraints, c)
				end
			end
			new_best_distribution = false
		end
		
	end
	return best_distribution
end

function generate_hard_distribution(problem::P, strategy::S, data::I; full_optimization_probability = 0.2, optimization_iter_regular = 5, 
	optimization_iter_extended = 50, solver_args...) where P <: Problems.ProblemType where S <: StrategyType where I <: InternalSolverDataType
	test_strategy = deepcopy(strategy)
	
	function oracle!(distribution::Matrix{Float64}, success_probabilities::Matrix{Float64})
		if(rand() < full_optimization_probability)
			optimize_strategy!(problem, strategy, test_strategy, distribution, data; 
				max_iter = optimization_iter_extended, stop_at_local_maximum = false)
		else
			optimize_strategy!(problem, strategy, test_strategy, distribution, data; 
				max_iter = optimization_iter_regular, stop_at_local_maximum = true)	
		end
		evaluate_success_probabilities!(problem, strategy, success_probabilities)
	end
	
	return generate_hard_distribution(problem.n_X, problem.n_Y, oracle!; solver_args...)
end


# In the case of a communication problem
function add_group_constraints_y!(model::Model, D::Matrix{VariableRef}, period::Int64)
	n_X,n_Y = size(D)
	
	for y=1:period:n_Y
		for l=1:period-1
			if(y + l <= n_Y)
				for x=1:n_X
					@constraint(model, D[x,y] == D[x,y+l])
				end
			end
		end
	end
end
	
	
	
	
					




