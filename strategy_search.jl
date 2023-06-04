include("utils.jl")
using JuMP
using COSMO
using LinearAlgebra
using Random
using RandomMatrices
using CPLEX
import Base.copyto!


### Black-box algorithm for generating a hard distribution ###

"""
	generate_hard_distribution(N, legal_position, oracle; max_iter = 1000, alpha = 0.003, M, = 10000 min_stabilization_iter = 50, trust_oracle = false, epsilon = 1e-03, time_until_suppression = 20)

Given a distribution size N, a boolean vector legal_position indicating which indices of the distribution are to be forced to zero, 
and an oracle which, given a distribution D on N elements, returns an array of N probabilities indicating the success probabilities for every index
of whatever task is being studied, so that the average success probability under distribution D is maximal. 

Benders decomposition (i.e. the dual of column generation) is run for at to generate a hard distribution, i.e. one so that for the task at hand, the success probability is minimal for any strategy. A simple stabilization
scheme in L1 norm is employed to facilitate convergence, in the style of the paper 'Stabilized Column Generation' by du Merle, Villeneuve, Desrosiers and Hansen (1997). Namely, at every iteration,
a free gap of alpha between the old and the new distribution in allowed in infinity norm, any further gap being penalized with respect to M. Setting M to zero disables stabilization. 
Stabilization kicks in at iteration min\\_stabilization. We wait until applying stabilization because there's no trouble with convergence initially. trust\\_oracle corresponds to whether the oracle solves the optimization problem exactly or only approximately. If set to true, 
a halting condition will be checked (namely, if the gap between the primal and dual values is no greater than epsilon), while if not, the full max\\_iter iterations will be run. Success probabilities 
with corresponding dual variable equal to zero for time\\_until\\_suppression iterations are thrown out. """
function generate_hard_distribution(N::Int64, legal_position::Vector{Bool}, oracle; max_iter=1000, alpha = 0.003, M = 10000, min_stabilization = 50, trust_oracle = false, epsilon = 1e-03,
									time_until_suppression = 20)
    model = Model(CPLEX.Optimizer)
	set_silent(model)
	
	# Declare 
	@variable(model, D[1:N]; lower_bound = 0.0)
	@constraint(model, sum(D) == 1) 
	
	valid_index = 0
	for i=1:N
		if(legal_position[i])
			valid_index = i
		else
			@constraint(model, D[i] == 0)
		end
	end
	@assert valid_index != 0
    current_distribution = [i == valid_index ? 1.0 : 0.0 for i=1:N]
	
	@variable(model, z >= 0) # Best winning probability in the worst case so far
	
	@variable(model, gap_plus[1:N]; lower_bound = 0.0)
	@variable(model, gap_minus[1:N]; lower_bound = 0.0)
	@variable(model, allowed_gap[1:N])
	@constraint(model, allowed_gap .<= alpha)
	@constraint(model, allowed_gap .>= -alpha)
	@objective(model, Min, z + M * sum(gap_plus) + M * sum(gap_minus))


	time_model = 0
	time_oracle = 0
	general_constraints = Dict() # Constraints to how long since the constraint had a nonzero reduced cost: constraints whose reduced cost has been zero for too long are deleted
	stabilizing_constraints = [] 
	current_z = 0;
	
	best_distribution = deepcopy(current_distribution)
	least_avg_success_probability = 1.0
	new_best_distribution = false

	for iter=1:max_iter
		time_oracle += @elapsed success_probabilities = oracle(current_distribution)
		avg_success_probability = dot(success_probabilities, current_distribution)
	
		if(trust_oracle)
			if(abs(avg_success_probability - current_z) < epsilon)
				break
			end
			
			if(avg_success_probability < least_avg_success_probability)
				least_avg_success_probability = avg_success_probability
				best_distribution[:,:] = current_distribution
				new_best_distribution = true
			end
		else
			best_distribution[:,:] = current_distribution
			new_best_distribution = true
		end
		
		c = @constraint(model, z >= sum(success_probabilities[i] * D[i] for i=1:N))
		general_constraints[c] = 0
		
		if(iter >= min_stabilization && new_best_distribution)
			for c in stabilizing_constraints
				delete(model, c)
			end
			stabilizing_constraints = []
			if(iter%10 != 0)
				for c in  @constraint(model, D .- best_distribution .== gap_plus - gap_minus + allowed_gap)
					push!(stabilizing_constraints, c)
				end
			end
			new_best_distribution = false
		end

		time_model += @elapsed optimize!(model)
		current_distribution[:] .= JuMP.value.(D)

		current_z = objective_value(model) - M * sum(JuMP.value.(gap_plus)) - M * sum(JuMP.value.(gap_minus))
		println(current_z, " ", least_success_probability, " ", time_model, " ", time_oracle)
		if(current_z > 0.5 + 1e-04)
			duals = []
			for constraint in keys(general_constraints)
				push!(duals, getdual(constraint))
			end
			i_constraint = 1
			for constraint in keys(general_constraints)
				if(duals[i_constraint] < 1e-04)
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
	end
	return best_distribution
end


### Black-box algorithm for local search ###

abstract type Strategy end
abstract type InternalSolverData end

mutable struct OptimizationOutput
	value::Float64
	success_probabilities::Matrix{Float64}
	strategy::T where T <: Strategy
end

function optimize_strategy!(problem::P, distribution::Matrix{Float64}, solver_data::I, optimization_output::OptimizationOutput; 
				max_iter::Int=50, epsilon::Float64 = 1e-04, stop_at_local_maximum::Bool = false) where P <: Problem where S <: Strategy where I <: InternalSolverData
				
	old_value = 0
	best_value = 0

	test_strategy = deepcopy(optimization_output.strategy)
	copy_strategy = false # Controls whether the strategy in test_strategy is the best strategy known so far and should be copied to strategy in the event that the iteration limit were to be hit
	for i=1:max_iter
		copy_strategy = false
		improve_strategy!(problem, test_strategy, distribution, solver_data)
		value = evaluate_success_probability(problem, test_strategy, distribution)
		if(abs(value - old_value) < epsilon || value > 1 - epsilon)
			if(value > best_value)
				best_value = value
				copyto!(optimization_output.strategy, test_strategy)

			end
			
			if(stop_at_local_maximum || value > 1 - epsilon)
				break
			end
			old_value = 0
			scramble_strategy!(problem, test_strategy)

		else
			if(value > best_value)
				copy_strategy = true
			end
	
			old_value = value
		end
	end
	
	if(copy_strategy)
		copyto!(optimization_output.strategy, test_strategy)
	end
	evaluate_success_probabilities!(problem, optimization_output.strategy, optimization_output.success_probabilities)
	optimization_output.value = best_value
end



#### Classical ####
struct DeterministicStrategy <: Strategy
    outputs_Alice::Array{Int64} # Array of length n_X corresponding to the choices of a
    outputs_Bob::Array{Int64} # Array of length n_Y corresponding to the choices of b
	function DeterministicStrategy(game::Game)
		new(rand(1:game.n_A, game.n_X), rand(1:game.n_B, game.n_Y))
	end
end

function copyto!(strat1::DeterministicStrategy, strat2::DeterministicStrategy)
	strat1.outputs_Alice[:] .= strat1.outputs_Alice
	strat2.outputs_Bob[:]   .= strat2.outputs_Bob
end

	

""" 
	evaluate_success_probabilities(game, strategy)
	
Given a game and a deterministic strategy, returns the success probabilities for every input pair. 
"""
function evaluate_success_probabilities!(game::Game, strategy::DeterministicStrategy, probabilities::Matrix{Float64})
	for x=1:game.n_X
		for y=1:game.n_Y
			probabilities[x,y] = (x,y,strategy.outputs_Alice[x],strategy.outputs_Bob[y]) in game.R
		end
	end			
end	

function evaluate_success_probability(game::Game, strategy::DeterministicStrategy, distribution::Matrix{Float64})::Float64
	tot = 0.0
	for x=1:game.n_X
		for y=1:game.n_Y
			if(distribution[x,y] > 1e-08)
				tot += distribution[x,y] * ((x,y,strategy.outputs_Alice[x],strategy.outputs_Bob[y]) in game.R)
			end
		end
	end
	return tot
end

struct ClassicalSolverData <: InternalSolverData
	window_size::Int
	attempts::Int
	S_x::Matrix{Float64}
	S_y::Matrix{Float64}
	
	function ClassicalSolverData(game::Game, window_size::Int, attempts::Int)
		new(window_size, attempts, zeros(window_size, game.n_A), zeros(window_size, game.n_B))
	end
end


"""
	improve_deterministic_strategy(game, strategy, distribution; window_size = 3)

Given a game, a deterministic strategy and a distribution on the inputs, uses local search to find a strategy that scores better on average under the given distribution. 
The way this is done is by selecting a subset of inputs at random, trying all new partial assignments to these inputs and keeping the best one. 
"""
function improve_strategy!(game::Game, strategy::DeterministicStrategy, distribution::Matrix{Float64}, data::ClassicalSolverData)
	initial_value = evaluate_success_probability(game, strategy, distribution)
	for _=1:data.attempts
		window_size = data.window_size
		@assert game.n_X >= window_size
		@assert game.n_Y >= window_size
		
		# For both x and y, pick the subset of values that do not have probability zero and shuffle them. 
		indices_x = shuffle(findall((x)->sum(distribution[x,:]) > 1e-07, 1:game.n_X))
		indices_y = shuffle(findall((y)->sum(distribution[:,y]) > 1e-07, 1:game.n_Y))
		window_size = min(window_size, length(indices_x), length(indices_y))
		@assert window_size > 0
		
		
		# We will try all possible new assignments to the first window_size values of indices_x and indices_y and choose the best one.
		# Before that, we do some precomputation to be able to compute the value of each assignment faster.
		constant_term = 0
		
		for i_x=1:window_size+1:game.n_X
			x = indices_x[i_x]
			for i_y = window_size+1:game.n_Y
				y = indices_y[i_y]
				if((x,y,strategy.outputs_Alice[x],strategy.outputs_Bob[y]) in game.R)
					constant_term += distribution[x,y]
				end
			end
		end
		
		data.S_x[:,:] .= 0
		data.S_y[:,:] .= 0
		for i=1:window_size
			x = indices_x[i]
			for a=1:game.n_A
				for i_y=window_size+1:game.n_Y
					y = indices_y[i_y]
					if((x,y,a,strategy.outputs_Bob[y]) in game.R)
						data.S_x[i,a] += distribution[x,y]
					end
				end
			end
			
			y = indices_y[i]
			for b=1:game.n_B
				for i_x=window_size+1:game.n_X
					x = indices_x[i_x]
					if((x,y,strategy.outputs_Alice[x],b) in game.R)
						data.S_y[i,b] += distribution[x,y]
					end			
				end
			end
		end
		
		evals = 0
		best_value = initial_value
		for new_outputs_Alice in Iterators.product([1:game.n_A for i=1:window_size]...)
		
			sum_alice = 0
			for i=1:window_size
				sum_alice += data.S_x[i, new_outputs_Alice[i]]
			end	
	
			
			for new_outputs_Bob in Iterators.product([1:game.n_B for i=1:window_size]...)
	
				value = constant_term + sum_alice
				for i=1:window_size
					value += data.S_y[i, new_outputs_Bob[i]]
				end
			
				for i_x=1:window_size
					x = indices_x[i_x]
					for i_y=1:window_size
						y = indices_y[i_y]
						
						if((x,y,new_outputs_Alice[i_x],new_outputs_Bob[i_y]) in game.R)
							value += distribution[x,y]
						end
					end
				end
				
				if(value > best_value)
					for i_x=1:window_size
						strategy.outputs_Alice[indices_x[i_x]] = new_outputs_Alice[i_x]
					end
				
					for i_y=1:window_size
						strategy.outputs_Bob[indices_y[i_y]] = new_outputs_Bob[i_y]
					end
					
					best_value = value
				end
		
			end
		end
		if(best_value > initial_value)
			break
		end
	end

end

function scramble_strategy!(game::Game, strategy::DeterministicStrategy)
	for x=1:game.n_X
		if(rand() < 0.2)
			strategy.outputs_Alice[x] = rand(1:game.n_A)
		end
	end
	
	for y=1:game.n_Y
		if(rand() < 0.2)
			strategy.outputs_Bob[y] = rand(1:game.n_B)
		end
	end	
end

#### Complex hermitian matrix modelling

function enforce_SDP_constraints(model::Model, variable::Symmetric{VariableRef, Matrix{VariableRef}})
	dim = div(size(variable,1), 2)
	for i=1:dim
		for j=i:dim
			@constraint(model, variable[i,j] == variable[i+dim, j+dim])
			@constraint(model, variable[i+dim,j] == -variable[i, j+dim])
		end
	end
end
	
			
#### Yao ####


struct YaoProtocol <: Strategy
	states::Vector{Matrix{Float64}}
	POVMs::Vector{Pair{Matrix{Float64}, Matrix{Float64}}} # Corresponding to the probability of outputting one
	
	function YaoProtocol(problem::OneWayCommunicationProblem) # Generates a protocol at random
		new([realify(gen_rho(problem.c)) for x=1:problem.n_X], [Pair(realify.(gen_rand_POVM(2,problem.c))...) for y=1:problem.n_Y])
	end

end

function copyto!(prot1::YaoProtocol, prot2::YaoProtocol)
	for x=1:length(prot1.states)
		prot1.states[x] .= prot2.states[x]
	end
	for y=1:length(prot1.POVMs)
		for i=1:2
			prot1.POVMs[y][i] .= prot2.POVMs[y][i]
		end
	end
end

function evaluate_success_probabilities!(problem::OneWayCommunicationProblem, protocol::YaoProtocol, success_probabilities::Matrix{Float64})
	for x=1:problem.n_X
		for y=1:problem.n_Y
			if(problem.promise(x,y))
				success_probabilities[x,y] = tr(protocol.states[x] * protocol.POVMs[y][problem.f(x,y)+1]) / 2.0
			else
				success_probabilities[x,y] = 0
			end
		end
	end
end

function evaluate_success_probability(problem::OneWayCommunicationProblem, protocol::YaoProtocol, distribution::Matrix{Float64})::Float64
	tot = 0.0
	for x=1:problem.n_X
		for y=1:problem.n_Y
			if(problem.promise(x,y))
				tot += distribution[x,y] * tr(protocol.states[x] * protocol.POVMs[y][problem.f(x,y)+1]) / 2.0
			end
		end
	end
	return tot
end

struct YaoSolverData <: InternalSolverData
	SDP_states::Model
	states::Vector{Symmetric{VariableRef, Matrix{VariableRef}}}
	SDP_POVMs
	POVMs::Vector{Pair{Symmetric{VariableRef, Matrix{VariableRef}}}}
	
	function YaoSolverData(problem::OneWayCommunicationProblem; eps_abs = 1e-05)
		dim = problem.c
		
		SDP_states = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "eps_abs" => eps_abs));	
		set_silent(SDP_states)
		states = [@variable(SDP_states, [1:2*dim, 1:2*dim], PSD) for x=1:problem.n_X]
		
		for x=1:problem.n_X
			@constraint(SDP_states, sum(states[x][i,i] for i=1:dim) == 1) # Force the trace to be one
			enforce_SDP_constraints(SDP_states, states[x])
		end
		
		SDP_POVMs = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "eps_abs" => eps_abs));	
		set_silent(SDP_POVMs)
		
		POVMs = [Pair(@variable(SDP_POVMs, [1:2*dim, 1:2*dim], PSD), @variable(SDP_POVMs, [1:2*dim, 1:2*dim], PSD)) for y=1:problem.n_Y]
		Id = diagm([1.0 for i=1:2*dim])
		for y=1:problem.n_Y
			@constraint(SDP_POVMs, POVMs[y][1] + POVMs[y][2] .== Id)
			enforce_SDP_constraints(SDP_POVMs, POVMs[y][1])
			enforce_SDP_constraints(SDP_POVMs, POVMs[y][2])
		end
		new(SDP_states, states, SDP_POVMs, POVMs)
	end
end

function improve_strategy!(problem::OneWayCommunicationProblem, protocol::YaoProtocol, distribution::Matrix{Float64}, data::YaoSolverData)
	@objective(data.SDP_states, Max, sum(problem.promise(x,y) ? (distribution[x,y] * tr(protocol.POVMs[y][problem.f(x,y)+1] * data.states[x])) : 0 for x=1:problem.n_X for y=1:problem.n_Y))
	optimize!(data.SDP_states)
	for x=1:problem.n_X
		protocol.states[x] .= JuMP.value.(data.states[x])
	end
	
	@objective(data.SDP_POVMs, Max, sum(problem.promise(x,y) ? (distribution[x,y] * tr(data.POVMs[y][problem.f(x,y)+1] * protocol.states[x])) : 0 for x=1:problem.n_X for y=1:problem.n_Y))
	optimize!(data.SDP_POVMs)
	for y=1:problem.n_Y
		for i=1:2
			protocol.POVMs[y][i] .= JuMP.value.(data.POVMs[y][i])
		end
	end	
end

function scramble_strategy!(problem::OneWayCommunicationProblem, protocol::YaoProtocol)
	for x=1:problem.n_X
		if(rand() < 0.3)
			new_state = realify(gen_rho(problem.c))
			protocol.states[x] = 0.5 * protocol.states[x] + 0.5 * new_state
		end
	end
		
	for y=1:problem.n_Y
		if(rand() < 0.3)
			new_POVM = gen_rand_POVM(2, problem.c)
			for i=1:2
				protocol.POVMs[y][i] .= 0.5 * protocol.POVMs[y][i] + realify(new_POVM[i])
			end
		end
	end
end

	

function optimize_yao(problem::OneWayCommunicationProblem, distribution::Matrix{Float64})
	data = YaoSolverData(problem)
	protocol = YaoProtocol(problem)
	optimization_output = OptimizationOutput(-1.0, zeros(problem.n_X, problem.n_Y), protocol)
	optimize_strategy!(problem, distribution, data, optimization_output)
	return optimization_output
end

function generate_hard_distribution_yao(problem::OneWayCommunicationProblem)
	data = YaoSolverData(problem)
	N = problem.n_X * problem.n_Y
end
	

############## Entangled values of nonlocal games ##############


struct EntangledStrategy <: Strategy
	A::Dict{Tuple{Int,Int}, Matrix{Float64}}
	B::Dict{Tuple{Int,Int}, Matrix{Float64}}
	dim::Int
	function EntangledStrategy(game::Game, dim::Int) # Generates a protocol at random
		A::Dict{Tuple{Int,Int}, Matrix{Float64}} = Dict()
		B::Dict{Tuple{Int,Int}, Matrix{Float64}} = Dict()
		
		for x=1:game.n_X
			POVM = gen_rand_POVM(game.n_A, dim)
			for a=1:game.n_A
				A[(x,a)] = realify(POVM[a])
			end
		end
		
		for y=1:game.n_Y
			POVM = gen_rand_POVM(game.n_B, dim)
			for b=1:game.n_B
				B[(y,b)] = realify(POVM[b])
			end
		end
		return new(A,B,dim)
	end	
	function EntangledStrategy(A,B,dim)
		return new(A,B,dim)
	end
end

""" 
	copyto!(strat1::Entanglement_strategy, strat2::Entanglement_strategy)

Copies the contents of strat2 into strat1. Assumes that strat1 and strat2 are for games of the same format.
"""
function copyto!(strat1::EntangledStrategy, strat2::EntangledStrategy)
	for p in keys(strat1.A)
		strat1.A[p][:,:] .= strat2.B[p][:,:]
	end
	
	for p in keys(strat1.B)
		strat1.B[p][:,:] .= strat2.B[p][:,:]
	end
end

function evaluate_success_probabilities!(game::Game, strategy::EntangledStrategy, success_probabilities::Matrix{Float64})
	for x=1:game.n_X
		for y=1:game.n_Y
			success_probabilities[x,y] = sum((x,y,a,b) in game.R ? tr(strategy.A[(x,a)] * strategy.B[(y,b)]) : 0 for a=1:game.n_A for b=1:game.n_B) / strategy.dim / 2
		end
	end	
end

function evaluate_success_probability(game::Game, strategy::EntangledStrategy, distribution::Matrix{Float64})
	tot = 0 
	for x=1:game.n_X
		for y=1:game.n_Y
			if(distribution[x,y] > 1e-10)
				tot += distribution[x,y] * sum((x,y,a,b) in game.R ? tr(strategy.A[(x,a)] * strategy.B[(y,b)]) : 0 for a=1:game.n_A for b=1:game.n_B) / strategy.dim / 2
			end
		end
	end	
	return tot
end

struct EntangledSolverData <: InternalSolverData
	SDP_A::Model
	A::Dict{Tuple{Int, Int}, Symmetric{VariableRef, Matrix{VariableRef}}}
	SDP_B::Model
	B::Dict{Tuple{Int, Int}, Symmetric{VariableRef, Matrix{VariableRef}}}
	
	function EntangledSolverData(game::Game, dim::Int; eps_abs::Float64 = 1e-06, trace_lower_bound::Float64 = 1.0, allow_state_optimization = true)
	
		### Building SDP_A ###
		SDP_A = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "eps_abs" => eps_abs));
		set_silent(SDP_A)
		A = Dict()
		for x=1:game.n_X
			for a=1:game.n_A
				A[(x,a)] = @variable(SDP_A, [1:2*dim, 1:2*dim], PSD)
				enforce_SDP_constraints(SDP_A, A[(x,a)])
				@constraint(SDP_A, tr(A[(x,a)]) >= trace_lower_bound)
			end
		end
		if(allow_state_optimization)
			tau = @variable(SDP_A, [1:dim, 1:dim], PSD)
			@constraint(SDP_A, tr(tau) == dim)
			for x=1:game.n_X
				@constraint(SDP_A, sum(A[(x,a)] for a=1:game.n_A)[1:dim, 1:dim] .== tau)
				@constraint(SDP_A, sum(A[(x,a)] for a=1:game.n_A)[1:dim, dim+1 : end] .== 0)
			end
		else
			for x=1:game.n_X
				@constraint(SDP_A, sum(A[(x,a)] for a=1:game.n_A) .== diagm([1.0 for i=1:2*dim]))
			end
		end
		
		### Building SDP_B ###
		Id = diagm([1 for i=1:dim])
		SDP_B = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "eps_abs" => 1e-6));
		set_silent(SDP_B)
		B = Dict()
		for y=1:game.n_Y
			for b=1:game.n_B
				B[(y,b)] = @variable(SDP_B, [1:2*dim, 1:2*dim], PSD)
				enforce_SDP_constraints(SDP_B, B[(y,b)])
				@constraint(SDP_B, tr(B[(y,b)]) >= trace_lower_bound)
			end
		end
		
		for y=1:game.n_Y
			@constraint(SDP_B, sum(B[(y,b)] for b=1:game.n_B) .== diagm([1.0 for i=1:2*dim]))
		end
		return new(SDP_A, A, SDP_B, B)
	end
		
	
end

function improve_strategy!(game::Game, strategy::EntangledStrategy, distribution::Matrix{Float64}, data::EntangledSolverData)
	@objective(data.SDP_A, Max, sum(distribution[x,y] * (((x,y,a,b) in game.R) ? tr(strategy.B[(y,b)] * data.A[(x,a)]) : 0) for a=1:game.n_A for b=1:game.n_B for x=1:game.n_X for y=1:game.n_Y))
	optimize!(data.SDP_A)
	for x=1:game.n_X
		for a=1:game.n_A
			strategy.A[(x,a)] .= JuMP.value.(data.A[(x,a)])
		end
	end
	
	@objective(data.SDP_B, Max, sum(distribution[x,y] * (((x,y,a,b) in game.R) ? tr(data.B[(y,b)] * strategy.A[(x,a)]) : 0) for a=1:game.n_A for b=1:game.n_B for x=1:game.n_X for y=1:game.n_Y))
	optimize!(data.SDP_B)
	for y=1:game.n_Y
		for b=1:game.n_B
			strategy.B[(y,b)] .= JuMP.value.(data.B[(y,b)])
		end
	end	
end

function scramble_strategy!(game::Game, strategy::EntangledStrategy)
 
	dim = div(size(strategy.A[(1,1)], 1), 2)
	for x=1:game.n_X
		if(rand() < 0.3)
			POVM = gen_rand_POVM(game.n_A, dim)
			for a=1:game.n_A
				strategy.A[(x,a)] = 0.5 * strategy.A[(x,a)] + 0.5 * realify(POVM[a])
			end
		end
	end
	
	for y=1:game.n_Y
		if(rand() < 0.3)
			POVM = gen_rand_POVM(game.n_B, dim)
			for b=1:game.n_B
				strategy.B[(y,b)] = 0.5 * strategy.B[(y,b)] + 0.5 * realify(POVM[b])
			end
		end
	end
end

function optimize_entangled(game::Game, dim::Int, distribution::Matrix{Float64})
	data = EntangledSolverData(game, dim)
	strategy = EntangledStrategy(game, dim)
	optimization_output = OptimizationOutput(-1.0, zeros(game.n_X, game.n_Y), strategy)
	optimize_strategy!(game, distribution, data, optimization_output)
	return optimization_output
end


### Examples ###
#const bits = [[0;0],[0;1],[1;0],[1;1]]

function V_msg(x,y,a,b)
   bits = [[0;0],[0;1],[1;0],[1;1]]
   column = [bits[a]; sum(bits[a]) % 2];
   row = [bits[b]; (sum(bits[b]) + 1) % 2];
   return column[y] == row[x]
end

const MagicSquareGame = Game(3,3,4,4,V_msg)


#function optimize_entangled(game::Game, strategy::EntangledStrategy, distribution::Matrix{Float64}; 

# """	
	# optimize_entanglement(game::Game, strategy::Entanglement_strategy, distribution::Matrix{Float64}; impose_maximally_entangled = false, max_iter=50, trace_lower_bound = 1.0, epsilon = 1e-04)

# Given a game, a starting strategy and a distribution on the inputs, attempts to find a strategy that scores better under the given distribution. The algorithm used is the one presented in 
# the paper 'Bounds on Quantum Correlations in Bell Inequality Experiments' by Liang and Doherty (2007). impose\\_maximally\\_entangled corresponds to whether the shared state 
# should be restricted to be a maximally entangled one, and max\\_iter corresponds to the largest number of iterations allowed. trace\\_lower\\_bound is a lower bound on the trace of the POVMs
# during the course of the algorithm, which was observed to help enormously in some cases. Setting it to zero nullifies this aspect of the algorithm. epsilon is the tolerance within which
# the algorithm assumes to have converged to a local maximum.
# """

# function optimize_entanglement(game::Game, dim::Int, distribution::Matrix{Float64}; impose_maximally_entangled = false, iterations=50)
	# return optimize_entanglement(game, Entanglement_strategy(game, dim), distribution; impose_maximally_entangled = impose_maximally_entangled, iterations = iterations)
# end

# colonnes = [[0;0], [0;1], [1;0],[1;1]];

# function V_carre(x,y,a,b)
   # colonne = [colonnes[a]; sum(colonnes[a]) % 2];
   # rangee = [colonnes[b]; (sum(colonnes[b]) + 1) % 2];
   # return colonne[y] == rangee[x]
# end

# function find_good_entangled_protocol(problem::OneWayCommunicationProblem, dim::Int, distribution::Matrix{Float64}; impose_maximally_entangled = false, iterations = 50)
	# R::Set{NTuple{4, Int64}} = Set()
	# new_distribution = zeros(Float64, problem.n_X, problem.n_Y * problem.c)
	# unsplit_y = (y) -> ((y-1)%problem.c + 1, div(y-1-(y-1)%problem.c, problem.c)+1)

	# for x=1:problem.n_X
		# for y=1:problem.n_Y * problem.c
			# new_distribution[x,y] = 1/problem.c * distribution[x, unsplit_y(y)[2]]
		# end
	# end
	
	# V = (x,y,a,b) -> (a != unsplit_y(y)[1]) || (b - 1 == problem.f(x, unsplit_y(y)[2]))

	# game = Game(problem.n_X, problem.n_Y * problem.c, problem.c, 2, V)
	# prob = dot(new_distribution, optimize_entanglement(game, dim, new_distribution; impose_maximally_entangled = impose_maximally_entangled, iterations = iterations))
	# return problem.c * prob - problem.c + 1
# end

	
		
		
