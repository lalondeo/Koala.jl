include("utils.jl")
using JuMP
using COSMO
using LinearAlgebra
using Random
using RandomMatrices
using CPLEX


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



#### Classical ####
mutable struct Deterministic_strategy
    outputs_Alice::Array{Int64} # Array of length n_X corresponding to the choices of a
    outputs_Bob::Array{Int64} # Array of length n_Y corresponding to the choices of b
end

""" 
	evaluate_success_probabilities(game, strategy)
	
Given a game and a deterministic strategy, returns the success probabilities for every input pair. 
"""
function evaluate_success_probabilities(game::Game, strategy::Deterministic_strategy)::Matrix{Float64}
	vals = zeros(game.n_X, game.n_Y)
	for x=1:game.n_X
		for y=1:game.n_Y
			vals[x,y] = (x,y,a,strategy.outputs_Alice[x],strategy.outputs_Bob[y]) in game.R
		end
	end
	return vals
				
end	


""" 
	evaluate_deterministic_strategy(game, strategy, distribution)
	
Given a game, a deterministic strategy and a distribution on the inputs, returns the average success probability.
"""
function evaluate_deterministic_strategy(game::Game, strategy::Deterministic_strategy, distribution::Matrix{Float64})::Float64
    return dot(distribution, evaluate_success_probabilities(game, strategy))
end


"""
	improve_deterministic_strategy(game, strategy, distribution; window_size = 3)

Given a game, a deterministic strategy and a distribution on the inputs, uses local search to find a strategy that scores better on average under the given distribution. 
The way this is done is by selecting a subset of inputs at random, trying all new partial assignments to these inputs and keeping the best one. 
"""
function improve_deterministic_strategy(game::Game, strategy::Deterministic_strategy, distribution::Matrix{Float64}; window_size = 3)::Float64
	@assert game.n_X >= window_size
	@assert game.n_Y >= window_size
	
	# For both x and y, pick the subset of values that do not have probability zero and shuffle them. 
	indices_x = shuffle(findall((x)->sum(distribution[x,:]) > 1e-05, 1:game.n_X))
	indices_y = shuffle(findall((y)->sum(distribution[:,y]) > 1e-05, 1:game.n_Y))
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
	
	S_x = zeros(window_size, game.n_A)
	S_y = zeros(window_size, game.n_B)
	
	for i=1:window_size
		x = indices_x[i]
		for a=1:game.n_A
			for i_y=window_size+1:game.n_Y
				y = indices_y[i_y]
				if((x,y,a,strategy.outputs_Bob[y]) in game.R)
					S_x[i,a] += distribution[x,y]
				end
			end
		end
		
		y = indices_y[i]
		for b=1:game.n_B
			for i_x=window_size+1:game.n_X
				x = indices_x[i_x]
				if((x,y,strategy.outputs_Alice[x],b) in game.R)
					S_y[i,b] += distribution[x,y]
				end			
			end
		end
	end
	
	best_value = evaluate_deterministic_strategy(game, strategy, distribution)
	
	for new_outputs_Alice in Iterators.product([vals_A for i=1:window_size]...)
		sum_alice = 0
		for i=1:window_size
			sum_alice += S_x[i, new_outputs_Alice[i]]
		end
		
        for new_outputs_Bob in Iterators.product([vals_B for i=1:window_size]...)
			value = constant_term + sum_alice
			for i=1:window_size
				value += S_y[i, new_outputs_Bob[i]]
			end
			
			for i_x=1:window_size
				x = indices_x[i_x]
				for i_y=1:window_size
					y = indices_y[i_y]
					if((x,y,new_outputs_Alice[x],new_outputs_Bob[y]) in game.R)
						value += distribution[x,y]
					end
				end
			end
			
			if(value > best_value)
				strategy.outputs_Alice[:] = new_outputs_Alice
				strategy.outputs_Bob[:] = new_outputs_Bob
				best_value = value
			end
		end
	end
	return best_value
end
			
			
	
	
	
	

#### Yao ####


function gen_rho_unif(n)
	U = rand(Haar(1), n)
	for i=1:n
		if(U[1,i] < 0)
			U[:,i] *= -1
		end
	end
	
	v = broadcast((x)->x^2, rand(Haar(1), n)[:,1])
	return U * diagm(v) * transpose(U)
end

function generate_POVM_element(n)
	rho = gen_rho_unif(n)
	return rho .* rand() ./ eigmax(rho)
end

mutable struct Yao_protocol
	states_Alice::Vector{Matrix{Float64}}
	POVMs_Bob::Vector{Matrix{Float64}} # Corresponding to the probability of outputting one
	
	function Yao_protocol(problem::OneWayCommunicationProblem) # Generates a protocol at random
		return new([gen_rho_unif(problem.c) for i=1:problem.n_X], [generate_POVM_element(problem.c) for i=1:problem.n_Y])
	end

end

function improve_rhos(problem::OneWayCommunicationProblem, protocol::Yao_protocol, distribution::Matrix{Float64})
	model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "eps_abs" => 1e-3));
	dimension = problem.c
	
	rhos = [@variable(model, [1:dimension, 1:dimension], PSD) for x=1:problem.n_X]	
	for x=1:problem.n_X
		@constraint(model, tr(rhos[x]) == 1)
	end
	
	@objective(model, Max, sum(problem.promise(x,y) ? distribution[x,y] * tr((problem.f(x,y) ? protocol.POVMs_Bob[y] : I - protocol.POVMs_Bob[y]) * rhos[x]) : 0 for x=1:problem.n_X for y=1:problem.n_Y))
	set_silent(model)
	optimize!(model)
	for x=1:problem.n_X
		protocol.states_Alice[x] = JuMP.value.(rhos[x])
	end
end

function improve_POVMs(problem::OneWayCommunicationProblem, protocol::Yao_protocol, distribution::Matrix{Float64})
	model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "eps_abs" => 1e-3));
	dimension = problem.c
	Id = diagm([1.0 for i=1:dimension]);
	POVMs_1 = [@variable(model, [1:dimension, 1:dimension], PSD) for y=1:problem.n_Y]	
	POVMs_0 = [@variable(model, [1:dimension, 1:dimension], PSD) for y=1:problem.n_Y]		
	for y=1:problem.n_Y
		@constraint(model, POVMs_0[y] + POVMs_1[y] .== Id)
	end
	
	@objective(model, Max, sum(problem.promise(x,y) ? distribution[x,y] * tr(protocol.states_Alice[x] * (problem.f(x,y) ? POVMs_1[y] : POVMs_0[y])) : 0 for x=1:problem.n_X for y=1:problem.n_Y))
	set_silent(model)
	optimize!(model)
	for y=1:problem.n_Y
		protocol.POVMs_Bob[y] = JuMP.value.(POVMs_1[y])
	end

end

function evaluate_success_probabilities(problem::OneWayCommunicationProblem, protocol::Yao_protocol)
	dimension = problem.c
	Id = diagm([1.0 for i=1:dimension]);
	success_probabilities = zeros(problem.n_X, problem.n_Y)
	for x=1:problem.n_X
		for y=1:problem.n_Y
			if(problem.promise(x,y))
				if(problem.f(x,y))
					success_probabilities[x,y] = tr(protocol.states_Alice[x] * protocol.POVMs_Bob[y])
				else
					success_probabilities[x,y] = tr(protocol.states_Alice[x] * (Id - protocol.POVMs_Bob[y]))
				end
			end
		end
	end
	return success_probabilities
end	


function optimize_yao(problem::OneWayCommunicationProblem, protocol::Yao_protocol, distribution::Matrix{Float64}; iterations = 50) # Returns the success probabilities
	old_success_probability = 0;
	for _=1:iterations
		improve_rhos(problem, protocol, distribution)
		improve_POVMs(problem, protocol, distribution)
		new_success_probability = dot(evaluate_success_probabilities(problem, protocol), distribution);
		if(new_success_probability < old_success_probability + 1e-04 || new_success_probability > 0.999)
			break
		end
		old_success_probability = new_success_probability;
	end
	return evaluate_success_probabilities(problem, protocol)
end

function optimize_yao(problem::OneWayCommunicationProblem, distribution::Matrix{Float64}; iterations = 50)
	return optimize_yao(problem, Yao_protocol(problem), distribution; iterations = iterations)
end

function generate_hard_distribution_yao(problem::OneWayCommunicationProblem)
	best_known_protocol = Yao_protocol(problem)
	N = problem.n_X * problem.n_Y
	legal_inputs = reshape([problem.promise(x,y) for x=1:problem.n_X, y=1:problem.n_Y], (N,))
	function oracle(_distribution::Array{Float64})
		distribution = reshape(_distribution, (problem.n_X, problem.n_Y))
		success_probabilities = optimize_yao(problem, best_known_protocol, distribution; iterations = 2);
		success_probability = dot(success_probabilities, distribution)
		if(rand() < 0.9 || success_probability > 0.99)
			return reshape(success_probabilities, (N,))
		end
			
		other_protocol = Yao_protocol(problem);
		other_success_probabilities = optimize_yao(problem, other_protocol, distribution)
		other_success_probability = dot(other_success_probabilities, distribution)
		if(success_probability < other_success_probability)
			best_known_protocol = other_protocol
			return reshape(other_success_probabilities, (N,))
		else
			return reshape(success_probabilities, (N,))
		end
	end
	return generate_hard_distribution(N, legal_inputs, oracle; iterations=1000)
end



### Nonlocal games ###

function gen_rand_POVM(n, dim)
	Id = diagm([1 for i=1:dim]);
	POVM = []
	for a=1:n-1
		diag = rand(dim);
		U = rand(Haar(2), dim)
		push!(POVM, U * diagm(diag) * adjoint(U))
	end
	tot = sum(POVM)
	val_max = maximum(real.(eigvals(tot)))
	
	for a=1:n-1
		POVM[a] *= (1-1/dim) / val_max
	end	
	push!(POVM, Id - tot * (1-1/dim) / val_max)
	shuffle!(POVM)
	return POVM
end

function realify(M)
	return [real(M) imag(M); -imag(M) real(M)]
end

function unrealify(M)
	dim = div(size(M,1),2)
	return (M[1:dim, 1:dim] + im * M[1:dim, dim+1:end])
end

mutable struct Entanglement_strategy
	A::Dict{Tuple{Int,Int}, Matrix{Float64}}
	B::Dict{Tuple{Int,Int}, Matrix{Float64}}
	dim::Int
	function Entanglement_strategy(game::Game, dim::Int) # Generates a protocol at random
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
	function Entanglement_strategy(A,B,dim)
		return new(A,B,dim)
	end
end

""" 
	copy!(strat1::Entanglement_strategy, strat2::Entanglement_strategy)

Copies the contents of strat2 into strat1. Assumes that strat1 and strat2 are for games of the same format.
"""
function copy!(strat1::Entanglement_strategy, strat2::Entanglement_strategy)
	for p in keys(strat1.A)
		strat1.A[p][:,:] .= strat2.B[p][:,:]
	end
	
	for p in keys(strat1.B)
		strat1.B[p][:,:] .= strat2.B[p][:,:]
	end
end

function evaluate_success_probabilities(game::Game, strategy::Entanglement_strategy)
	success_probabilities = zeros(game.n_X, game.n_Y)
	for x=1:game.n_X
		for y=1:game.n_Y
			success_probabilities[x,y] = sum((x,y,a,b) in game.R ? tr(strategy.A[(x,a)] * strategy.B[(y,b)]) : 0 for a=1:game.n_A for b=1:game.n_B) / strategy.dim / 2
		end
	end	
	return success_probabilities
end



function SDP_entanglement_A(game::Game, strategy::Entanglement_strategy, distribution::Matrix{Float64}; allow_state_optimization = false, trace_lower_bound = 1.0)
#	Id = diagm([1 for i=1:strategy.dim])

	model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "eps_abs" => 1e-6));
	A = Dict()
	for x=1:game.n_X
		for a=1:game.n_A
			A[(x,a)] = @variable(model, [1:2*strategy.dim, 1:2*strategy.dim], PSD)
			for i=1:strategy.dim
				for j=1:strategy.dim
					@constraint(model, A[(x,a)][i,j] == A[(x,a)][i + strategy.dim, j + strategy.dim])
				end
			end
			@constraint(model, tr(A[(x,a)]) >= trace_lower_bound)
				
		end
	end
	
	if(allow_state_optimization)
		tau = @variable(model, [1:strategy.dim, 1:strategy.dim], PSD)
		@constraint(model, tr(tau) == strategy.dim)
		for x=1:game.n_X
			@constraint(model, sum(A[(x,a)] for a=1:game.n_A)[1:strategy.dim, 1:strategy.dim] .== tau)
			@constraint(model, sum(A[(x,a)] for a=1:game.n_A)[1:strategy.dim, strategy.dim + 1 : end] .== 0)
		end
	else
		for x=1:game.n_X
			@constraint(model, sum(A[(x,a)] for a=1:game.n_A) .== diagm([1.0 for i=1:2*strategy.dim]))
		end
	end
	
	
	
	@objective(model, Max, sum(distribution[x,y] * (((x,y,a,b) in game.R) ? tr(strategy.B[(y,b)] * A[(x,a)]) : 0) for a=1:game.n_A for b=1:game.n_B for x=1:game.n_X for y=1:game.n_Y))
	
	set_silent(model)
	optimize!(model)
	for a=1:game.n_A
		for x=1:game.n_X
			strategy.A[(x,a)] = JuMP.value.(A[(x,a)])
		end
	end
end

function SDP_entanglement_B(game::Game, strategy::Entanglement_strategy, distribution::Matrix{Float64}; trace_lower_bound = 1.0)
	Id = diagm([1 for i=1:strategy.dim])
	model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "eps_abs" => 1e-6));
	B = Dict()
	for y=1:game.n_Y
		for b=1:game.n_B
			B[(y,b)] = @variable(model, [1:2*strategy.dim, 1:2*strategy.dim], PSD)
			for i=1:strategy.dim
				for j=1:strategy.dim
					@constraint(model, B[(y,b)][i,j] == B[(y,b)][i + strategy.dim, j + strategy.dim])
				end
			end
			@constraint(model, tr(B[(y,b)]) >= trace_lower_bound)
		end
	end
	
	for y=1:game.n_Y
		@constraint(model, sum(B[(y,b)] for b=1:game.n_B) .== diagm([1.0 for i=1:2*strategy.dim]))
	end

	@objective(model, Max, sum(distribution[x,y] *  (((x,y,a,b) in game.R) ? tr(B[(y,b)] * strategy.A[(x,a)]) : 0) for a=1:game.n_A for b=1:game.n_B for x=1:game.n_X for y=1:game.n_Y))
	
	set_silent(model)
	optimize!(model)
	for y=1:game.n_Y
		for b=1:game.n_B
			strategy.B[(y,b)] = JuMP.value.(B[(y,b)])
		end
	end
end

"""	
	optimize_entanglement(game::Game, strategy::Entanglement_strategy, distribution::Matrix{Float64}; impose_maximally_entangled = false, max_iter=50, trace_lower_bound = 1.0, epsilon = 1e-04)

Given a game, a starting strategy and a distribution on the inputs, attempts to find a strategy that scores better under the given distribution. The algorithm used is the one presented in 
the paper 'Bounds on Quantum Correlations in Bell Inequality Experiments' by Liang and Doherty (2007). impose\\_maximally\\_entangled corresponds to whether the shared state 
should be restricted to be a maximally entangled one, and max\\_iter corresponds to the largest number of iterations allowed. trace\\_lower\\_bound is a lower bound on the trace of the POVMs
during the course of the algorithm, which was observed to help enormously in some cases. Setting it to zero nullifies this aspect of the algorithm. epsilon is the tolerance within which
the algorithm assumes to have converged to a local maximum.
"""
function improve_entangled_strategy(game::Game, strategy::Entanglement_strategy, distribution::Matrix{Float64}; 
				impose_maximally_entangled::Bool = false, max_iter::Int=50, trace_lower_bound::Float64 = 1.0, epsilon::Float64 = 1e-04, stop_at_local_maximum::Bool = false)
				
	old_value = 0
	best_value = 0
	best_success_probabilities = zeros(game.n_X, game.n_Y)
	test_strategy = deepcopy(strategy)
	copy_strategy = false # Controls whether the strategy in test_strategy is the best strategy known so far and should be copied to strategy in the event that the iteration limit were to be hit
	for i=1:max_iter
		copy_strategy = false
		A = SDP_entanglement_A(game, test_strategy, distribution; allow_state_optimization = !impose_maximally_entangled)
		B = SDP_entanglement_B(game, test_strategy, distribution)
		success_probabilities .= evaluate_success_probabilities(game, test_strategy)
		value = dot(success_probabilities, distribution)
		if(abs(value - old_value) < epsilon || value > 1 - epsilon)
			if(value > best_value)
				best_success_probabilities[:,:] = success_probabilities
				best_value = value
				for x=1:game.n_X
					for a=1:game.n_A
						game.A[(x,a)][:,:] = test_strategy.A[(x,a)]
					end
				end
				
				
				
			end
			if(stop_at_local_maximum || value > 1 - epsilon)
				break
			end
		elseif(value > best_value)
			copy_strategy = true
		end
	
		old_value = value
	end
	return success_probabilities
end

function optimize_entanglement(game::Game, dim::Int, distribution::Matrix{Float64}; impose_maximally_entangled = false, iterations=50)
	return optimize_entanglement(game, Entanglement_strategy(game, dim), distribution; impose_maximally_entangled = impose_maximally_entangled, iterations = iterations)
end

colonnes = [[0;0], [0;1], [1;0],[1;1]];

function V_carre(x,y,a,b)
   colonne = [colonnes[a]; sum(colonnes[a]) % 2];
   rangee = [colonnes[b]; (sum(colonnes[b]) + 1) % 2];
   return colonne[y] == rangee[x]
end

function find_good_entangled_protocol(problem::OneWayCommunicationProblem, dim::Int, distribution::Matrix{Float64}; impose_maximally_entangled = false, iterations = 50)
	R::Set{NTuple{4, Int64}} = Set()
	new_distribution = zeros(Float64, problem.n_X, problem.n_Y * problem.c)
	unsplit_y = (y) -> ((y-1)%problem.c + 1, div(y-1-(y-1)%problem.c, problem.c)+1)

	for x=1:problem.n_X
		for y=1:problem.n_Y * problem.c
			new_distribution[x,y] = 1/problem.c * distribution[x, unsplit_y(y)[2]]
		end
	end
	
	V = (x,y,a,b) -> (a != unsplit_y(y)[1]) || (b - 1 == problem.f(x, unsplit_y(y)[2]))

	game = Game(problem.n_X, problem.n_Y * problem.c, problem.c, 2, V)
	prob = dot(new_distribution, optimize_entanglement(game, dim, new_distribution; impose_maximally_entangled = impose_maximally_entangled, iterations = iterations))
	return problem.c * prob - problem.c + 1
end

	
		
		
