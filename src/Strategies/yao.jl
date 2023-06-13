export YaoProtocol, YaoSolverData, generate_hard_distribution_yao, optimize_yao

######################### Protocol definition and helper functions #########################


struct YaoProtocol <: StrategyType
	states::Vector{Matrix{Float64}}
	POVMs::Vector{Pair{Matrix{Float64}, Matrix{Float64}}} # Corresponding to the probability of outputting zero and one
	function YaoProtocol(n_X::Int64, n_Y::Int64, C::Int64)
		new([realify(gen_rho(C)) for x=1:n_X], [Pair(realify.(gen_rand_POVM(2,C))...) for y=1:n_Y])
	end
	
	function YaoProtocol(problem::Problems.OneWayCommunicationProblem) # Generates a protocol at random
		YaoProtocol(problem.n_X, problem.n_Y, problem.C)
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

function evaluate_success_probabilities!(problem::Problems.OneWayCommunicationProblem, protocol::YaoProtocol, success_probabilities::Matrix{Float64})
	for x=1:problem.n_X
		for y=1:problem.n_Y
			if(problem.promise[x,y])
				success_probabilities[x,y] = tr(protocol.states[x] * protocol.POVMs[y][problem.f[x,y]+1]) / 2.0
			else
				success_probabilities[x,y] = -1000000
			end
		end
	end
end

function evaluate_success_probability(problem::Problems.OneWayCommunicationProblem, protocol::YaoProtocol, distribution::Matrix{Float64})::Float64
	tot = 0.0
	for x=1:problem.n_X
		for y=1:problem.n_Y
			if(problem.promise[x,y])
				tot += distribution[x,y] * tr(protocol.states[x] * protocol.POVMs[y][problem.f[x,y]+1]) / 2.0
			end
		end
	end
	return tot
end

function scramble_strategy!(protocol::YaoProtocol, problem::Problems.OneWayCommunicationProblem)
	for x=1:problem.n_X
		if(rand() < 0.5)
			new_state = realify(gen_rho(problem.C))
			protocol.states[x] = 0.5 * protocol.states[x] + 0.5 * new_state
		end
	end
		
	for y=1:problem.n_Y
		if(rand() < 0.5)
			new_POVM = gen_rand_POVM(2, problem.C)
			for i=1:2
				protocol.POVMs[y][i] .= 0.5 * protocol.POVMs[y][i] + realify(new_POVM[i])
			end
		end
	end
end

######################### Solver data object #########################

struct YaoSolverData <: InternalSolverDataType
	SDP_states::Model
	states::Vector{Symmetric{VariableRef, Matrix{VariableRef}}}
	SDP_POVMs
	POVMs::Vector{Pair{Symmetric{VariableRef, Matrix{VariableRef}}}}
	
	function YaoSolverData(n_X::Int64, n_Y::Int64, dim::Int64; eps_abs = 1e-05)
					
		SDP_states = JuMP.Model(SDP_solver(eps_abs));	
		set_silent(SDP_states)
		states = [@variable(SDP_states, [1:2*dim, 1:2*dim], PSD) for x=1:n_X]
		
		for x=1:n_X
			@constraint(SDP_states, sum(states[x][i,i] for i=1:dim) == 1) # Force the trace to be one
			enforce_SDP_constraints(SDP_states, states[x])
		end
		
		SDP_POVMs = JuMP.Model(SDP_solver(eps_abs));	
		set_silent(SDP_POVMs)
		
		POVMs = [Pair(@variable(SDP_POVMs, [1:2*dim, 1:2*dim], PSD), @variable(SDP_POVMs, [1:2*dim, 1:2*dim], PSD)) for y=1:n_Y]
		Id = diagm([1.0 for i=1:2*dim])
		for y=1:n_Y
			@constraint(SDP_POVMs, POVMs[y][1] + POVMs[y][2] .== Id)
			enforce_SDP_constraints(SDP_POVMs, POVMs[y][1])
			enforce_SDP_constraints(SDP_POVMs, POVMs[y][2])
		end
		new(SDP_states, states, SDP_POVMs, POVMs)
	end
	
	function YaoSolverData(problem::Problems.OneWayCommunicationProblem; eps_abs = 1e-05)
		YaoSolverData(problem.n_X, problem.n_Y, problem.C; eps_abs = eps_abs)
	end
	
end

######################### Optimization functions #########################

function improve_strategy!(problem::Problems.OneWayCommunicationProblem, protocol::YaoProtocol, distribution::Matrix{Float64}, data::YaoSolverData)
	@objective(data.SDP_states, Max, sum(problem.promise[x,y] ? (distribution[x,y] * tr(protocol.POVMs[y][problem.f[x,y]+1] * data.states[x])) : 0 for x=1:problem.n_X for y=1:problem.n_Y))
	optimize!(data.SDP_states)
	for x=1:problem.n_X
		protocol.states[x] .= JuMP.value.(data.states[x])
	end
	
	@objective(data.SDP_POVMs, Max, sum(problem.promise[x,y] ? (distribution[x,y] * tr(data.POVMs[y][problem.f[x,y]+1] * protocol.states[x])) : 0 for x=1:problem.n_X for y=1:problem.n_Y))
	optimize!(data.SDP_POVMs)
	for y=1:problem.n_Y
		for i=1:2
			protocol.POVMs[y][i] .= JuMP.value.(data.POVMs[y][i])
		end
	end	
end

yao_warning = false
"""
	optimize_yao(game::Problems.Game, distribution::Matrix{Float64}; iterations = 50)::Float64
	
Given a game and a distribution on the inputs, returns a lower bound on the best achievable winning probability classically. If this function
is to be called repeatedly, consider builidng the ClassicalStrategy and ClassicalSolverData objects and calling optimize_strategy! directly. """
function optimize_yao(problem::Problems.OneWayCommunicationProblem, distribution::Matrix{Float64}; iterations = 50)::Float64
	global yao_warning	
	if(!(yao_warning))
		@warn "If you are calling this function multiple times during the execution of your program, consider building your own YaoProtocol and YaoSolverData objects\
				and calling optimize_strategy! instead, as this will put less pressure on the garbage collector. "
		yao_warning = true
	end
	strategy = YaoProtocol(problem)
	data = YaoSolverData(problem)
	optimize_strategy!(problem, strategy, distribution, data; max_iter = iterations)
	return evaluate_success_probability(problem, strategy, distribution)
end

######################### Distributional things #########################


function generate_hard_distribution_yao(problem::Problems.OneWayCommunicationProblem, protocol::YaoProtocol, data::YaoSolverData; kwargs...)
	function enforce_promise(model::Model, D::Matrix{VariableRef})
		for x=1:problem.n_X
			for y=1:problem.n_Y
				if(!(problem.promise[x,y]))
					@constraint(model, D[x,y] == 0)
				end
			end
		end
	end
	
	return generate_hard_distribution(problem, protocol, data; additional_constraints = enforce_promise, kwargs...)
end

function generate_hard_distribution_yao(problem::Problems.OneWayCommunicationProblem; kwargs...)
	protocol = YaoProtocol(problem)
	data = YaoSolverData(problem)
	return generate_hard_distribution_yao(problem, protocol, data; kwargs...)
end
				
