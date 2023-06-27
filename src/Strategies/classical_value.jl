
export ClassicalStrategy, ClassicalSolverData, optimize_classical_strategy

######################### Strategy definition and helper functions #########################


struct ClassicalStrategy <: StrategyType
	outputs_Alice::Vector{Int64}
	outputs_Bob::Vector{Int64}
	
	function ClassicalStrategy(n_X::Int64, n_Y::Int64, n_A::Int64, n_B::Int64)
		new(rand(1:n_A, n_X), rand(1:n_B, n_Y))
	end
	
	function ClassicalStrategy(game::Problems.Game)
		ClassicalStrategy(game.n_X, game.n_Y, game.n_A, game.n_B)
	end
	
end

function copyto!(strat1::ClassicalStrategy, strat2::ClassicalStrategy)
	strat1.outputs_Alice .= strat2.outputs_Alice
	strat1.outputs_Bob .= strat2.outputs_Bob
end


function evaluate_success_probabilities!(game::Problems.Game, strategy::ClassicalStrategy, success_probabilities::Matrix{Float64})
	for x=1:game.n_X
		for y=1:game.n_Y
			success_probabilities[x,y] = game.R[x,y,strategy.outputs_Alice[x],strategy.outputs_Bob[y]]
		end
	end
end

function evaluate_success_probability(game::Problems.Game, strategy::ClassicalStrategy, distribution::Matrix{Float64})::Float64
	return sum(distribution[x,y] * game.R[x,y,strategy.outputs_Alice[x],strategy.outputs_Bob[y]] for x=1:game.n_X for y=1:game.n_Y)
end

function scramble_strategy!(strategy::ClassicalStrategy, game::Problems.Game)
	for x=1:game.n_X
		if(rand() < 0.5)
			strategy.outputs_Alice[x] = rand(1:game.n_A)
		end
	end
	
	for y=1:game.n_Y
		if(rand() < 0.5)
			strategy.outputs_Bob[y] = rand(1:game.n_B)
		end
	end
end






######################### Solver data object #########################


struct ClassicalSolverData <: InternalSolverDataType
	T_x::Matrix{Float64}
	T_y::Matrix{Float64}
	iterator::Base.Iterators.ProductIterator
	function ClassicalSolverData(n_X::Int64, n_Y::Int64, n_A::Int64, n_B::Int64)
		new(zeros(n_X, n_A), zeros(n_Y, n_B), Iterators.product(1:n_X, 1:n_Y, 1:n_A, 1:n_B))
	end
	
	
	function ClassicalSolverData(game::Problems.Game)
		ClassicalSolverData(game.n_X, game.n_Y, game.n_A, game.n_B)
	end
	
end


######################### Optimization functions #########################


function improve_strategy!(game::Problems.Game, strategy::ClassicalStrategy, distribution::Matrix{Float64}, data::ClassicalSolverData)
	data.T_x .= 0
	
	for (x,y,a,b) in data.iterator
		if(game.R[x,y,a,b] && b == strategy.outputs_Bob[y])
			data.T_x[x,a] += distribution[x,y]
		end
	end
	
	for x=1:game.n_X
		strategy.outputs_Alice[x] = argmax(data.T_x[x,:])
	end
	
	data.T_y .= 0
	
	for (x,y,a,b) in data.iterator
		if(game.R[x,y,a,b] && a == strategy.outputs_Alice[x])
			data.T_y[y,b] += distribution[x,y]
		end
	end
	
	for y=1:game.n_Y
		strategy.outputs_Bob[y] = argmax(data.T_y[y,:])
	end
end

classical_warning = false


"""
	optimize_classical_strategy(game::Problems.Game, distribution::Matrix{Float64})::Float64
	
Given a game and a distribution on the inputs, returns a lower bound on the best achievable winning probability classically. If this function
is to be called repeatedly, consider builidng the ClassicalStrategy and ClassicalSolverData objects and calling optimize_strategy! directly. """
function optimize_classical_strategy(game::Problems.Game, distribution::Matrix{Float64}; kwargs...)::Float64
	global classical_warning	
	if(!(classical_warning))
		@warn "If you are calling this function multiple times during the execution of your program, consider building your own ClassicalStrategy and ClassicalSolverData objects\
				and calling optimize_strategy! instead, as this will put less pressure on the garbage collector. "
		classical_warning = true
	end
	strategy = ClassicalStrategy(game)
	data = ClassicalSolverData(game)
	optimize_strategy!(game, strategy, distribution, data; kwargs...)
	return evaluate_success_probability(game, strategy, distribution)
end