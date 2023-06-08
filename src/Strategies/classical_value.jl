
export ClassicalStrategy, ClassicalSolverData

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
			success_probabilities[x,y] = (x,y,strategy.outputs_Alice[x],strategy.outputs_Bob[y]) in game.R
		end
	end
end

function evaluate_success_probability(game::Problems.Game, strategy::ClassicalStrategy, distribution::Matrix{Float64})::Float64
	return sum(distribution[x,y] * ((x,y,strategy.outputs_Alice[x],strategy.outputs_Bob[y]) in game.R) for x=1:game.n_X for y=1:game.n_Y)
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
	
	function ClassicalSolverData(n_X::Int64, n_Y::Int64, n_A::Int64, n_B::Int64)
		new(zeros(n_X, n_A), zeros(n_Y, n_B))
	end
	
	
	function ClassicalSolverData(game::Problems.Game)
		ClassicalSolverData(game.n_X, game.n_Y, game.n_A, game.n_B)
	end
	
end


######################### Optimization functions #########################


function improve_strategy!(game::Problems.Game, strategy::ClassicalStrategy, distribution::Matrix{Float64}, data::ClassicalSolverData)
	data.T_x .= 0
	
	for (x,y,a,b) in game.R
		if(b == strategy.outputs_Bob[y])
			data.T_x[x,a] += distribution[x,y]
		end
	end
	
	for x=1:game.n_X
		strategy.outputs_Alice[x] = argmax(data.T_x[x,:])
	end
	
	data.T_y .= 0
	
	for (x,y,a,b) in game.R
		if(a == strategy.outputs_Alice[x])
			data.T_y[y,b] += distribution[x,y]
		end
	end
	
	for y=1:game.n_Y
		strategy.outputs_Bob[y] = argmax(data.T_y[y,:])
	end
end