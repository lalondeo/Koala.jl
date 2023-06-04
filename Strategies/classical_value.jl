
struct ClassicalStrategy <: StrategyType
	outputs_Alice::Vector{Int64}
	outputs_Bob::Vector{Int64}
	
	function ClassicalStrategy(game::Game)
		new(rand(1:game.n_A, game.n_X), rand(1:game.n_B, game.n_Y))
	end
end

function scramble_strategy!(strategy::ClassicalStrategy, game::Game)
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


function copyto!(strat1::ClassicalStrategy, strat2::ClassicalStrategy)
	strat1.outputs_Alice .= strat2.outputs_Alice
	strat1.outputs_Bob .= strat2.outputs_Bob
end

function evaluate_success_probabilities!(game::Game, strategy::ClassicalStrategy, success_probabilities::Matrix{Float64})
	for x=1:game.n_X
		for y=1:game.n_Y
			success_probabilities[x,y] = (x,y,strategy.outputs_Alice[x],strategy.outputs_Bob[y]) in game.R
		end
	end
end

function evaluate_success_probability(game::Game, strategy::ClassicalStrategy, distribution::Matrix{Float64})
	return sum(distribution[x,y] * ((x,y,strategy.outputs_Alice[x],strategy.outputs_Bob[y]) in game.R) for x=1:game.n_X for y=1:game.n_Y)
end

struct ClassicalSolverData <: InternalSolverDataType
	T_x::Matrix{Float64}
	T_y::Matrix{Float64}
	
	function ClassicalSolverData(game::Game)
		new(zeros(game.n_X, game.n_A), zeros(game.n_Y, game.n_B))
	end
end

function improve_strategy!(game::Game, strategy::ClassicalStrategy, distribution::Matrix{Float64}, data::ClassicalSolverData)
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


	
