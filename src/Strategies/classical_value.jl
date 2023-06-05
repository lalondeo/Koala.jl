
export ClassicalStrategy, ClassicalSolverData, HasPerfectClassicalStrategyInfo, has_perfect_classical_strategy!

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

######################### Code specifically for checking whether a given game has a perfect classical strategy #########################

struct HasPerfectClassicalStrategyInfo
	legal_outputs_alice::Matrix{Bool}
	legal_outputs_bob::Matrix{Bool}
	history::Array{Tuple{Int,Int,Int}}
	branching::Array{Int}
	function HasPerfectClassicalStrategyInfo(game::Problems.Game)
		new(zeros(Bool, game.n_X, game.n_A), zeros(Bool, game.n_Y, game.n_B), [], [])
	end
end

function has_perfect_classical_strategy!(game::Problems.Game, strategy::ClassicalStrategy, info::HasPerfectClassicalStrategyInfo)::Bool
	info.legal_outputs_alice .= true
	info.legal_outputs_bob .= true
	strategy.outputs_Alice .= 0
	strategy.outputs_Bob .= 0
	empty!(info.history)
	empty!(info.branching)
	
	if(game.n_A <= game.n_B)
		push!(info.branching, 1)
	else
		push!(info.branching, game.n_X + 1)
	end
	k = 1
	while(k > 0 && k <= game.n_X + game.n_Y)
		if(k > length(info.branching))
			best_score = 1000
			best_var = -1
			for x=1:game.n_X
				if(strategy.outputs_Alice[x] == 0)
					number_legal = sum(info.legal_outputs_alice[x,:])
					if(number_legal < best_score)
						best_var = x
						best_score = number_legal
					end
				end
			end
			
			for y=1:game.n_Y
				if(strategy.outputs_Bob[y] == 0)
					number_legal = sum(info.legal_outputs_bob[y,:])
					if(number_legal < best_score)
						best_var = y + game.n_X
						best_score = number_legal
					end
				end
			end
			@assert best_var != -1
			push!(info.branching, best_var)
				
		end
		
		var = info.branching[k]
		if(var <= game.n_X)
			# Alice
			x = var
			if(strategy.outputs_Alice[x] != 0)
				while(!(isempty(info.history)) && k == info.history[end][1])
					tuple = pop!(info.history)
					info.legal_outputs_bob[tuple[2],tuple[3]] = true
				end
			end
			strategy.outputs_Alice[x] += 1
			while(strategy.outputs_Alice[x] <= game.n_A && !(info.legal_outputs_alice[x,strategy.outputs_Alice[x]]))
				strategy.outputs_Alice[x] += 1
			end
			
			
			if(strategy.outputs_Alice[x] > game.n_A)
				pop!(info.branching)
				strategy.outputs_Alice[x] = 0
				k -= 1
				continue
			end			
			
			for y=1:game.n_Y
				if(strategy.outputs_Bob[y] == 0)
					for b=1:game.n_B
						if(info.legal_outputs_bob[y,b] && !((x,y,strategy.outputs_Alice[x],b) in game.R))
							push!(info.history, (k, y, b))
							info.legal_outputs_bob[y,b] = false
						end
					end
				end
			end
			k += 1
			
		else
			# Bob
			y = var - game.n_X
			if(strategy.outputs_Bob[y] != 0)
				while(!(isempty(info.history)) && k == info.history[end][1])
					tuple = pop!(info.history)
					info.legal_outputs_alice[tuple[2],tuple[3]] = true
				end
			end
			
			strategy.outputs_Bob[y] += 1
			while(strategy.outputs_Bob[y] <= game.n_B && !(info.legal_outputs_bob[y,strategy.outputs_Bob[y]]))
				strategy.outputs_Bob[y] += 1
			end
			
			if(strategy.outputs_Bob[y] > game.n_B)
				pop!(info.branching)
				strategy.outputs_Bob[y] = 0
				k -= 1
				continue
			end

			for x=1:game.n_X
				if(strategy.outputs_Alice[x] == 0)
					for a=1:game.n_A
						if(info.legal_outputs_alice[x,a] && !((x,y,a,strategy.outputs_Bob[y]) in game.R))
							push!(info.history, (k, x, a))
							info.legal_outputs_alice[x,a] = false
						end
					end
				end
			end
			k += 1
		end
	end
	
	return k != 0
end
		
	


	
