using Combinatorics

export HasPerfectClassicalStrategyInfo, has_perfect_classical_strategy!, has_perfect_classical_strategy, game_is_critical!, game_is_critical

struct HasPerfectClassicalStrategyInfo
	legal_outputs_alice::Matrix{Bool}
	legal_outputs_bob::Matrix{Bool}
	history::Vector{Tuple{Int,Int,Int}}
	branching::Vector{Int}
	function HasPerfectClassicalStrategyInfo(n_X::Int64, n_Y::Int64, n_A::Int64, n_B::Int64)
		new(zeros(Bool, n_X, n_A), zeros(Bool, n_Y, n_B), [], [])
	end
	
	function HasPerfectClassicalStrategyInfo(game::Game)
		HasPerfectClassicalStrategyInfo(game.n_X, game.n_Y, game.n_A, game.n_B)
	end
end


"""
	has_perfect_classical_strategy!(game::Game, strategy::ClassicalStrategy, info::HasPerfectClassicalStrategyInfo)::Bool

Given a game, a strategy object for said game (the contents of which are irrelevant) and an instance of HasPerfectClassicalStrategyInfo, determines whether
the given game has a perfect classical strategy or not. Unlike the other optimization functions in this module which are heuristic, this function is exact. If an exact strategy is found, 
it will be stored in strategy. 
 """
function has_perfect_classical_strategy!(game::Game, strategy::ClassicalStrategy, info::HasPerfectClassicalStrategyInfo)::Bool
	info.legal_outputs_alice .= true
	info.legal_outputs_bob .= true
	strategy.outputs_Alice .= 0
	strategy.outputs_Bob .= 0
	empty!(info.history)
	empty!(info.branching)
	
	# Adaptation of the standard backtracking algorithm for graph coloring
	if(game.n_A <= game.n_B)
		push!(info.branching, 1)
	else
		push!(info.branching, game.n_X + 1)
	end
	k = 1
	while(k > 0 && k <= game.n_X + game.n_Y)
		if(k > length(info.branching))
			# Choosing the next input to branch on
			# We select the input with the smallest number of legal outputs
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
						if(info.legal_outputs_bob[y,b] && !(game.R[x,y,strategy.outputs_Alice[x],b]))
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
						if(info.legal_outputs_alice[x,a] && !(game.R[x,y,a,strategy.outputs_Bob[y]]))
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



"""
	has_perfect_classical_strategy(game::Game)::Bool

Given a game, determines whether the given game has a perfect classical strategy or not. Unlike the other optimization functions in this module which are heuristic, this function is exact. """
function has_perfect_classical_strategy(game::Game)::Bool
	strategy = ClassicalStrategy(game)
	info = HasPerfectClassicalStrategyInfo(game)
	return has_perfect_classical_strategy!(game, strategy, info)
end

"""
	game_is_critical!(game::Game, strategy::ClassicalStrategy, info::HasPerfectClassicalStrategyInfo)

Given a game and the required data, determines whether the game is classically critical, i.e. if the game does not have a perfect classical strategy and 
if adding any answer to the game causes it to have a perfect classical strategy. """
function game_is_critical!(game::Game, strategy::ClassicalStrategy, info::HasPerfectClassicalStrategyInfo)
	if(has_perfect_classical_strategy!(game, strategy, info))
		return false
	end
	
	for x=1:game.n_X
		for y=1:game.n_Y
			for a=1:game.n_A
				for b=1:game.n_B
					if(!(game.R[x,y,a,b]))
						game.R[x,y,a,b] = true
						if(!(has_perfect_classical_strategy!(game, strategy, info)))
							return false
						end
						game.R[x,y,a,b] = false
					end
				end
			end
		end
	end
	return true
end

"""
	game_is_critical(game::Game)

Given a game, determines whether the game is classically critical, i.e. if the game does not have a perfect classical strategy and 
if adding any answer to the game causes it to have a perfect classical strategy. """
function game_is_critical(game::Game)
	strategy = ClassicalStrategy(game)
	info = HasPerfectClassicalStrategyInfo(game)
	return game_is_critical!(game, strategy, info)
end