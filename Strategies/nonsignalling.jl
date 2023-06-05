export NonSignallingSolverData, optimize_non_signalling!

struct NonSignallingSolverData <: InternalSolverDataType
	model::Model
	correlations::Dict{Tuple{Int,Int}, Matrix{VariableRef}}
	probabilities_alice::Matrix{VariableRef}
	probabilities_bob::Matrix{VariableRef}
	
	function NonSignallingSolverData(game::Game)
		model = Model(CPLEX.Optimizer)
		correlations = Dict()
		for x=1:game.n_X
			for y=1:game.n_Y
				correlations[(x,y)] = @variable(model, [1:game.n_X, 1:game.n_Y]; lower_bound = 0.0)
			end
		end
		
		probabilities_alice = @variable(model, [1:game.n_X, 1:game.n_A])
		probabilities_bob = @variable(model, [1:game.n_X, 1:game.n_A])
		

		for x=1:game.n_X
			for a=1:game.n_A
				for y=1:game.n_Y
					@constraint(model, sum(correlations[(x,y)][a,b] for b=1:game.n_B) == probabilities_alice[x,a])
				end
			end
		end
		
		for y=1:game.n_Y
			for b=1:game.n_B
				for x=1:game.n_X
					@constraint(model, sum(correlations[(x,y)][a,b] for a=1:game.n_A) == probabilities_bob[y,b])
				end
			end
		end
					
		new(model, correlations, probabilities_alice, probabilities_bob)
	end
end

function optimize_non_signalling!(game::Game, distribution::Matrix{Float64}, success_probabilities::Matrix{Float64}, data::NonSignallingSolverData)::Float64
	@objective(data.model, Max, sum(distribution[x,y] * data.model.correlations[(x,y)][(a,b)] for (x,y,a,b) in game.R))
	optimize!(data.model)
	success_probabilities .= 0.0
	for (x,y,a,b) in game.R
		success_probabilities[(x,y)] += data.model.correlations[(x,y)][(a,b)]
	end
end

function optimize_non_signalling(game::Game, distribution::Matrix{Float64})::Float64
	data = NonSignallingSolverData(game)
	success_probabilities = zeros(game.n_X, game.n_Y)
	return optimize_non_signalling!(game, distribution, success_probabilities, data)
end

