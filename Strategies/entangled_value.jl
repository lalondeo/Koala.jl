
export EntangledStrategy, EntangledSolverData

struct EntangledStrategy <: StrategyType
	A::Dict{Tuple{Int,Int}, Matrix{Float64}}
	B::Dict{Tuple{Int,Int}, Matrix{Float64}}
	dim::Int
	
	function EntangledStrategy(n_X::Int, n_Y::Int, n_A::Int, n_B::Int, dim::Int)
		A::Dict{Tuple{Int,Int}, Matrix{Float64}} = Dict()
		B::Dict{Tuple{Int,Int}, Matrix{Float64}} = Dict()
		
		for x=1:n_X
			POVM = gen_rand_POVM(n_A, dim)
			for a=1:n_A
				A[(x,a)] = realify(POVM[a])
			end
		end
		
		for y=1:n_Y
			POVM = gen_rand_POVM(n_B, dim)
			for b=1:n_B
				B[(y,b)] = realify(POVM[b])
			end
		end
		return new(A,B,dim)
	end	
	
	function EntangledStrategy(game::Game, dim::Int) # Generates a protocol at random
		EntangledStrategy(game.n_X, game.n_Y, game.n_A, game.n_B)
	end
		
	function EntangledStrategy(A,B,dim)
		return new(A,B,dim)
	end
end


function scramble_strategy!(strategy::EntangledStrategy, game::Game)
	dim = div(size(strategy.A[(1,1)], 1), 2)
	for x=1:game.n_X
		if(rand() < 0.5)
			POVM = gen_rand_POVM(game.n_A, dim)
			for a=1:game.n_A
				strategy.A[(x,a)] = 0.5 * strategy.A[(x,a)] + 0.5 * realify(POVM[a])
			end
		end
	end
	
	for y=1:game.n_Y
		if(rand() < 0.5)
			POVM = gen_rand_POVM(game.n_B, dim)
			for b=1:game.n_B
				strategy.B[(y,b)] = 0.5 * strategy.B[(y,b)] + 0.5 * realify(POVM[b])
			end
		end
	end
end

""" 
	copyto!(strat1::Entanglement_strategy, strat2::Entanglement_strategy)

Copies the contents of strat2 into strat1. Assumes that strat1 and strat2 are for games of the same format.
"""
function copyto!(strat1::EntangledStrategy, strat2::EntangledStrategy)
	for p in keys(strat1.A)
		strat1.A[p] .= strat2.A[p]
	end
	
	for p in keys(strat1.B)
		strat1.B[p] .= strat2.B[p]
	end
end

function evaluate_success_probabilities!(game::Game, strategy::EntangledStrategy, success_probabilities::Matrix{Float64})
	success_probabilities .= 0.0
	for (x,y,a,b) in game.R
		success_probabilities[x,y] += tr(strategy.A[(x,a)] * strategy.B[(y,b)]) / strategy.dim / 2
	end
end

function evaluate_success_probability(game::Game, strategy::EntangledStrategy, distribution::Matrix{Float64})
	return sum(distribution[x,y] * tr(strategy.A[(x,a)] * strategy.B[(y,b)]) / strategy.dim / 2 for (x,y,a,b) in game.R)

end


struct EntangledSolverData <: InternalSolverDataType
	SDP_A::Model
	A::Dict{Tuple{Int, Int}, Symmetric{VariableRef, Matrix{VariableRef}}}
	SDP_B::Model
	B::Dict{Tuple{Int, Int}, Symmetric{VariableRef, Matrix{VariableRef}}}
	
	function EntangledSolverData(game::Game, dim::Int; eps_abs::Float64 = 1e-06, trace_lower_bound::Float64 = 1.0, impose_maximally_entangled = false)
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
		if(impose_maximally_entangled)
			for x=1:game.n_X
				@constraint(SDP_A, sum(A[(x,a)] for a=1:game.n_A) .== diagm([1.0 for i=1:2*dim]))
			end	
		else
			tau = @variable(SDP_A, [1:dim, 1:dim], PSD)
			@constraint(SDP_A, tr(tau) == dim)
			for x=1:game.n_X
				@constraint(SDP_A, sum(A[(x,a)] for a=1:game.n_A)[1:dim, 1:dim] .== tau)
				@constraint(SDP_A, sum(A[(x,a)] for a=1:game.n_A)[1:dim, dim+1 : end] .== 0)
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
	#@objective(data.SDP_A, Max, sum(distribution[x,y] * (((x,y,a,b) in game.R) ? tr(strategy.B[(y,b)] * data.A[(x,a)]) : 0) for a=1:game.n_A for b=1:game.n_B for x=1:game.n_X for y=1:game.n_Y))
	@objective(data.SDP_A, Max, sum(distribution[x,y] * tr(data.A[(x,a)] * strategy.B[(y,b)]) for (x,y,a,b) in game.R))
	optimize!(data.SDP_A)
	for x=1:game.n_X
		for a=1:game.n_A
			strategy.A[(x,a)] .= JuMP.value.(data.A[(x,a)])
		end
	end
	
	@objective(data.SDP_B, Max, sum(distribution[x,y] * tr(strategy.A[(x,a)] * data.B[(y,b)]) for (x,y,a,b) in game.R))
	optimize!(data.SDP_B)
	for y=1:game.n_Y
		for b=1:game.n_B
			strategy.B[(y,b)] .= JuMP.value.(data.B[(y,b)])
		end
	end	
end


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