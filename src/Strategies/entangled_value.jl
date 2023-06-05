export EntangledStrategy, EntangledSolverData

######################### Strategy definition and helper functions #########################

struct EntangledStrategy <: StrategyType
	A::Dict{Tuple{Int,Int}, Matrix{Float64}}
	B::Dict{Tuple{Int,Int}, Matrix{Float64}}
	dim::Int
	
	"""
		EntangledStrategy(n_X::Int, n_Y::Int, n_A::Int, n_B::Int, dim::Int)
	
	Given input size n_X and n_Y and output size n_A and n_B as well as the local dimension dim, returns a random strategy, i.e.
	n_X POVMs of n_A elements each for Alice and n_Y POVMs of n_B elements each for Bob, all over C^n. The POVMs are represented in 
	real form because solvers typically do not handle complex numbers. """
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

	"""
		EntangledStrategy(n_X::Int, n_Y::Int, n_A::Int, n_B::Int, dim::Int)
	
	Given a game as well as the local dimension dim, returns a random strategy, i.e.
	n_X POVMs of n_A elements each for Alice and n_Y POVMs of n_B elements each for Bob, all over C^n. The POVMs are represented in 
	real form because solvers typically do not handle complex numbers. The actual contents of the game is irrelevant, all that matters are the input/output dimensions."""
	function EntangledStrategy(game::Problems.Game, dim::Int) # Generates a protocol at random
		EntangledStrategy(game.n_X, game.n_Y, game.n_A, game.n_B, dim)
	end
	
	function EntangledStrategy(A,B,dim)
		return new(A,B,dim)
	end
end

function copyto!(strat1::EntangledStrategy, strat2::EntangledStrategy)
	for p in keys(strat1.A)
		strat1.A[p] .= strat2.A[p]
	end
	
	for p in keys(strat1.B)
		strat1.B[p] .= strat2.B[p]
	end
end


function evaluate_success_probabilities!(game::Problems.Game, strategy::EntangledStrategy, success_probabilities::Matrix{Float64})
	success_probabilities .= 0.0
	for (x,y,a,b) in game.R
		success_probabilities[x,y] += tr(strategy.A[(x,a)] * strategy.B[(y,b)]) / strategy.dim / 2
	end
end

function evaluate_success_probability(game::Problems.Game, strategy::EntangledStrategy, distribution::Matrix{Float64})::Float64
	return sum(distribution[x,y] * tr(strategy.A[(x,a)] * strategy.B[(y,b)]) / strategy.dim / 2 for (x,y,a,b) in game.R)

end

function scramble_strategy!(strategy::EntangledStrategy, game::Problems.Game)
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

######################### Solver data object #########################

struct EntangledSolverData <: InternalSolverDataType
	SDP_A::Model
	A::Dict{Tuple{Int, Int}, Symmetric{VariableRef, Matrix{VariableRef}}}
	SDP_B::Model
	B::Dict{Tuple{Int, Int}, Symmetric{VariableRef, Matrix{VariableRef}}}
	
	"""
		EntangledSolverData(n_X::Int64, n_Y::Int64, n_A::Int64, n_B::Int64, dim::Int; eps_abs::Float64 = 1e-06, trace_lower_bound::Float64 = 1.0, impose_maximally_entangled = false)
	
	Builds a solver data object for finding good strategies with local dimension dim for nonlocal games with input sizes n_X and n_Y and output sizes n_A and n_B. The keywords arguments are as follows:
		- eps_abs: the tolerance for the SDP solver
		- trace_lower_bound: A lower bound that is imposed on the traces of the POVMs. This was observed to help a lot with finding the best strategy in some cases.
		- impose_maximally_entangled: whether the joint state should be forced to be maximally entangled. """
	function EntangledSolverData(n_X::Int64, n_Y::Int64, n_A::Int64, n_B::Int64, dim::Int; eps_abs::Float64 = 1e-06, trace_lower_bound::Float64 = 1.0, impose_maximally_entangled = false)
		### Building SDP_A ###
		SDP_A = JuMP.Model(SDP_solver(eps_abs));
		set_silent(SDP_A)
		A = Dict()
		for x=1:n_X
			for a=1:n_A
				A[(x,a)] = @variable(SDP_A, [1:2*dim, 1:2*dim], PSD)
				enforce_SDP_constraints(SDP_A, A[(x,a)])
				@constraint(SDP_A, tr(A[(x,a)]) >= trace_lower_bound)
			end
		end
		if(impose_maximally_entangled)
			for x=1:n_X
				@constraint(SDP_A, sum(A[(x,a)] for a=1:n_A) .== diagm([1.0 for i=1:2*dim]))
			end	
		else
			tau = @variable(SDP_A, [1:dim, 1:dim], PSD)
			@constraint(SDP_A, tr(tau) == dim)
			for x=1:n_X
				@constraint(SDP_A, sum(A[(x,a)] for a=1:n_A)[1:dim, 1:dim] .== tau)
				@constraint(SDP_A, sum(A[(x,a)] for a=1:n_A)[1:dim, dim+1 : end] .== 0)
			end
		end
		
		### Building SDP_B ###
		Id = diagm([1 for i=1:dim])
		SDP_B = JuMP.Model(SDP_solver(eps_abs));
		set_silent(SDP_B)
		B = Dict()
		for y=1:n_Y
			for b=1:n_B
				B[(y,b)] = @variable(SDP_B, [1:2*dim, 1:2*dim], PSD)
				enforce_SDP_constraints(SDP_B, B[(y,b)])
				@constraint(SDP_B, tr(B[(y,b)]) >= trace_lower_bound)
			end
		end
		
		for y=1:game.n_Y
			@constraint(SDP_B, sum(B[(y,b)] for b=1:n_B) .== diagm([1.0 for i=1:2*dim]))
		end
		return new(SDP_A, A, SDP_B, B)
	end
	
	function EntangledSolverData(game::Problems.Game, dim::Int; eps_abs::Float64 = 1e-06, trace_lower_bound::Float64 = 1.0, impose_maximally_entangled = false)
		new(game.n_X, game.n_Y, game.n_A, game.n_B, dim; eps_abs = eps_abs, trace_lower_bound = trace_lower_bound, impose_maximally_entangled = impose_maximally_entangled)
	end
	
end	

######################### Optimization functions #########################

function improve_strategy!(game::Problems.Game, strategy::EntangledStrategy, distribution::Matrix{Float64}, data::EntangledSolverData)
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

	# game = Problems.Game(problem.n_X, problem.n_Y * problem.c, problem.c, 2, V)
	# prob = dot(new_distribution, optimize_entanglement(game, dim, new_distribution; impose_maximally_entangled = impose_maximally_entangled, iterations = iterations))
	# return problem.c * prob - problem.c + 1
# end