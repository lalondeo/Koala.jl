include("game.jl")

export OneWayCommunicationProblem, gameify, convert_distribution!, convert_distribution, EQ, promise_equality, CHSH_n_comm

struct OneWayCommunicationProblem <: ProblemType
	n_X::Int64
	n_Y::Int64
	C::Int64
	f::Matrix{Bool}
	promise::Matrix{Bool}
	function OneWayCommunicationProblem(n_X::Int, n_Y::Int, C::Int, f:: Function)
		new(n_X, n_Y, C, [f(x,y) for x=1:n_X, y=1:n_Y], [true for x=1:n_X, y=1:n_Y])
	end
	
	function OneWayCommunicationProblem(n_X::Int, n_Y::Int, C::Int, f::Function, promise::Function)
		new(n_X, n_Y, C, [f(x,y) for x=1:n_X,  y=1:n_Y], [promise(x,y) for x=1:n_X, y=1:n_Y])
	end
	
	function OneWayCommunicationProblem(n_X::Int64, n_Y::Int64, C::Int64, f::Matrix{Bool}, promise::Matrix{Bool})
		new(n_X, n_Y, C, f, promise)
	end
	
end


""" 
	gameify(problem::OneWayCommunicationProblem)::Game

Given a one-way communication complexity problem, converts it into a nonlocal game. If the register size in the original problem is C and in any of the classical/entangled settings, 
any protocol for the communication problem with success probability p corresponds to a strategy in the nonlocal game with success probability (C-1)/C + 1/C * p, and conversely.
"""
function gameify(problem::OneWayCommunicationProblem)::Game
	R = zeros(Bool, problem.n_X, problem.n_Y * problem.C, problem.C, 2)
	for x=1:problem.n_X	
		for a=1:problem.C
			y = 1
			for (_y, b) in Iterators.product(1:problem.n_Y, 1:problem.C)
				if(a != b)
					R[x,y,a,1] = R[x,y,a,2] = true
				else
					R[x,y,a,problem.f[x,_y] + 1] = true
				end
				y += 1
			end
		end
	end
	return Game(R)
end

"""
	convert_distribution!(problem::OneWayCommunicationProblem, distribution_i::Matrix{Float64}, distribution_f::Matrix{Float64})

Given a distribution distribution_i for the problem (which is a n_X times n_Y matrix), converts it into a distribution distribution_f (which is a n_X times (n_Y * C) matrix) for
the gameified version of the problem.
"""
function convert_distribution!(problem::OneWayCommunicationProblem, distribution_i::Matrix{Float64}, distribution_f::Matrix{Float64})
	for x=1:problem.n_X
		y = 1
		for (_y, c) in Iterators.product(1:problem.n_Y, 1:problem.C)
			distribution_f[x,y] = distribution_i[x,_y] / problem.C
			y += 1
		end
	end
end

"""
	convert_distribution(problem::OneWayCommunicationProblem, distribution_i::Matrix{Float64})::Matrix{Float64}

Given a distribution distribution for the problem (which is a n_X times n_Y matrix), returns a distribution for the gameified 
"""
function convert_distribution(problem::OneWayCommunicationProblem, distribution_i::Matrix{Float64})::Matrix{Float64}
	distribution_f = zeros(problem.n_X, problem.n_Y * problem.C)
	convert_distribution!(problem, distribution_i, distribution_f)
	return distribution_f
end



### Examples ###

"""
	The venerable equality function. N is the cardinality of the inputs and c is the register size of the communication. """
function EQ(N, c)
	return OneWayCommunicationProblem(N, N, c, (x,y) -> (x==y))
end

""" First defined by de Wolf in his PhD thesis. Given a graph G, encodes the following problem: Alice and Bob are given vertices of G with the promise that they are either
equal or adjacent and they must decide which of the two is the case. """
function promise_equality(G::Matrix{Bool}, c::Int64)::OneWayCommunicationProblem
	n = size(G, 1)
	f = zeros(Bool, n, n)
	promise = zeros(Bool, n, n)
	for x=1:n
		for y=1:n
			if(x == y)
				promise[x,y] = true
				f[x,y] = 1
			elseif(G[x,y])
				promise[x,y] = true
				f[x,y] = 0
			else
				promise[x,y] = false
			end
		end
	end
	return OneWayCommunicationProblem(n, n, c, f, promise)
end

function CHSH_n_comm(n::Int64)::OneWayCommunicationProblem
	@assert n >= 2
	edges = []
	X = []
	for i=1:n
		push!(X, (i,0))
		push!(X, (i,1))
		for j=1:n
			if(i != j)
				push!(edges, (i,j))
			end
		end
	end

	promise = (x,y) -> X[x][1] in edges[y]
	f = (x,y) -> (((edges[y][1] < edges[y][2]) ? (X[x][1] == edges[y][1]) : 1) + X[x][2]) % 2
	return OneWayCommunicationProblem(2*n, n * (n-1), 2, f, promise)
end
	
	
			