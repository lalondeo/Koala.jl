include("game.jl")

export OneWayCommunicationProblem, gameify, EQ, promise_equality

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
	R::Set{NTuple{4, Int64}} = Set()
	for x=1:problem.n_X	
		for a=1:game.C
			y = 1
			for (_y, b) in Iterators.product(1:game.n_Y, 1:game.C)
				if(a != b)
					push!(R, (x,y,a,1))
					push!(R, (x,y,a,2))
				else
					push!(R, (x,y,a,problem.f[x,_y] + 1))
				end
				y += 1
			end
		end
	end
	return Game(game.n_X, game.n_Y*game.C, game.C, 2, R)
end


### Examples ###
function EQ(N, c)
	return OneWayCommunicationProblem(N, N, c, (x,y) -> (x==y))
end

function promise_equality(G::Matrix{Float64}, c::Int64)
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
			