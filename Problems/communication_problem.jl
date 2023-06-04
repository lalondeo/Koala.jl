include("../utils2.jl")

include("game.jl")

export OneWayCommunicationProblem, gameify

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
end



function gameify(problem::OneWayCommunicationProblem)
	R::Set{NTuple{4, Int64}} = Set()
	for x=1:problem.n_X	
		for a=1:game.C
			y = 1
			for (_y, b) in Iterators.product(1:game.n_Y, 1:game.C)
				if(a != b)
					push!(R, (x,y,a,1))
					push!(R, (x,y,a,2))
				else
					push!(R, (x,y,a,problem.f[x,_y]))
				end
				y += 1
			end
		end
	end
	return Game(game.n_X, game.n_Y*game.C, game.C, 2, R)
end
			