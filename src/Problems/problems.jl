
export ProblemType, uniform_distribution
abstract type ProblemType end

include("game.jl")
include("communication_problem.jl") 
include("isomorphism.jl")


""" 
	uniform_distribution(n_X::Int, n_Y::Int; promise = (x,y) -> true)

Returns a uniform distribution over all inputs for which the promise is true.
"""
function uniform_distribution(n_X::Int, n_Y::Int; promise = (x,y) -> true)::Matrix{Float64}
	distribution = zeros(n_X, n_Y)
	for x=1:n_X
		for y=1:n_Y
			distribution[x,y] = promise(x,y)
		end
	end
	distribution ./= sum(distribution)
	return distribution
end



