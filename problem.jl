module Problem
export Problem

abstract type Problem end

struct Game <: Problem
	n_X::Int64
	n_Y::Int64
	n_A::Int64
	n_B::Int64
	R::Set{NTuple{4, Int64}}
	
	function Game(n_X::Int, n_Y::Int, n_A::Int, n_B::Int, V)
		R::Set{NTuple{4, Int64}} = Set()
		for x=1:n_X
			for y=1:n_Y
				for a=1:n_A
					for b=1:n_B
						if(V(x,y,a,b))
							push!(R, (x,y,a,b))
						end
					end
				end
			end
		end
		new(n_X, n_Y, n_A, n_B, R)
	end
end


struct OneWayCommunicationProblem <: Problem
	n_X::Int64
	n_Y::Int64
	c::Int64
	f::Dict{Tuple{Int,Int}, Bool} # The value of the function
	promise::Dict{Tuple{Int,Int}, Bool}
	function OneWayCommunicationProblem(n_X::Int, n_Y::Int, c::Int, f <: Function)
		new(n_X, n_Y, c, Dict([((x,y), f(x,y)) for x=1:n_X for y=1:n_Y]), [true for x=1:n_X for y=1:n_Y])
	end
	
	function OneWayCommunicationProblem(n_X::Int, n_Y::Int, c::Int, f::Function, promise::Function)
		new(n_X, n_Y, c, Dict([((x,y), f(x,y)) for x=1:n_X for y=1:n_Y]), [promise(x,y) for x=1:n_X for y=1:n_Y])
	end
end

function encode_char(tab::Array{Bool})::Char
	return(Char(128+sum(2^(7-i) * tab[i] for i=1:length(tab))))
end

function decode_char(_c::Char)::Array{Bool}
	c = Int(_c)
	return [((c & 2^(7-i)) != 0) for i=1:7]
end
	
function encode_binary_array(tab::Array{Bool})::String
	return string([encode_char(tab[i:i+6]) for i=1:7:length(tab)]...)
end

function decode_binary_array(str::String, n::Int)::Array{Bool}
	return vcat([decode_char(c) for c in str]...)[1:n]
end

function encode_game(G::Game)::String
	tab = [false for i=1:G.n_X * G.n_Y * G.n_A * G.n_B]
	i = 1
	for x=1:G.n_X
		for y=1:G.n_Y
			for a=1:G.n_A
				for b=1:G.n_B
					tab[i] = (x,y,a,b) in G.R
					i += 1
				end
			end
		end
	end
	return encode_binary_array(tab)
end

function decode_game(str::String, n_X, n_Y, n_A, n_B)::Game
	tab = decode_binary_array(str, n_X*n_Y*n_A*n_B)
	R::Set{NTuple{4, Int64}} = Set()
	k = 1
	for x=1:n_X
		for y=1:n_Y
			for a=1:n_A
				for b=1:n_B
					if(tab[k])
						push!(R, (x,y,a,b))
					end
					k += 1
				end
			end
		end
	end
	return Game(n_X, n_Y, n_A, n_B, R)
end


function uniform_distribution(n_X::Int, n_Y::Int; promise = (x,y) -> true)
	distribution = zeros(n_X, n_Y)
	for x=1:n_X
		for y=1:n_Y
			distribution[x,y] = promise(x,y)
		end
	end
	distribution ./= sum(distribution)
	return distribution
end


