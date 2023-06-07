export Game, encode_game, decode_game, MagicSquareGame, CHSH, coloring_game

struct Game <: ProblemType
	n_X::Int64
	n_Y::Int64
	n_A::Int64
	n_B::Int64
	R::Set{NTuple{4, Int64}}
	function Game(n_X::Int, n_Y::Int, n_A::Int, n_B::Int, V::Function)
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
	
	function Game(n_X::Int, n_Y::Int, n_A::Int, n_B::Int, R::Set{NTuple{4, Int64}})
		new(n_X, n_Y, n_A, n_B, R)
	end
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


### Examples ###
function V_magic_square_game(x,y,a,b)
   bits = [[0;0],[0;1],[1;0],[1;1]]
   column = [bits[a]; sum(bits[a]) % 2];
   row = [bits[b]; (sum(bits[b]) + 1) % 2];
   return column[y] == row[x]
end

const MagicSquareGame = Game(3,3,4,4,V_magic_square_game)

const CHSH = Game(2,2,2,2, (x,y,a,b) -> ((x==2)&&(y==2)) == (a!=b))

function tilted_CHSH(p)
	R::Set{NTuple{4, Int64}} = Set()
	dist = zeros(2,4)
	for x=1:2
		y = 1
		for _y=1:2
			for c=1:2
				dist[x,y] = 1/4 * (c==1 ? p : (1-p))
				if(c == 1)
					bit = ((x==2) && (_y == 2))
					for a=1:2
						push!(R, (x, y, a, (a + bit) % 2 + 1))
					end
				else
					push!(R, (x,y,1,1))
					push!(R, (x,y,1,2))
				end
				
				y += 1
			end
		end
	end
	return (Game(2,4,2,2,R), dist)
end
			
"""
	coloring_game(G::Matrix{Bool}, C::Int64)::Game

Given a graph G and the number of colors C, builds the corresponding coloring game """
function coloring_game(G::Matrix{Bool}, C::Int64)::Game
	R::Set{NTuple{4, Int64}} = Set()
	n = size(G,2)
	for x=1:n
		for y=1:n
			if(x==y)
				for c=1:C
					push!(R, (x,y,c,c))
				end
			else
				for c1=1:C
					for c2=1:C
						if(!(G[x,y]) || (c1 != c2))
							push!(R, (x,y,c1,c2))
						end
					end
				end
			end
		end
	end
	return Game(n, n, C, C, R)
end
				




