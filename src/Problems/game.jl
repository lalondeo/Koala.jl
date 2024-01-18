export Game, encode_game, decode_game, random_game, MagicSquareGame, CHSH, TiltedCHSH, build_tilted_CHSH_distribution, coloring_game, G_13, G_14
import Base.copyto!

struct Game <: ProblemType
	n_X::Int64
	n_Y::Int64
	n_A::Int64
	n_B::Int64
	R::Array{Bool, 4}
	function Game(n_X::Int, n_Y::Int, n_A::Int, n_B::Int, V::Function)
		R = zeros(Bool, n_X, n_Y, n_A, n_B)
		for x=1:n_X
			for y=1:n_Y
				for a=1:n_A
					for b=1:n_B
						R[x,y,a,b] = V(x,y,a,b)
					end
				end
			end
		end
		new(n_X, n_Y, n_A, n_B, R)
	end
	
	function Game(R::Array{Bool, 4})
		new(size(R)..., R)
	end
	
end

function copyto!(game_i::Game, game_f::Game)
	@assert game_i.n_X == game_f.n_X
	@assert game_i.n_Y == game_f.n_Y
	@assert game_i.n_A == game_f.n_A
	@assert game_i.n_B == game_f.n_B
	
	game_f.R .= game_i.R
end

"""
	random_game(n_X::Int, n_Y::Int, n_A::Int, n_B::Int; p::Float64 = 0.5)

Given the dimensions of the game and a probability p, returns a randomly chosen game with these dimensions where every answer is taken to win the game independently with probability p.
"""
function random_game(n_X::Int, n_Y::Int, n_A::Int, n_B::Int; p::Float64 = 0.5)::Game
	return Game(n_X, n_Y, n_A, n_B, (x,y,a,b) -> rand() < p)
end


# function encode_game(G::Game)::String
	# tab = [false for i=1:G.n_X * G.n_Y * G.n_A * G.n_B]
	# i = 1
	# for x=1:G.n_X
		# for y=1:G.n_Y
			# for a=1:G.n_A
				# for b=1:G.n_B
					# tab[i] = G.R[x,y,a,b]
					# i += 1
				# end
			# end
		# end
	# end
	# return encode_binary_array(tab)
# end

# function decode_game(str::String, n_X, n_Y, n_A, n_B)::Game
	# tab = decode_binary_array(str, n_X*n_Y*n_A*n_B)
	# R::Set{NTuple{4, Int64}} = Set()
	# k = 1
	# for x=1:n_X
		# for y=1:n_Y
			# for a=1:n_A
				# for b=1:n_B
					# if(tab[k])
						# push!(R, (x,y,a,b))
					# end
					# k += 1
				# end
			# end
		# end
	# end
	# return Game(n_X, n_Y, n_A, n_B, R)
# end



### Examples ###

# The mother of all nonlocal games, the CHSH game
const CHSH = Game(2,2,2,2, (x,y,a,b) -> ((x==2)&&(y==2)) == (a!=b))


function V_magic_square_game(x,y,a,b)
   bits = [[0;0],[0;1],[1;0],[1;1]]
   column = [bits[a]; sum(bits[a]) % 2];
   row = [bits[b]; (sum(bits[b]) + 1) % 2];
   return column[y] == row[x]
end

# The classic magic square game of Mermin and Peres
const MagicSquareGame = Game(3,3,4,4,V_magic_square_game)


R = zeros(Bool, 2, 4, 2, 2)

for x=1:2
	y = 1
	for _y=1:2
		for c=1:2
			if(c == 1)
				bit = ((x==2) && (_y == 2))
				for a=1:2
					R[x,y,a, (a + bit) % 2 + 1] = true
				end
			else
				R[x,y,1,1] = true
				R[x,y,1,2] = true
			end
			
			y += 1
		end
	end
end

# The tilted CHSH game, built from the tilted CHSH inequalities first introduced in "Randomness versus Nonlocality and Entanglement" by Acin, Massar and Pironio (2012)
const TiltedCHSH = Game(R)

"""
	build_tilted_CHSH_distribution(p::Float64)::Matrix{Float64}
	
For a parameter p between 0 and 1, returns the corresponding distribution for the tilted CHSH game (TiltedCHSH), introduced by Acin, Massar and Pironio (2012). p = 0 corresponds to the usual CHSH game; p = 1 corresponds
to a game that's winnable with probability 1 classically; and all choices in-between yield games for which the optimal value cannot be attained with a maximally entangled state, although
the gap is generally quite small. Choosing p = 0.8 will yield a game with value approximately 80.5% but with maximally entangled value approximately 80%. """
function build_tilted_CHSH_distribution(p::Float64)::Matrix{Float64}
	dist = zeros(2,4)
	for x=1:2
		y = 1
		for _y=1:2
			for c=1:2
				dist[x,y] = 1/4 * (c==1 ? p : (1-p))
				y += 1
			end
		end
	end
	return dist
end

"""
	CHSH_n_CMMN(n::Int64)::Game

Returns the nth game from the family of games introduced in 'A generalization of CHSH and the algebraic structure of optimal strategies' by Cui, Mehta, Mousavi and Nezhadi (2020), which
generalize the usual CHSH game. The original CHSH game is the special case where n = 2. The nth game has binary input sets and output sets of size n, which should be greater or equal to two.
As in the case of the tilted CHSH games, for n > 3, the maximally entangled value of the game is always lesser than the entangled value. n = 3 has an entangled value of roughly 79% and maximally entangled value of roughly 77%. 
"""
function CHSH_n_CMMN(n::Int64)::Game
	V = (x,y,a,b) -> ((x==2) && (y == 2) && (((a - b) % n) == 1)) || (((x!=2) || (y != 2)) && (((a - b) % n) == 0))
	return Game(2,2,n,n,V)
end
	
function tilted_CHSH_n_CMMN(n::Int64)::Game
	V = (x,y,a,b) -> (y >= 3) ? (a == 1) : (((x==2) && (y == 2) && (((a - b) % n) == 1)) || (((x!=2) || (y != 2)) && (((a - b) % n) == 0)))
	return Game(2,4,n,n,V)
end	


			
"""
	coloring_game(G::Matrix{Bool}, k::Int64)::Game

Given a graph G and the number of colors k, builds the corresponding coloring game ('On the chromatic number of a graph' by Cameron et al., 2007) and the distribution which is uniform on legal inputs. """
function coloring_game(G::Matrix{Bool}, k::Int64)::Tuple{Game, Matrix{Float64}}
	n = size(G,2)
	R = zeros(Bool, n, n, k, k)
	distribution = ones(n,n)
	for x=1:n
		for y=1:n
			if(x==y)
				for c=1:k
					R[x,y,c,c] = true
				end
				
			elseif(G[x,y])
				for c1=1:k
					for c2=1:k
						if((c1 != c2))
							R[x,y,c1,c2] = true
						end
					end
				end
				
			else
				distribution[x,y] = 0.0
			end
		end
	end
	
	distribution ./= sum(distribution)
	
	return Game(R), distribution
end



Vs = [[1;0;0],[0;1;0],[0;0;1],[1;1;0],[1;-1;0],[1;0;1],[1;0;-1],[0;1;1],[0;1;-1],[1;1;1],[1;1;-1],[1;-1;1],[-1;1;1]]

# G_{13} from "Oddities of quantum colorings" by Mancinska and Roberson
G_13 = zeros(Bool, 13, 13);
for i=1:13
	for j=1:13
		if(dot(Vs[i], Vs[j]) == 0)
			G_13[i,j] = true
		end
	end
end

# G_{14} from the same source
const G_14 = zeros(Bool, 14, 14);
for i=1:13
	for j=1:13
		G_14[i,j] = G_13[i,j]
	end
	G_14[i,14] = true
	G_14[14,i] = true
end