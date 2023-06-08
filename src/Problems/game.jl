export Game, encode_game, decode_game, MagicSquareGame, CHSH, coloring_game

using Combinatorics

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

######################### Code for finding a distinguished representative of a given game ######################### 



function build_iterator(tab::Vector{Int32})
	dictionary = Tuple{Int32,Int32}[]
	for i=1:length(tab)
		push!(dictionary, (i, tab[i]))
	end
	
	sort!(dictionary, by = (x) -> x[2])
	permutation_iterators = [];
	i = 1;
	while(i <= length(tab))
		value = dictionary[i][2]
		if(i == length(tab) || value != dictionary[i+1][2])
			push!(permutation_iterators, dictionary[i][1])
			i += 1
		else
			indices = [dictionary[i][1]]
			while(i != length(tab) && dictionary[i][2] == dictionary[i+1][2])
				push!(indices, dictionary[i+1][1])
				i += 1
			end
			push!(permutation_iterators, permutations(indices))
		end
	end
	
	return Iterators.product(permutation_iterators...)
end
	

function flatten_permutation!(perms, perm::Vector{Int32})
	empty!(perm)
	for x in perms
		for y in x
			push!(perm, y)
		end
	end
end

function invert_permutation!(original_permutation::Vector{Int32}, target_permutation::Vector{Int32})
	for i=1:length(original_permutation)
		target_permutation[original_permutation[i]] = i
	end
end
	
struct RepresentativeInfo
	vals_X::Vector{Int32}
	vals_Y::Vector{Int32}
	vals_A::Vector{Int32}
	vals_B::Vector{Int32}
	tmp_i::Vector{Int32}
	tmp_j::Vector{Int32}
	joint_tab::Matrix{Int32}
	tmp_X::Vector{Int32}
	tmp_Y::Vector{Int32}
	tmp_A::Vector{Int32}
	tmp_B::Vector{Int32}
	perm_X::Vector{Int32}
	perm_Y::Vector{Int32}
	perm_A::Vector{Int32}
	perm_B::Vector{Int32}
	best_perm_X::Vector{Int32}
	best_perm_Y::Vector{Int32}
	best_perm_A::Vector{Int32}
	best_perm_B::Vector{Int32}
	function RepresentativeInfo(n_X::Int64, n_Y::Int64, n_A::Int64, n_B::Int64)
		N = max(n_X, n_Y, n_A, n_B);
		new(zeros(Int32, n_X), zeros(Int32, n_Y), zeros(Int32, n_A), zeros(Int32, n_B), zeros(Int32, N), zeros(Int32, N), zeros(Int32, N, N), 
		zeros(Int32, n_X), zeros(Int32, n_Y), zeros(Int32, n_A), zeros(Int32, n_B), zeros(Int32, n_X), zeros(Int32, n_Y), zeros(Int32, n_A), zeros(Int32, n_B), 
		zeros(Int32, n_X), zeros(Int32, n_Y), zeros(Int32, n_A), zeros(Int32, n_B))
	end
			
end

function find_distinguished_representative(game::Game, info::RepresentativeInfo)::Game
	info.vals_X .= 0
	info.vals_Y .= 0
	info.vals_A .= 0
	info.vals_B .= 0
	
	vals = (info.vals_X, info.vals_Y, info.vals_A, info.vals_B)

	sizes = (game.n_X, game.n_Y, game.n_A, game.n_B)
	N = max(game.n_X, game.n_Y, game.n_A, game.n_B)

	for i=1:4
		for j=i+1:min(4,i+2)
			info.joint_tab .= 0
			for A in game.R
				info.joint_tab[A[i],A[j]] += 1
			end
			
			info.tmp_i .= 1
			info.tmp_j .= 1
			
			for v=1:sizes[i]
				for w=1:sizes[j]
					hashed_value = hash(info.joint_tab[v,w])
					info.tmp_i[v] = (info.tmp_i[v] * hashed_value) % 16777216
					info.tmp_j[w] = (info.tmp_j[w] * hashed_value) % 16777216
				end
			end
			
			for v=1:sizes[i]
				vals[i][v] += info.tmp_i[v]
			end
			
			for w=1:sizes[j]
				vals[j][w] += info.tmp_j[w]
			end
			
		end
	end

	iterator_X = build_iterator(info.vals_X)
	iterator_Y = build_iterator(info.vals_Y)
	iterator_A = build_iterator(info.vals_A)
	iterator_B = build_iterator(info.vals_B)
	best_score::Int64 = -1
	
	for p_x in iterator_X
		flatten_permutation!(p_x, info.tmp_X)
		invert_permutation!(info.tmp_X, info.perm_X)
		for p_y in iterator_Y
			flatten_permutation!(p_y, info.tmp_Y)
			invert_permutation!(info.tmp_Y, info.perm_Y)
			for p_a in iterator_A
				flatten_permutation!(p_a, info.tmp_A)
				invert_permutation!(info.tmp_A, info.perm_A)
				for p_b in iterator_B
					flatten_permutation!(p_b, info.tmp_B)
					invert_permutation!(info.tmp_B, info.perm_B)
					score = 0
					for (x,y,a,b) in game.R
						score += hash((info.perm_X[x], info.perm_Y[y], info.perm_A[a], info.perm_B[b]))
					end
					score = score % 4294967296
					if(score > best_score)
						best_score = score
						info.best_perm_X .= info.perm_X
						info.best_perm_Y .= info.perm_Y
						info.best_perm_A .= info.perm_A
						info.best_perm_B .= info.perm_B
					end
				end
			end
		end
	end
			#return game
	R = NTuple{4, Int64}[];
	for (x,y,a,b) in game.R
		push!(R, (info.best_perm_X[x], info.best_perm_Y[y], info.best_perm_A[a], info.best_perm_B[b]))
	end
	
	return Game(game.n_X, game.n_Y, game.n_A, game.n_B, Set(R))
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
				




