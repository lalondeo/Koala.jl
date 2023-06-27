export RepresentativeInfo, select_distinguished_representative!, permute_game

using Combinatorics

function build_iterator(tab::Vector{Int})
	dictionary = Tuple{Int,Int}[]
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
			i += 1
			push!(permutation_iterators, permutations(indices))
		end
	end

	
	return Iterators.product(permutation_iterators...)
end
	

function flatten_permutation!(perms, perm::Vector{Int})
	empty!(perm)
	for x in perms
		for y in x
			push!(perm, y)
		end
	end
end

function invert_permutation!(original_permutation::Vector{Int}, target_permutation::Vector{Int})
	@assert (sort(original_permutation) == collect(1:maximum(original_permutation)))

	for i=1:length(original_permutation)
		target_permutation[original_permutation[i]] = i
	end
end

function permute_game!(game_i::Game, game_f::Game, perm_X::Vector{Int}, perm_Y::Vector{Int}, perm_A::Vector{Int}, perm_B::Vector{Int})
	game_f.R .= false
	
	for x=1:game_i.n_X
		for y=1:game_i.n_Y
			for a=1:game_i.n_A
				for b=1:game_i.n_B
					if(game_i.R[x,y,a,b])
						game_f.R[perm_X[x], perm_Y[y], perm_A[a], perm_B[b]] = true
					end
				end
			end
		end
	end
end



	
struct RepresentativeInfo
	iterator::Base.Iterators.ProductIterator
	vals_X::Vector{Int}
	vals_Y::Vector{Int}
	vals_A::Vector{Int}
	vals_B::Vector{Int}
	tmp_i::Vector{Int}
	tmp_j::Vector{Int}
	joint_tab::Matrix{Int}
	tmp_X::Vector{Int}
	tmp_Y::Vector{Int}
	tmp_A::Vector{Int}
	tmp_B::Vector{Int}
	perm_X::Vector{Int}
	perm_Y::Vector{Int}
	perm_A::Vector{Int}
	perm_B::Vector{Int}
	best_perm_X::Vector{Int}
	best_perm_Y::Vector{Int}
	best_perm_A::Vector{Int}
	best_perm_B::Vector{Int}
	
	function RepresentativeInfo(n_X::Int64, n_Y::Int64, n_A::Int64, n_B::Int64)
		N = max(n_X, n_Y, n_A, n_B);
		new(Iterators.product(1:n_X, 1:n_Y, 1:n_A, 1:n_B), zeros(Int, n_X), zeros(Int, n_Y), zeros(Int, n_A), zeros(Int, n_B), zeros(Int, N), zeros(Int, N), zeros(Int, N, N), 
		zeros(Int, n_X), zeros(Int, n_Y), zeros(Int, n_A), zeros(Int, n_B), zeros(Int, n_X), zeros(Int, n_Y), zeros(Int, n_A), zeros(Int, n_B), 
		zeros(Int, n_X), zeros(Int, n_Y), zeros(Int, n_A), zeros(Int, n_B))
	end
	
	function RepresentativeInfo(game::Game)
		return RepresentativeInfo(game.n_X, game.n_Y, game.n_A, game.n_B)
	end
	
end

	


"""
	function select_distinguished_representative!(game::Game, output::Game, info::RepresentativeInfo)

Given a game and a RepresentativeInfo object, finds a game that is isomorphic to game and writes it in output. With overwhelming probability, calling this function on two distinct but
isomorphic games will return the same game. """
function select_distinguished_representative!(game::Game, output::Game, info::RepresentativeInfo)
	info.vals_X .= 0
	info.vals_Y .= 0
	info.vals_A .= 0
	info.vals_B .= 0
	
	# The first step is to assign numerical values to every input and output that are invariant under permutations of the other inputs / outputs
	# We then list all games that are isomorphic to the game in input with the property that for every input/output, the aforementioned numerical values are sorted, hash
	# each one, and output the one with the largest hash
	vals = (info.vals_X, info.vals_Y, info.vals_A, info.vals_B)

	sizes = (game.n_X, game.n_Y, game.n_A, game.n_B)
	N = max(game.n_X, game.n_Y, game.n_A, game.n_B)

	for i=1:4
		for j=i+1:min(4,i+2)
			info.joint_tab .= 0
			for r in info.iterator
				if(game.R[r...])
					info.joint_tab[r[i],r[j]] += 1
				end
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
					permute_game!(game, output, info.perm_X, info.perm_Y, info.perm_A, info.perm_B)
					score = hash(output.R) >> 1
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
	
	permute_game!(game, output, info.best_perm_X, info.best_perm_Y, info.best_perm_A, info.best_perm_B)
end