export parse_graph6, write_graph6, is_clique, generate_maximal_clique, find_good_maximal_clique, find_maximum_clique, list_maximal_cliques, get_largest_clique,
	has_k_coloring, compute_chromatic_number, test_edge_criticality, generate_edge_critical_graph, pick_subgraph, compute_invariant, random_graph

############### Graph6 ###############

"""
	parse_graph6(str::String)::Matrix{Bool}

Given a graph6 representation of a graph, returns its adjacency matrix.
"""
function parse_graph6(str::String)::Matrix{Bool}
	n = Int32(str[1]) - 63
	
	G = zeros(Bool, n, n)
	k::Int32 = 2
	l::Int32 = 5
	m::Int32 = Int32(str[2])-63
	for i::Int32=1:n
		for j::Int32=1:i-1
			if((m & 2^l) != 0)
				G[i,j] = G[j,i] = 1 
			end
			if(l == 0)
				l = 5
				k += 1
				if(k <= length(str))
					m = Int32(str[k]) - 63
				end
			else
				l -= 1
			end
		 end
	end
	return G
end	

"""
	write_graph6(G::Matrix{Bool})::str{String}

Given the adjacency matrix of a graph, returns its graph6 representation
"""
function write_graph6(G::Matrix{Bool})::String
	n = size(G)[1]
	str = string(Char(63+n))
	l = 5
	m = 0
	for i=1:n
		for j=1:i-1
			if(G[i,j])
				m |= 2^l
			end
			
			if(l == 0 || (i == n && j == n - 1))
				l = 5
				str *= Char(m + 63)
				m = 0
			else
				l -= 1
			end
		end
	end
	return str
end

############### Code pertaining to cliques ###############
"""
	is_clique(G::Matrix{Bool}, clique::Vector{Int})::Bool

Given the adjacency matrix of a graph and a set of vertices of G, checks if that set is indeed a clique of G
"""
function is_clique(G::Matrix{Bool}, S::Vector{Int})::Bool
	l = length(S)
	for i=1:l
		for j=i+1:l
			if(!G[S[i],S[j]])
				return false
			end
		end
	end
	return true
end

"""
	generate_maximal_clique(G::Matrix{Bool})::Vector{Int}

Given the adjacency matrix of a graph, generates a maximal clique of G at random.
"""
function generate_maximal_clique(G::Matrix{Bool})::Vector{Int}
	n = size(G)[1]
	clique = Int[]
	for v in shuffle(1:n)
		ok = true
		for v2 in clique
			ok &= G[v,v2]
		end
		if(ok)
			push!(clique, v)
		end
	end
	return clique
end


"""
	find_good_maximal_clique(G::Matrix{Bool}; N::Int64 = 100)::Vector{Int}

Given the adjacency matrix of a graph, generates N maximal cliques of G using generate_maximal_clique and returns the largest one.
"""
function find_good_maximal_clique(G::Matrix{Bool}; N::Int64 = 20)::Vector{Int}
	clique = Int[1]
	for i=1:N
		new_clique = generate_maximal_clique(G)
		if(length(new_clique) > length(clique))
			clique = new_clique
		end
	end
	sort!(clique)
	return clique
end

"""
	find_maximum_clique(G::Matrix{Bool})::Vector{Int}
	
Given the adjacency matrix of a graph, returns a clique of G that's as large as possible. 
"""
function find_maximum_clique(G::Matrix{Bool})::Vector{Int}
	n = size(G,1)
	max_clique = find_good_maximal_clique(G; N = 5);
	best_size = length(max_clique)
	current_clique = Int[0]
	degrees = G * ones(Int, n)
	while(!isempty(current_clique))
		current_clique[end] += 1
		while(current_clique[end] <= n)
			ok = degrees[current_clique[end]] >= best_size - 1
			
			for i=1:length(current_clique)-1
				ok &= G[current_clique[i], current_clique[end]]
			end
			
			if(ok)
				break
			end
			current_clique[end] += 1
			
		end
		
		if(current_clique[end] > n)
			pop!(current_clique)
		else
			ok = true
			if(length(current_clique) > best_size)
				empty!(max_clique)
				append!(max_clique, current_clique)
				best_size = length(max_clique)
				
				for i=1:length(current_clique)
					if(degrees[current_clique[i]] < best_size - 1)
						ok = false
						filter!((x)->x >current_clique[i], current_clique)
						break
					end
				end
			end
			if(ok)
				push!(current_clique, current_clique[end])
			end
		end
	end
	return max_clique

end

"""
	compute_clique_number(G::Matrix{Bool})::Int

Given the adjacency matrix of a graph, computes its clique number. """
function compute_clique_number(G::Matrix{Bool})::Int
	return length(find_maximum_clique(G))
end
			

"""
   list_maximal_cliques(G::Matrix{Bool})::Vector{Vector{Int}}
   
   Given a graph G, returns the list of all maximal cliques of G, i.e. cliques of G that aren't subsets of strictly larger cliques of G.

"""
function list_maximal_cliques(G::Matrix{Bool})::Vector{Vector{Int}}
	n = size(G)[1]
	if(sum(G) == n^2 - n)
		return [[i for i=1:n]]
	end
	maximal_cliques = [] 
	
	i = 2
	index = [k <= 2 ? 1 : 0 for k=1:n]
	found_extension = [false for i=1:n]
	while(i > 0)
		if(index[i] >= n)
			if(!found_extension[i])
				clique = index[1:i-1]
				is_maximal = true
				for j=1:n
					is_extension = true
					for x in clique
						is_extension &= G[j,x]
					end
					
					if(is_extension)
						is_maximal = false
						break
					end
				end
				
				if(is_maximal)
					push!(maximal_cliques, clique)
				end
				
			end
			
			index[i] = 0
			found_extension[i] = false
			i-= 1
			continue
		end
			
		index[i] += 1
		ok = true
		for j=1:i-1
			ok &= G[index[j], index[i]]
		end
		
		if(ok)	
			index[i+1] = index[i]
			for j=1:i
				found_extension[j] = true
			end
			i += 1
		end
		
	end
	return maximal_cliques
end

function get_largest_clique(G::Matrix{Bool})::Vector{Int}
	return argmax((x)->length(x), list_maximal_cliques(G))
end

############### Code pertaining to (classical) graph colorings ###############


"""
	has_k_coloring(G::Matrix{Bool}, k::Int; clique::Vector{Int} = [])::Bool

Given the adjacency matrix of a graph and a number of colors, returns whether the graph can be colored classically with k colors. If a clique of the graph is not provided, one will be generated using
find_good_maximal_clique.
"""
function has_k_coloring(G::Matrix{Bool}, k::Int; clique::Vector{Int} = Int[])::Bool
	n = size(G,1)
	degrees = G * ones(Bool, n);
	
	if(isempty(clique))
		clique = find_good_maximal_clique(G)
	elseif(length(clique) == n)
		return k >= n
	elseif(length(clique) > k)
		return false
	end

	min_i = length(clique) + 1
	colors = [0 for i=1:n];
	is_colored = [0 for i=1:n];
	for i=1:length(clique)
		colors[clique[i]] = i
		is_colored[clique[i]] = 1
	end
	
	branches = [0 for i=1:n];
	current_branch = 1;
	while(current_branch > 0 && 0 in colors)
		vertex = branches[current_branch]
		if(vertex == 0)
			vertex = argmax((1000 * (G * is_colored) .+ degrees) .* (1 .- is_colored) .- is_colored)
			branches[current_branch] = vertex
			is_colored[vertex] = true
			colors[vertex] = 0
		end
		
		colors[vertex] += 1
		
		while(colors[vertex] <= k)
			ok = true
			for vertex2=1:n
				ok &= (!G[vertex,vertex2] || colors[vertex] != colors[vertex2])
			end
			if(ok)
				break
			else
				colors[vertex] += 1
			end
		end
		
		if(colors[vertex] == k+1)
			colors[vertex] = 0
			is_colored[vertex] = 0
			branches[current_branch] = 0
			current_branch -= 1
		else
			current_branch += 1
		end
		
	end
	return current_branch != 0
	
end

""" 
	compute_chromatic_number(G::Matrix{Bool}; clique::Vector{Int} = [])

Given the adjacency matrix of a graph, returns its chromatic number.
"""
function compute_chromatic_number(G::Matrix{Bool}; clique::Vector{Int} = Int[])::Int
	if(isempty(clique))
		clique = find_good_maximal_clique(G)
	end
	
	chi = length(clique)
	while(!(has_k_coloring(G, chi; clique = clique)))
		chi += 1
	end
	return chi
end

"""
	test_edge_criticality(G::Matrix{Bool})::Bool

Checks whether the graph is edge-critical. 
"""

function test_edge_criticality(G::Matrix{Bool})::Bool
	n = size(G,1)
	if(0 in G * ones(Int, n)) 
		return false # Isolated vertex
	end
	
	clique = get_largest_clique(G)
	chi = compute_chromatic_number(G; clique = clique)

	ok = true
	for v1=1:n
		if(!ok) break end
		
		for v2=(v1+1):n
			if(!ok) break end
			
			if(G[v1,v2])
				G[v1,v2] = G[v2,v1] = false
				_clique = clique
				if(v1 in clique && v2 in clique)
					_clique = filter((x)->x != v1, clique)
				end
				ok &= has_k_coloring(G, chi-1; clique = _clique)
				G[v1,v2] = G[v2,v1] = true
			end
		end
	end
	return ok
end

"""
	generate_edge_critical_graph(n::Int, k::Int)::Matrix{Bool}

Given a number of vertices n and a chromatic number k, attempts to generate an edge-k-critical graph on n vertices at random. Returns Nothing in case of failure. 
"""
function generate_edge_critical_graph(n::Int, k::Int)::Union{Matrix{Bool}, Nothing}
	@assert n > k
	G = ones(Bool, n, n)
	for i=1:n G[i,i] = 0 end
	order = collect(1:n)
	one_vector = ones(Bool, n)
	degrees = zeros(Int, n)
	
	clique = collect(1:n)
	while(true)
		@assert is_clique(G, clique)

		shuffle!(order)
		mul!(degrees, G, one_vector)
		deleted_edge = false
		for i=1:n
			u = order[i]
			if(degrees[u] == k-1) continue end
			for j=i+1:n
				v = order[j]
				if(degrees[v] == k-1 || !G[u,v]) continue end

				is_edge_in_clique = u in clique && v in clique
				
				if(length(clique) < k || (length(clique) == k && is_edge_in_clique))
					# Check that deleting (u,v) from the graph won't make the graph (k-1)-colorable
					G[u,v] = G[v,u] = false
					if(has_k_coloring(G, k-1; clique = (is_edge_in_clique ? filter((x)->x != u, clique) : clique)))
						G[u,v] = G[v,u] = true
					else
						deleted_edge = true
						if(is_edge_in_clique)
							clique = find_good_maximal_clique(G)
						end
						@goto out_of_loop
					end
					
				else
					G[u,v] = G[v,u] = false
					deleted_edge = true
					if(is_edge_in_clique)
						clique = find_good_maximal_clique(G)
					end
					@goto out_of_loop
				end
			end
		end
		@label out_of_loop
		if(!deleted_edge) break end
	end

	if(test_edge_criticality(G))
		return G
	else
		return nothing
	end
end

############### Subgraph stuff ###############

# order is a random permutation of 1:size(G,1)
function _pick_subgraph(G::Matrix{Bool}, G2::Matrix{Bool}, vertex_list::Vector{Int}, order::Vector{Int})::Nothing
	j = length(vertex_list) + 1
	for i in order
		if(i in vertex_list) 
			continue 
		end
		
		ok = true
		for k=1:j-1
			ok &= (G2[k,j] == 0) || (G[vertex_list[k], i] == 1)
		end
		
		if(ok)
			push!(vertex_list, i)
			if(j == size(G2, 1))
				return 
			else
				_pick_subgraph(G, G2, vertex_list, order)
				if(length(vertex_list) == size(G2, 1))
					return
				end
			end
			pop!(vertex_list);
		end
	end
end


"""
	pick_subgraph(G::Matrix{Bool}, G2::Matrix{Bool})::Vector{Int}

Given graphs G and G2, with G2 having fewer vertices than G, either returns a random list S of cardinality |V(G_2)| such that G_2 is a subgraph of G[S,S] if such a list exists and an empty list otherwise.
Simple backtracking algorithm, not designed to be efficient on large graphs.  
"""
function pick_subgraph(G::Matrix{Bool}, G2::Matrix{Bool})::Vector{Int}
	@assert size(G2,1) <= size(G,1) && size(G2,1) > 0 
	vertex_list = Int[]
	_pick_subgraph(G, G2, vertex_list, shuffle(1:size(G,1)))
	return vertex_list
end
	
############### Miscellaneous ###############

"""
	compute_invariant(G::Matrix{Bool})::UInt64

Given the adjacency matrix of a graph, returns an unsigned integer. This function is invariant under isomorphisms of G, and will likely return different values for nonisomorphic graphs. 
"""
function compute_invariant(G::Matrix{Bool})::UInt64
	n = size(G,1)
	hashes::Vector{Int64} = Int64[];
	row = zeros(Int, n)
	Gpow::Matrix{Int64} = G^(n + (n %2 == 0 ? 1 : 0)) # make sure that the exponent is odd, because otherwise this function will return the same value for two nonisomorphic graphs that square to the same thing
	for i=1:n
		row .= Gpow[:,i]
		sort!(row)
		push!(hashes, hash(row) >> 10)
	end
	sort!(hashes)
	return hash(hashes)
end

"""
	random_graph(n::Int, p::Float64)::Matrix{Bool}

Generate a random graph on n vertices, in which any two distinct vertices are adjacent with probability p. """
function random_graph(n::Int, p::Float64)::Matrix{Bool}
	G = zeros(Bool, n, n)
	for i=1:n
		for j=i+1:n
			G[i,j] = G[j,i] = rand() < p
		end
	end
	return G
end


		
		
		
		


