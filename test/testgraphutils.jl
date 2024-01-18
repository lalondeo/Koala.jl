using Random
Random.seed!(1)

############### Test graph6 ###############


for i=1:1000
	global G
	G = random_graph(rand(20:40), 0.3)
	@test parse_graph6(write_graph6(G)) == G
end

############# Graph parameters ###########




############# Clique stuff #############
for _=1:1000
	G = random_graph(rand(20:70), rand() * 0.4)
	
	_posited_maximal_cliques = list_maximal_cliques(G)
	
	for clique::Vector{Int} in _posited_maximal_cliques # Check every subset to make sure that it is indeed a clique
		@test issorted(clique) && is_clique(G, clique)
	end
	
	# Test maximum clique: check that it is a clique and that it is of maximum size wrt the cliques in _posited_maximal_cliques
	maximum_clique = find_maximum_clique(G)
	@test issorted(maximum_clique)
	@test is_clique(G, maximum_clique)
	@test maximum(length, _posited_maximal_cliques) == length(maximum_clique)
	
	# Check that there are no repetitions
	posited_maximal_cliques = Set(_posited_maximal_cliques)
	@test length(posited_maximal_cliques) == length(_posited_maximal_cliques)
	@test maximum_clique in posited_maximal_cliques
	
	# Cross-check the list produced by list_maximal_cliques with randomly generated maximal cliques
	for _=1:100
		random_maximal_clique = generate_maximal_clique(G)
		sort!(random_maximal_clique)
		@test random_maximal_clique in posited_maximal_cliques
	end
	
	
end

########## Colorings #############

# The following is horrible but correct code for checking whether a graph is k-colorable that was written by ChatGPT. Used only to cross-check has_k_coloring.
function is_graph_k_colorable(adjacency_matrix, k)
    num_vertices = size(adjacency_matrix, 1)
    coloring = fill(0, num_vertices)

    function is_safe(vertex, color)
        for i in 1:num_vertices
            if adjacency_matrix[vertex, i] == 1 && coloring[i] == color
                return false
            end
        end
        return true
    end

    function graph_coloring_util(vertex)
        if vertex == num_vertices + 1
            return true
        end

        for color in 1:k
            if is_safe(vertex, color)
                coloring[vertex] = color

                if graph_coloring_util(vertex + 1)
                    return true
                end

                coloring[vertex] = 0
            end
        end

        return false
    end

    return graph_coloring_util(1)
end

for _=1:100
	# G is taken to be smaller than previously so that ChatGPT's horrible code can handle it. 
	global G
	G = random_graph(rand(15:30), rand() * 0.4)
	maximum_clique = find_maximum_clique(G)

	chi = compute_chromatic_number(G)
	@test has_k_coloring(G, chi; clique = maximum_clique) && is_graph_k_colorable(G, chi)
	@test !(has_k_coloring(G, chi-1; clique = maximum_clique)) && !(is_graph_k_colorable(G, chi-1))	
end

########## pick_subgraph #############
 
for _=1:10
	G = random_graph(50, 0.4)
	n = rand(5:10)
	G2 = random_graph(n, 0.5)
	if(pick_subgraph(G, G2) != nothing)
		for _=1:1000
			S = pick_subgraph(G,G2)
			ok = true
			for i=1:n
				for j=i+1:n
					ok &= (!G2[i,j]) || G[S[i],S[j]]
				end
			end
			@test ok
		end
	end
end

######### compute_invariant #############
perm = collect(1:50)
for _=1:100
	G = random_graph(50, 0.4)
	invariant = compute_invariant(G)
	for _=1:10
		shuffle!(perm)
		@test invariant == compute_invariant(G[perm, perm])
	end
end
	
	
	

	