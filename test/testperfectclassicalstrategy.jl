using Random

##### Checks that in case a perfect strategy is reported to be found, the strategy output is actually valid, as well as a consistency check #####
for i=1:5000
	Ns = rand(2:7, 4)
	info = Koala.Strategies.HasPerfectClassicalStrategyInfo(Ns...)
	strategy = Koala.Strategies.ClassicalStrategy(Ns...)
	threshold = rand()
	game = Koala.Problems.Game(Ns..., (x,y,a,b) -> rand() < threshold)
	answer = Koala.Strategies.has_perfect_classical_strategy!(game, strategy, info)
	if(answer)
		@test Koala.Strategies.evaluate_success_probability(game, strategy, ones(Ns[1], Ns[2]) / Ns[1] / Ns[2]) > 1 - 1e-08
	end
	temp = deepcopy(game)
	for i=1:10
		Koala.Problems.permute_game!(game, temp, [shuffle(collect(1:Ns[i])) for i=1:4]...)
		@test Koala.Strategies.has_perfect_classical_strategy!(temp, strategy, info) == answer
	end
		
end
	
	
##### Checks that has_perfect_classical_strategy works properly for coloring games #####

# Backtracking code for checking whether a graph is k-colorable. Written by ChatGPT.
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


# For a bunch of graphs, computes the chromatic number chi, checks that the coloring game with chi colors is winnable and the coloring game with chi-1 colors is not
for i=1:10000
	N = rand(5:12)
	G = zeros(Bool, N, N)
	for i=1:N
		for j=i+1:N
			G[i,j] = G[j,i] = rand(0:1)
		end
	end
	chi = N
	while(is_graph_k_colorable(G, chi))
		chi -= 1
	end
	chi += 1
	game = Koala.Problems.coloring_game(G, chi)
	@test Koala.Strategies.has_perfect_classical_strategy(game)
	if(chi > 1)
		game2 = Koala.Problems.coloring_game(G, chi - 1)
		@test !(Koala.Strategies.has_perfect_classical_strategy(game2))
	end
end

