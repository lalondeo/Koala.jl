using Random

##### Checks that in case a perfect strategy is reported to be found, the strategy output is actually valid, as well as a consistency check #####
for i=1:5000
	Ns = rand(2:7, 4)
	info = HasPerfectClassicalStrategyInfo(Ns...)
	strategy = ClassicalStrategy(Ns...)
	game = random_game(Ns...; p = rand())
	answer = has_perfect_classical_strategy!(game, strategy, info)
	if(answer)
		@test evaluate_success_probability(game, strategy, ones(Ns[1], Ns[2]) / Ns[1] / Ns[2]) > 1 - 1e-08
	end
	temp = deepcopy(game)
	for i=1:10
		permute_game!(game, temp, [shuffle(collect(1:Ns[i])) for i=1:4]...)
		@test has_perfect_classical_strategy!(temp, strategy, info) == answer
	end
		
end
	
	
##### Checks that has_perfect_classical_strategy works properly for coloring games #####


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
	while(has_k_coloring(G, chi))
		chi -= 1
	end
	chi += 1
	game = coloring_game(G, chi)
	@test has_perfect_classical_strategy(game)
	if(chi > 1)
		game2 = coloring_game(G, chi - 1)
		@test !(has_perfect_classical_strategy(game2))
	end
end

