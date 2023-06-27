using Random

# Generates a bunch of games; for each game, generates 10 isomorphic versions, and checks that select_distinguished_representative! yields the same game in all cases
for i=1:1000
	Ns = [rand(2:7) for i=1:4];
	
	threshold = rand();
	random_game = Koala.Problems.Game(Ns..., (x,y,a,b) -> rand() < threshold)
	temp = deepcopy(random_game)
	info = Koala.Problems.RepresentativeInfo(random_game)
	output = deepcopy(random_game)
	Koala.Problems.find_distinguished_representative!(random_game, output, info)
	val_hash = hash(output.R)
	R2 = deepcopy(output.R)
	for j=1:9
		output.R .= false
		Koala.Problems.permute_game!(random_game, temp, [shuffle(collect(1:Ns[i])) for i=1:4]...)
		Koala.Problems.select_distinguished_representative!(temp, output, info)
		ree = sum(output.R[x,y,a,b] ? hash((x,y,a,b)) : 0 for x=1:Ns[1] for y=1:Ns[2] for a=1:Ns[3] for b=1:Ns[4])
		@test hash(output.R) == val_hash
	end
end