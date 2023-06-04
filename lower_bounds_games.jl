include("utils.jl")

mutable struct Deterministic_strategy
    outputs_Alice::Array{Int64}
    outputs_Bob::Array{Int64}
end

function evaluate_deterministic_strategy(game::Game, strategy::Deterministic_strategy, distribution::Array{Float64,2})
    tot::Float64 = 0
    for x=1:game.n_X
        for y=1:game.n_Y
			if((x,y,strategy.outputs_Alice[x], strategy.outputs_Bob[y]) in game.R)
				tot += distribution[x,y]
			end
        end
    end
    
    return tot
end

#function improve_deterministic_strategy(game::Game, strategy::Deterministic_strategy, distribution::Array{Float64,2}; window_size = 3)
#	T = zeros(6, game.n_A)
#	U = zeros(6, game.n_B)
	

#function find_good_strategy(game::Game, strategy::Deterministic_strategy, distribution::Array{Float64,2}; iterations=150, window_size = 3)


#function find_good_strategy(game::Game, distribution::Array{Float64,2}; iterations=150, window_size = 3)



# Uses column generation to generate a hard distribution 
# N is the domain size
# legal_position is a function which takes in an element i in {1,...,N} and returns a boolean corresponding to whether D[i] should be forced to zero
# oracle is a function which takes in a distribution D on {1,...,N} and returns an array P of N probabilities so that \sum_i D_i P_i is maximal within the constraints corresponding to the problem
function generate_hard_distribution(N::Int64, legal_position, oracle; iterations=1000)
    model = Model(CPLEX.Optimizer)
	set_silent(model)
	
	# Declare 
	@variable(model, D[1:N]; lower_bound = 0.0)
	@constraint(model, sum(D) == 1) 
	
	valid_index = 0
	for i=1:N
		if(legal_position[i])
			valid_index = i
		else
			@constraint(model, D[i] == 0)
		end
	end
	
    current_distribution = [i == valid_index ? 1.0 : 0.0 for i=1:N]
	
	@variable(model, z >= 0) # Best winning probability
	
	# Additional variables for facilitating convergence
	# We only allow the model to consider distributions that are within 0.1 of the hardest distribution found so far in every position, with larger deviations being penalized
	@variable(model, gap_plus[1:N]; lower_bound = 0.0)
	@variable(model, gap_minus[1:N]; lower_bound = 0.0)
	@variable(model, allowed_gap[1:N])
	@constraint(model, allowed_gap .<= 0.1)
	@constraint(model, allowed_gap .>= -0.1)
	
	@objective(model, Min, z + 1000 * sum(gap_plus) + 1000 * sum(gap_minus))


	time_model = 0
	time_search = 0
	val_strat = 0
	general_constraints = Dict() # Constraints to how long since the constraint had a nonzero reduced cost: constraints whose reduced cost has been zero for too long are deleted
	stabilizing_constraints = [] # Constraints to 
	
	best_distribution = deepcopy(current_distribution)
	least_success_probability = 1.0

	for _=1:iterations
		success_probabilities = oracle(current_distribution)
		success_probability = sum(success_probabilities[i] * current_distribution[i])
		if(success_probability < least_success_probability)
			least_success_probability = success_probability
			best_distribution[:] .= current_distribution
		end
		
		c = @constraint(model, z >= sum(success_probabilities[i] * D[i] for i=1:N))
		general_constraints[c] = 0
		
		if(i >= 50)
			for c in stabilizing_constraints
				delete(model, c)
			end
			stabilizing_constraints = []
			if(i%10 != 0)
				for c in  @constraint(model, D .- best_distribution .== gap_plus - gap_minus + allowed_gap)
					push!(stabilizing_constraints, c)
				end
			end
		end

		time_model += @elapsed optimize!(model)
		current_distribution[:] .= JuMP.value.(pi)

		current_z = objective_value(model)


		
		if(current_z > 0.5 + 1e-04)
			for constraint in keys(general_constraints)
				if(getdual(constraint) < 1e-04)
					general_constraints[contrainte] += 1
					if(general_constraints[constraint] > 10)
						delete!(general_constraints, constraint)
						delete(model, constraint)
					end
				else
					general_constraints[constraint] = 0
				end
			end
		end
	end
	return distribution
end
