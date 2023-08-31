using ArgParse
using Base.Threads
using Combinatorics
using LinearAlgebra
using Serialization
using ProgressMeter

BLAS.set_num_threads(1)

include("../src/koala.jl")

function get_functions(n_X::Int, n_Y::Int)
    nbits_X, nbits_Y = trunc(Int, log2(n_X)), trunc(Int, log2(n_Y))

    @assert nbits_X == log2(n_X) && nbits_X > 0  && nbits_Y == log2(n_Y) && nbits_Y > 0 (
        "Sizes of sets X and Y should be powers of 2, n_X=$n_X, n_Y=$n_Y"
    )

    # Compute and return the set of possible functions
    return filter(function_filter(n_X, n_Y), vec(collect(Iterators.product(Iterators.repeated(0:2, n_X * n_Y)...))))
end

function function_filter(n_X::Int64, n_Y::Int64)
    return function (f::NTuple)
        matrix = reshape(collect(f), (n_X, n_Y))
        return issorted([hash(sort(row)) for row in eachrow(matrix)]) && issorted([hash(sort(column)) for column in eachcol(matrix)]) 
    end
end

# Assign a function to a thread
function assign(f::NTuple{16, Int64}, nthreads::Int64)
    # a single hash(f) call made it so that some threads had 0 functions assigned :(
    return hash(hash(f)) % nthreads + 1
end

# Represent a function vector (vector with values from 0 to 2 with 2 representing an impossible value) and its promise as matrices
function function_vector_to_matrices(n_X::Int64, n_Y::Int64, f::NTuple{16, Int64})
    return [f[(x-1) * n_X + y] == 1 for x=1:n_X,  y=1:n_Y], [f[(x-1) * n_X + y] != 2 for x=1:n_X,  y=1:n_Y]
end

# Represent a function vector (vector with values from 0 to 2 with 2 representing an impossible value) and its promise as matrices
function update_problem!(problem::Koala.Problems.OneWayCommunicationProblem, n_X::Int64, n_Y::Int64, f::NTuple{16, Int64})
    # Update function
    for x=1:n_X,  y=1:n_Y
        problem.f[x, y] = f[(x-1) * n_X + y] == 1
    end
    # Update promise
    for x=1:n_X,  y=1:n_Y
        problem.promise[x, y] = f[(x-1) * n_X + y] != 2
    end
end


struct SingleExperimentResults
    index_within_process_id::Int64
    
    experiment_start_time::Float64
    experiment_end_time::Float64

    problem::Koala.Problems.OneWayCommunicationProblem
    distribution::Union{Matrix{Float64}, Nothing}

    classical_strat_lower_bound::Float64
    yao_strat_lower_bound::Float64

    cleve_buhrman_strat_upper_bound_GENERAL::Float64
    cleve_buhrman_strat_upper_bound_EPR::Float64

    skipped::Bool

    function SingleExperimentResults(index_within_process_id::Int64, experiment_start_time::Float64, experiment_end_time::Float64, problem::Koala.Problems.OneWayCommunicationProblem, distribution::Union{Matrix{Float64}, Nothing},
        classical_strat_lower_bound::Float64, yao_strat_lower_bound::Float64, cleve_buhrman_strat_upper_bound_GENERAL:: Float64, cleve_buhrman_strat_upper_bound_EPR::Float64; skipped::Bool=false)
        new(index_within_process_id, experiment_start_time, experiment_end_time, problem, distribution, classical_strat_lower_bound, yao_strat_lower_bound, cleve_buhrman_strat_upper_bound_GENERAL, cleve_buhrman_strat_upper_bound_EPR, skipped)
    end

    function SingleExperimentResults(index_within_process_id::Int64, experiment_start_time::Float64, experiment_end_time::Float64, problem::Koala.Problems.OneWayCommunicationProblem)
        SingleExperimentResults(index_within_process_id, experiment_start_time, experiment_end_time, problem, Nothing(), 1., 1., 1., 1.; skipped=true)
    end
end

struct SplitExperimentsResults
    total_processes::Int64
    process_id::Int64
    total_assigned_problems_to_process::Int64

    start_index::Int64
    last_index::Int64

    start_time::Float64
    last_update_time::Float64

    results::Vector{SingleExperimentResults}
end


function get_command_line_args()
    settings = ArgParseSettings()

    @add_arg_table settings begin
        "--x-size", "-x"
            help = "Size of X, the set of possible inputs for Alice"
            arg_type = Int
            required = true
        "--y-size", "-y"
            help = "Size of Y, the set of possible inputs for Bob"
            arg_type = Int
            required = true
        "--c-size", "-c"
            help = "Size of C, the set of possible communications between Alice and Bob"
            arg_type = Int
            required = true
        # "--e-dim", "-e"
        #     help = "Local dimension of the entangled quantum state for entangled strategies"
        #     arg_type = Int
        #     required = true
        "--npa-lvl", "-l"
            help = "Level in the NPA hierarchy of the semi-definite program to compute the higher bound of the success probability of the entangled strategies."
            arg_type = Int
            default = 2
            required = false
        "--n-processes", "-n"
            help = "Total number of processes used, used to assign problems to processes"
            arg_type = Int
            required = true
        "--process-id", "-i"
            help = "Process id, used to assign problems to processes"
            arg_type = Int
            required = true
        "--results-dir", "-d"
            help = "Directory where the program will save the results. *Make sure the program has write permissions to it"
            arg_type = String
            default = "./"
            required = false
    end
    return parse_args(ARGS, settings)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = get_command_line_args()

    n_X, n_Y, c, npa_lvl, n_processes, process_id, dir = args["x-size"], args["y-size"], args["c-size"], args["npa-lvl"], args["n-processes"], args["process-id"], args["results-dir"]

    @time begin

        F = get_functions(n_X, n_Y)
        # processed_counts = zeros(Threads.nthreads())
        # skipped_counts = zeros(Threads.nthreads())
        assignments = [[] for i=1:n_processes]
        for f in F
            push!(assignments[assign(f, n_processes)], f)
        end
        # assigned_counts = [size(assignments[i]) for i in 1:Threads.nthreads()]
    end
    @time begin
    
        problem = Koala.Problems.OneWayCommunicationProblem(n_X, n_Y, c, (x, y) -> true)

        game = Koala.Problems.gameify(problem)

        perfect_classical_strat_info = Koala.Strategies.HasPerfectClassicalStrategyInfo(game)
        classical_solver_data = Koala.Strategies.ClassicalSolverData(game)
        yao_solver_data = Koala.Strategies.YaoSolverData(n_X, n_Y, c)
        # entangled_solver_data = Koala.Strategies.EntangledSolverData(game, e_dim)

        npa_info = Koala.Bounds.NPAGeneral(problem, npa_lvl)
        npa_epr_info = Koala.Bounds.NPAGeneral(problem, npa_lvl; impose_maximally_entangled=true)


        classical_strat = Koala.Strategies.ClassicalStrategy(game)
        yao_strategy = Koala.Strategies.YaoProtocol(problem)
        # entangled_strat = Koala.Strategies.EntangledStrategy(game, e_dim)

        converted_dist = [0.0 for x=1:n_X, y=1:(n_Y * c)]


        split_start_time = time()
        ended_time = 0

        progress_bar = Progress(length(assignments[process_id]), showspeed=true)
    
        println(length(assignments[process_id]))


        for (index, func) in enumerate(assignments[process_id])
            # try
                # if isfile("$dir$process_id-$index.results")
                    # continue
                # end

                if !isfile("$dir$process_id-$index.ERROR")
                    continue
                end
                
                println("index : $index")

                start_time = time()
                update_problem!(problem, n_X, n_Y, func)
                
                global game = Koala.Problems.gameify(problem)

                if Koala.Strategies.has_perfect_classical_strategy!(game, classical_strat, perfect_classical_strat_info)
                    # processed_counts[Threads.threadid()] += 1
                    # skipped_counts[Threads.threadid()] += 1
                    # next!(progress_bar, showvalues = [(:"Total functions processed", "$(progress_bar.counter + 1) / $(progress_bar.n)"), (:"Functions processsed per thread", processed_counts), (:"Functions assigned per thread", assigned_counts), (:"Skipped", skipped_counts)])
                    
                    global ended_time = time() 

                    open("$dir$process_id-$index.results", "w") do io
                        serialize(io, SplitExperimentsResults(length(assignments), process_id, length(assignments[process_id]), index, index, time(), time(), [SingleExperimentResults(index, start_time, ended_time, deepcopy(problem))]))    
                    end
                    
                    next!(progress_bar)
                    continue
                end
                # println("dist")
                # @time begin
                    yao_hard_dist = Koala.Strategies.generate_hard_distribution_yao(problem, yao_strategy, yao_solver_data; max_iter=100, verbose=false)
                # end
                Koala.Problems.convert_distribution!(problem, yao_hard_dist, converted_dist)

                # println("classical")
                # @time begin
                    Koala.Strategies.optimize_strategy!(game, classical_strat, converted_dist, classical_solver_data; max_iter=50)
                    p_c = Koala.Strategies.evaluate_success_probability(game, classical_strat, converted_dist)
                # end

                # println("yao")
                # @time begin
                    Koala.Strategies.optimize_strategy!(problem, yao_strategy, yao_hard_dist, yao_solver_data; max_iter=50)
                    p_y = Koala.Strategies.evaluate_success_probability(problem, yao_strategy, yao_hard_dist)
                # end

                # println("e")
                # @time begin
                    # Koala.Strategies.optimize_strategy!(game, entangled_strat, converted_dist, entangled_solver_data; max_iter=50)
                    # p_e = Koala.Strategies.evaluate_success_probability(game, entangled_strat, converted_dist)
                # end

                # println("npa")
                # @time begin
                    p_e_ub = Koala.Bounds.upper_bound_game(game, converted_dist, npa_info; target=1/2 + 1/2 * p_y + 0.01, decompose=true, rho=0.01, eps_abs = 1e-08, eps_rel = 1e-08, max_iter=10000, verbose=false) 
                # end
                    p_e_ub_epr = Koala.Bounds.upper_bound_game(game, converted_dist, npa_epr_info; target=1/2 + 1/2 * p_y + 0.01, decompose=true, rho=0.01, eps_abs = 1e-08, eps_rel = 1e-08, max_iter=10000, verbose=false) 
                global ended_time = time()

                open("$dir$process_id-$index.results", "w") do io
                    serialize(io, SplitExperimentsResults(length(assignments), process_id, length(assignments[process_id]), index, index, time(), time(), [SingleExperimentResults(index, start_time, ended_time, deepcopy(problem), yao_hard_dist, p_c, p_y, p_e_ub, p_e_ub_epr)]))
                end
                next!(progress_bar)

            # end
            
            # catch
                # open("$dir$process_id-$index.ERROR", "w") do io
                #     serialize(io, "")
                # end
                # next!(progress_bar)
            # end
        end
    end
end
