using Koala
using Test

@testset "Koala.jl" begin
	include("testgraphutils.jl")
	include("testisomorphism.jl")
	include("testperfectclassicalstrategy.jl")
end