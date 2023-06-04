module Strategies

include("../utils2.jl")
include("../Problems/problems.jl")
using .Problems

include("blackbox.jl")
include("classical_value.jl")
include("yao.jl")
include("entangled_value.jl")
include("nonsignalling.jl")

end