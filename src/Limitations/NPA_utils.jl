using Random
import Base.*
import Base.==
import Base.isequal
import Base.length
import Base.copy

# Type corresponding to a NPA object
# Is expected to have subfields model for the corresponding SDP model as well as correlation_components::Dict{Tuple{Int,Int,Int,Int}, Tuple{Int,Int}} for the index in the
# positive semidefinite matrix corresponding to the probability that the output is (a,b) given that the input is (x,y)
abstract type NPA end

struct Projector
	i::Int # Input
	o::Int # Output
end


# This is thrown by * in case a term of the form E^v_c E^v_c' occurs with c != c'
# Note that we use many exceptions for control flow in our NPA code. Normally, this would be 
# criminal performance-wise, but the model's building time is irrelevant here as it is only 
# built once for every class of instances.
struct ZeroException end 

function (*)(m1::Array{Projector}, m2::Array{Projector})::Array{Projector}
	if(!isempty(m1) && !isempty(m2) && m1[end].i == m2[1].i)
		if(m1[end].o != m2[1].o)
			throw(ZeroException)
		end
		return [m1[1:end-1]; m2]

	else
		return [m1; m2]
	end

end

@enum Filtering same_output_reduced same_output probabilistic full 

"""
   build_atomic_monomials(n_i, n_o, level, filtering)
   
   Returns the list of atomic monomials with the required properties. The monomials are listed in increasing size, with the identity coming first.

	
"""
function build_atomic_monomials(n_i::Int64, n_o::Int64, level::Int64, filtering::Filtering = same_output)::Array{Array{Projector}}
	atomic_monomials = [Projector[]]
	
	current_atomic_monomials = [] 
	# Projectors of size 1 
	for i=1:n_i
		for o=1:n_o
			push!(atomic_monomials, [Projector(i,o)])
			push!(current_atomic_monomials, [Projector(i,o)])
		end
	end

	Random.seed!(1)
	# We build every level from the previous one
	for l=2:level
		new_atomic_monomials = []
		for m in current_atomic_monomials
			for i=1:n_i
				if(i != m[end].i && (!(filtering != same_output_reduced) || i < m[end].i))
					push!(new_atomic_monomials, [m; Projector(i, m[end].o)])
					push!(atomic_monomials, [m; Projector(i, m[end].o)])
					if(filtering != same_output && filtering != same_output_reduced)
						for o=1:n_o
							if(o != m[end].o && (filtering == full || rand() < 0.5))
								new_poly = [m; Projector(i, o)]
								push!(new_atomic_monomials, new_poly)
								push!(atomic_monomials, new_poly)
							end
						end
					end
								
				end
			end
		end
		current_atomic_monomials = new_atomic_monomials
	end
	return atomic_monomials
end

# Equivalence relation for the case where the state is maximally entangled, in which case the final monomials obey a cyclicity property
# Given a monomial, returns a distinguished representative that is equivalent to it
function eta(m_::Array{Projector})::Array{Projector}
	if(length(m_) <= 1) return deepcopy(m_) end # Nothing to be done there
	
	m = [m_[end]] * m_[1:end-1] # Takes care of the case where the first and last projectors in m have the same vertex
	
	best_hash = hash(m)
	best_m = m
	
	# We compute the hash of every possible representative of m and pick the one with the smallest hash value
	# This is purely arbitrary, this is just so that this function will return the same thing for every possible representative of m as input
	# This is up to length(m)-1 because the representative corresponding to i = length(m) is just m itself
	for i=1:length(m)-1
		m_ = [m[i+1:end]; m[1:i]]
		hash_ = hash(m_)
		if(hash_ < best_hash)
			best_m = m_
			best_hash = hash_
		elseif(hash_ == best_hash) 
			@assert best_m == m_ # Finding different things that hash to the same thing by pure chance is obviously rather unlikely, this is just for correctness's sake
		end
	end
	return best_m
	
end