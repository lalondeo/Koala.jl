using Random
import Base.*
import Base.==
import Base.isequal
import Base.length
import Base.copy
import Base.hash

export same_output_reduced, same_output, probabilistic, full 

# This file contains basic functions for working with monomials


# Type corresponding to a NPA object
# Is expected to have subfields model for the corresponding SDP model as well as correlation_components::Dict{Tuple{Int,Int,Int,Int}, Tuple{Int,Int}} for the index in the
# positive semidefinite matrix corresponding to the probability that the output is (a,b) given that the input is (x,y)
abstract type NPA end

struct Projector
	i::Int # Input
	o::Int # Output
end

function hash(P::Projector)
	return hash("$(P.i)|$(P.o)")
end

function (*)(m1::Nothing, m2::Vector{Projector})::Nothing
	return nothing
end

function (*)(m1::Array{Projector}, m2::Nothing)::Nothing
	return nothing
end

function(*)(m1::Nothing, m2::Nothing)::Nothing
	return nothing
end


function (*)(m1::Vector{Projector}, m2::Vector{Projector})::Union{Vector{Projector}, Nothing}
	if(!isempty(m1) && !isempty(m2) && m1[end].i == m2[1].i)
		if(m1[end].o != m2[1].o)
			return nothing
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
function build_atomic_monomials(n_i::Int64, n_o::Int64, level::Int64, filtering::Filtering = same_output)::Vector{Vector{Projector}}
	atomic_monomials = Vector{Projector}[Projector[]]
	current_atomic_monomials = Vector{Projector}[] 
	# Projectors of size 1 
	for i=1:n_i
		for o=1:n_o
			push!(current_atomic_monomials, [Projector(i,o)])
		end
	end
	append!(atomic_monomials, current_atomic_monomials)

	Random.seed!(1)
	# We build every level from the previous one
	for l=2:level
		new_atomic_monomials = Vector{Projector}[]
		for m in current_atomic_monomials
			for i=1:n_i
				if((i != m[end].i) && (i < m[end].i || ((filtering != same_output_reduced) && (filtering != probabilistic || rand() < 0.5))))
					for o=1:n_o
						if((o == m[end].o || (filtering != same_output_reduced && filtering != same_output)))
							push!(new_atomic_monomials, [m; Projector(i, o)])
						end
					end
				end
			end
		end
		append!(atomic_monomials, new_atomic_monomials)
		current_atomic_monomials = new_atomic_monomials
	end
	
	return atomic_monomials
end

# Equivalence relation for the case where the state is maximally entangled, in which case the final monomials obey a cyclicity property
# Given a monomial, returns a distinguished representative that is equivalent to it
# The name comes from the paper of Russell referenced in NPA_synchronous
function eta(m_::Vector{Projector})::Union{Vector{Projector}, Nothing}
	if(length(m_) <= 1) return deepcopy(m_) end # Nothing to be done there
	
	m = [m_[end]] * m_[1:end-1] # Takes care of the case where the first and last projectors in m have the same input
	if(m == nothing)
		return nothing
	end
	
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
        
	m = Base.reverse(m)
	for i=0:length(m)-1
		m_ = [m[i+1:end]; m[1:i]]
		hash_ = hash(m_)
		if(hash_ < best_hash)
			best_m = m_
			best_hash = hash_
		elseif(hash_ == best_hash) 
			@assert best_m == m_
		end
	end
	return best_m
	
end

function eta(m_::Nothing)::Nothing
	return nothing
end

	
	
	
	
