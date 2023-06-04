using Random
using RandomMatrices
using LinearAlgebra

import Base.*

### Graph stuff
"""
   parse_graph6(str)
   
   Given a graph represented in graph6 format, returns the graph's adjacency matrix

	
"""

function parse_graph6(str::String)::Matrix{Bool}
	n = Int32(str[1]) - 63
	
	G = zeros(Bool, n, n)
	k::Int32 = 2
	l::Int32 = 5
	m::Int32 = Int32(str[2])-63
	for i::Int32=1:n
		for j::Int32=1:i-1
			if((m & 2^l) != 0)
				G[i,j] = G[j,i] = 1 
			end
			if(l == 0)
				l = 5
				k += 1
				if(k <= length(str))
					m = Int32(str[k]) - 63
				end
			else
				l -= 1
			end
		 end
	end
	return G
end	

"""
   list_maximal_cliques(G)
   
   Given a graph G, returns the list of all maximal cliques of G, i.e. cliques of G that aren't subsets of strictly larger cliques of G

	
"""

function list_maximal_cliques(G::Matrix{Bool})::Array{Array{Int}}
	n = size(G)[1]
	if(sum(G) == n^2 - n)
		return [[i for i=1:n]]
	end
	maximal_cliques = [] 
	
	i = 2
	index = [k <= 2 ? 1 : 0 for k=1:n]
	found_extension = [false for i=1:n]
	while(i > 0)
		if(index[i] >= n)
			if(!found_extension[i])
				clique = index[1:i-1]
				is_maximal = true
				for j=1:n
					is_extension = true
					for x in clique
						is_extension &= G[j,x]
					end
					
					if(is_extension)
						is_maximal = false
						break
					end
				end
				
				if(is_maximal)
					push!(maximal_cliques, clique)
				end
				
			end
			
			index[i] = 0
			found_extension[i] = false
			i-= 1
			continue
		end
			
		index[i] += 1
		ok = true
		for j=1:i-1
			ok &= G[index[j], index[i]]
		end
		
		if(ok)	
			index[i+1] = index[i]
			for j=1:i
				found_extension[j] = true
			end
			i += 1
		end
		
	end
	return maximal_cliques
	
end

### Pseudotelepathy games stuff

abstract type Problem end

struct Game <: Problem
	n_X::Int64
	n_Y::Int64
	n_A::Int64
	n_B::Int64
	R::Set{NTuple{4, Int64}}
	
	function Game(n_X::Int, n_Y::Int, n_A::Int, n_B::Int, V)
		R::Set{NTuple{4, Int64}} = Set()
		for x=1:n_X
			for y=1:n_Y
				for a=1:n_A
					for b=1:n_B
						if(V(x,y,a,b))
							push!(R, (x,y,a,b))
						end
					end
				end
			end
		end
		new(n_X, n_Y, n_A, n_B, R)
	end
end

struct OneWayCommunicationProblem <: Problem
	n_X::Int64
	n_Y::Int64
	c::Int64
	f # (n_X, n_Y) -> {0,1}: The value of the function
	promise # (n_X, n_Y) -> {0,1}: (n_X, n_Y) is a legal input or not
	function OneWayCommunicationProblem(n_X, n_Y, c, f)
		new(n_X, n_Y, c, f, (x,y)->true)
	end

	function OneWayCommunicationProblem(n_X, n_Y, c, f, promise)
		new(n_X, n_Y, c, f, promise)
	end
end

function encode_char(tab::Array{Bool})::Char
	return(Char(128+sum(2^(7-i) * tab[i] for i=1:length(tab))))
end

function decode_char(_c::Char)::Array{Bool}
	c = Int(_c)
	return [((c & 2^(7-i)) != 0) for i=1:7]
end
	
function encode_binary_array(tab::Array{Bool})::String
	return string([encode_char(tab[i:i+6]) for i=1:7:length(tab)]...)
end

function decode_binary_array(str::String, n::Int)::Array{Bool}
	return vcat([decode_char(c) for c in str]...)[1:n]
end

function encode_game(G::Game)::String
	tab = [false for i=1:G.n_X * G.n_Y * G.n_A * G.n_B]
	i = 1
	for x=1:G.n_X
		for y=1:G.n_Y
			for a=1:G.n_A
				for b=1:G.n_B
					tab[i] = (x,y,a,b) in G.R
					i += 1
				end
			end
		end
	end
	return encode_binary_array(tab)
end

function decode_game(str::String, n_X, n_Y, n_A, n_B)::Game
	tab = decode_binary_array(str, n_X*n_Y*n_A*n_B)
	R::Set{NTuple{4, Int64}} = Set()
	k = 1
	for x=1:n_X
		for y=1:n_Y
			for a=1:n_A
				for b=1:n_B
					if(tab[k])
						push!(R, (x,y,a,b))
					end
					k += 1
				end
			end
		end
	end
	return Game(n_X, n_Y, n_A, n_B, R)
end


function uniform_distribution(n_X::Int, n_Y::Int; promise = (x,y) -> true)
	distribution = zeros(n_X, n_Y)
	for x=1:n_X
		for y=1:n_Y
			distribution[x,y] = promise(x,y)
		end
	end
	distribution ./= sum(distribution)
	return distribution
end
 

### Useful linear algebra manipulations

function gen_rand_POVM(n, dim)
	Id = diagm([1 for i=1:dim]);
	POVM = []
	for a=1:n-1
		diag = rand(dim);
		U = rand(Haar(2), dim)
		push!(POVM, U * diagm(diag) * adjoint(U))
	end
	tot = sum(POVM)
	val_max = maximum(real.(eigvals(tot)))
	
	for a=1:n-1
		POVM[a] *= (1-1/dim) / val_max
	end	
	push!(POVM, Id - tot * (1-1/dim) / val_max)
	shuffle!(POVM)
	return POVM
end

function realify(M)
	return [real(M) imag(M); -imag(M) real(M)]
end

function unrealify(M)
	dim = div(size(M,1),2)
	return (M[1:dim, 1:dim] + im * M[1:dim, dim+1:end])
end

function gen_rho(dim)
	U = rand(Haar(2), dim)
	coeffs = abs.(randn(dim))
	coeffs /= sum(coeffs)
	return U * diagm(coeffs) * adjoint(U)
end



### NPA hierarchy stuff that is common to both the general case and the graph case


struct Monomial
	i::Int # Input
	o::Int # Output
end

# This is thrown by * in case a term of the form E^v_c E^v_c' occurs with c != c'
# Note that we use many exceptions for control flow in our NPA code. Normally, this would be 
# criminal performance-wise, but the model's building time is irrelevant here as it is only 
# built once for every class of instances.
struct ZeroException end 

function (*)(p1::Array{Monomial}, p2::Array{Monomial})::Array{Monomial}
	if(!isempty(p1) && !isempty(p2) && p1[end].i == p2[1].i)
		if(p1[end].o != p2[1].o)
			throw(ZeroException)
		end
		return [p1[1:end-1]; p2]

	else
		return [p1; p2]
	end

end

@enum Filtering turboreduit same_output_reduced same_output probabilistic full 

"""
   build_atomic_polynomials(n_i, n_o, level, filtering)
   
   Returns the list of atomic polynomials with the required properties. The polynomials are listed in increasing size, with the identity coming first.

	
"""

function build_atomic_polynomials(n_i::Int64, n_o::Int64, level::Int64, filtering::Filtering = same_output)::Array{Array{Monomial}}
	atomic_polynomials = [Monomial[]]
	
	current_atomic_polynomials = [] # Atomic polys of size l-1
	# Monomials of size 1 
	for i=1:n_i
		for o=1:n_o
			push!(atomic_polynomials, [Monomial(i,o)])
			push!(current_atomic_polynomials, [Monomial(i,o)])
		end
	end

	# We build every level from the previous one
	for l=2:level
		new_atomic_polynomials = []
		for p in current_atomic_polynomials

			for i=1:n_i
				if(i != p[end].i && (!(filtering in [same_output_reduced, turboreduit]) || i < p[end].i) && (filtering != turboreduit || i < 10))
					push!(new_atomic_polynomials, [p; Monomial(i, p[end].o)])
					push!(atomic_polynomials, [p; Monomial(i, p[end].o)])
					if(filtering != same_output && filtering != same_output_reduced && filtering != turboreduit)
						for o=1:n_o
							if(o != p[end].o && (filtering == full || rand() < 0.5))
								new_poly = [p; Monomial(i, o)]
								push!(new_atomic_polynomials, new_poly)
								push!(atomic_polynomials, new_poly)
							end
						end
					end
								
				end
			end
		end
		current_atomic_polynomials = new_atomic_polynomials
	end
	println(length(atomic_polynomials))
	return atomic_polynomials
end

# Given a final polynomial p, throws ZeroException if it is equal to zero, and otherwise returns a distinguished simplified representative
# Assumes that the polynomial was built from polynomial multiplication, so that for every consecutive pair (a_i, a_o), (b_i, b_o), we have that a_i != b_i
function eta(p_::Array{Monomial})::Array{Monomial}
	if(length(p_) <= 1) return deepcopy(p_) end # Nothing to be done there
	
	p = [p_[end]] * p_[1:end-1] # Takes care of the case where the first and last monomials in p have the same vertex
	
	
	best_hash = hash(p)
	best_p = p
	
	# We compute the hash of every possible representative of p and pick the one with the smallest hash value
	# This is purely arbitrary, this is just so that this function will return the same thing for every possible representative of p as input
	# This is up to length(p)-1 because the representative corresponding to i = length(p) is just p itself
	for i=1:length(p)-1
		p_ = [p[i+1:end]; p[1:i]]
		hash_ = hash(p_)
		if(hash_ < best_hash)
			best_p = p_
			best_hash = hash_
		elseif(hash_ == best_hash) 
			@assert best_p == p_ # Finding different things that hash to the same thing by pure chance is obviously rather unlikely, this is just for correctness's sake
		end
	end
	
	return best_p
end
