using RandomMatrices
using JuMP
using LinearAlgebra
using Random

export generate_random_quantum_correlation
######################### String manipulations #########################

function encode_char(tab::Vector{Bool})::Char
    length(tab) <= 6
    return(Char(63+sum(2^(6-i) * tab[i] for i=1:length(tab))))
end

function decode_char(_c::Char)::Vector{Bool}
    c = Int(_c) - 63
    return [((c & 2^(6-i)) != 0) for i=1:6]
end

function encode_binary_array(tab::Vector{Bool})::String
    return string([encode_char(tab[i:min(i+5, length(tab))]) for i=1:6:length(tab)]...)                                                                                                                                                                            
end

function decode_binary_array(str::String, n::Int)::Vector{Bool}
    return vcat([decode_char(c) for c in str]...)[1:n]
end


######################### Complex hermitian matrix modelling #########################

function enforce_SDP_constraints(model::Model, variable::Symmetric{VariableRef, Matrix{VariableRef}})
	dim = div(size(variable,1), 2)
	for i=1:dim
		for j=i:dim
			@constraint(model, variable[i,j] == variable[i+dim, j+dim])
			@constraint(model, variable[i+dim,j] == -variable[i, j+dim])
		end
	end
end

function realify(M)
	return [real(M) imag(M); -imag(M) real(M)]
end

function unrealify(M)
	dim = div(size(M,1),2)
	return (M[1:dim, 1:dim] + im * M[1:dim, dim+1:end])
end


######################### Generation of random POVMs and states #########################

""" 
	gen_rand_POVM(n::Int, dim::Int)::Vector{Matrix{ComplexF64}}

Returns a randomly generated POVM with n elements over a space of dimension dim
"""
function gen_rand_POVM(n::Int, dim::Int)::Vector{Matrix{ComplexF64}}
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

"""
	gen_projective_measurement(n::Int, dim::Int)::Vector{Matrix{ComplexF64}}
	
"""
function gen_projective_measurement(n::Int, dim::Int)::Vector{Matrix{Float64}}
	@assert dim >= n
	basis = rand(Haar(1), dim)
	current = zeros(Float64, dim, dim)
	measurement = []
	projector_dimension = div(dim, n)
	k = 0
	for j=1:dim
		current += basis[:,j] * adjoint(basis[:,j])
		k += 1
		if(k >= projector_dimension)
			push!(measurement, current)
			current = zeros(Float64, dim, dim)
			k = 0
		end
	end
	if(k != 0)
		push!(measurement, current)
	end
	shuffle!(measurement)
	return measurement
end
	
""" 
	gen_rho(dim::Int64)::Matrix{ComplexF64}

Returns a random mixed state on a space of dimension dim
"""
function gen_rho(dim::Int64)::Matrix{ComplexF64}
	U = rand(Haar(1), dim)
	coeffs = abs.(randn(dim))
	coeffs /= sum(coeffs)
	return U * diagm(coeffs) * adjoint(U)
end
	

""" 
	generate_random_quantum_correlation(n_X::Int, n_Y::Int, n_A::Int, n_B::Int, dim::Int)::Array{Float64, 4}

Given the input and output sizes and the dimension dim, returns a correlation that can be realized by participants who share a maximally entangled state of dimension dim.
"""
function generate_random_quantum_correlation(n_X::Int, n_Y::Int, n_A::Int, n_B::Int, dim::Int)::Array{Float64, 4}
	res = zeros(Float64, n_X, n_Y, n_A, n_B)
	measurements_alice = [gen_projective_measurement(n_A, dim) for _=1:n_X]
	measurements_bob = [gen_projective_measurement(n_B, dim) for _=1:n_Y]
	for x=1:n_X
		for y=1:n_Y
			for a=1:n_A
				for b=1:n_B
					res[x,y,a,b] = abs(tr(measurements_alice[x][a] * adjoint(measurements_bob[y][b])))
				end
			end
		end
	end
	return res
end





