using RandomMatrices
using JuMP
using LinearAlgebra
using Random

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

function gen_rho(dim)
	U = rand(Haar(2), dim)
	coeffs = abs.(randn(dim))
	coeffs /= sum(coeffs)
	return U * diagm(coeffs) * adjoint(U)
end





