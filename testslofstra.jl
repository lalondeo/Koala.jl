using COSMO, JuMP, LinearAlgebra, Random


function gen_pos(dim)
	A = rand(dim,dim)
	A += transpose(A)
	t = eigmin(A)
	return A - diagm([t for i=1:dim])
	
end

function generate_POVM(dim, k)
	vals_1 = [gen_pos(dim) for i=1:k-1]
	vals_1 ./= eigmax(sum(vals_1)) * k / (k-1)
	push!(vals_1, diagm([1 for i=1:dim]) - sum(vals_1))
	return shuffle(vals_1)
end

function SDP(n_X, n_Y, n_A, n_B, f, F, dim, legal_input = (x,y) -> true)
	E = Dict()
	m = JuMP.Model(COSMO.Optimizer)
	Id = diagm([1 for i=1:dim])
	for x=1:n_X
		for a=1:n_A
			E[(x,a)] = @variable(m, [1:dim,1:dim], PSD)
		end
		@constraint(m, sum(E[(x,a)] for a=1:n_A) .== Id)
	end
	
	@variable(m, t)
	@objective(m, Max, t)
	for x=1:n_X
		for y=1:n_Y
			if(legal_input(x,y))
				@constraint(m, t <= 1/dim*sum(f(x,y,a,b) * tr(F[(y,b)] * E[(x,a)]) for a=1:n_A for b=1:n_B))
			end
		end
	end
	set_silent(m)
	JuMP.optimize!(m)
	_E = Dict()
	for x=1:n_X
		for a=1:n_A
			_E[(x,a)] = JuMP.value.(E[(x,a)])
		end
	end
	return _E, JuMP.value.(t)
end
	

function chercher_strat(n_X, n_Y, n_A, n_B, f, dim, N = 1000, legal_input = (x,y) -> true)
	legal_input_2 = (y,x) -> legal_input(x,y)
	f_2 = (y,x,b,a) -> f(x,y,a,b)
	E = Dict()
	F = Dict()
	for x=1:n_X
		POVM = generate_POVM(dim, n_A)
		for a=1:n_A
			E[(x,a)] = POVM[a]
		end
	end
	for y=1:n_Y
		POVM = generate_POVM(dim, n_B)
		for b=1:n_B
			F[(y,b)] = POVM[b]
		end
	end
	t = 0
	for i=1:N
		_E,t_ = SDP(n_X, n_Y, n_A, n_B, f, F, dim, legal_input)
		t = max(t, t_)
		diff = 0
		for x=1:n_X
			for a=1:n_A
				diff += norm(_E[(x,a)] - E[(x,a)])
			end
		end
		if(diff / n_X / n_A <= 0.01)
			#println("ICI 1")
			for x=1:n_X
				bruit = generate_POVM(dim, n_A)
				for a=1:n_A
					E[(x,a)] = 0.001 * E[(x,a)] + 0.999 * bruit[a]
				end
			end
		else
			for x=1:n_X
				for a=1:n_A
					E[(x,a)] = _E[(x,a)]
				end
			end
		end
		
		_F,t_ = SDP(n_Y, n_X, n_B, n_A, f_2, E, dim, legal_input_2)
		t = max(t, t_)
		diff = 0
		for y=1:n_Y
			for b=1:n_B
				diff += norm(_F[(y,b)] - F[(y,b)])
			end
		end
		
		if(diff / n_Y / n_B <= 0.01)
			#println("ICI 2")
			for y=1:n_Y
				bruit = generate_POVM(dim, n_B)
				for b=1:n_B
					F[(y,b)] = 0.001 * F[(y,b)] + 0.999 * bruit[b]
				end
			end
		else	
			for y=1:n_Y
				for b=1:n_B
					F[(y,b)] = _F[(y,b)]
				end
			end	
		end
		println(t, " ", t_)	
	end
	return t
end

# Probleme de comm
f = Dict()
N = 4
aretes = []
for i=1:N
	for j=1:N
		if(i != j)
			push!(aretes, [i,j])
		end
	end
end

for x=0:2*length(aretes)-1
	c = x % 2
	_x = div(x-c, 2)+1
	for y=0:2*N-1
		d = y % 2
		_y = div(y-d, 2) + 1
		if(!(_y in aretes[_x]))
			f[(x+1,y+1)] = 2
		else
			f[(x+1,y+1)] = (c + d + (_y == aretes[_x][1])) % 2
		end
	end
end
		
			
	
			
	
	
