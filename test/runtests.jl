using Test, LinearAlgebra, MixedModels, CategoricalArrays

##


@testset "linalg.jl" begin
    X = rand(10^2, 202)
    X2 = copy(X)
    Y = rand(10^2, 202)
    Y2 = copy(Y)

    GOFLMM_temp.sub!(Y, X)
    @test Y == Y2 - X2

    Y3 = copy(Y)
    GOFLMM_temp.add!(Y, X)
    @test Y == Y3 + X
end


@testset "indexes_from_refs" begin
    refs = [1, 1, 2, 1, 3, 3, 2]
    indxs = [1, 2, 3, 5, 8, 6, 7]
end


@testset "simulate.jl" begin
    lm1 = GOFLMM_temp.classic_lmm(10,    1, [1.0, 0.0], 0.0, 0.0)
    @test lm1.y == lm1.X[:, 1]

    lm2 = GOFLMM_temp.classic_lmm(10^3, 10, [1.0, 1.0], 0.25, 0.25)
    fm2 = GOFLMM_temp.fit(lm2)

    @test maximum(abs.(fm2.beta .- 1)) < 0.2
    @test abs(fm2.lambda[1][1] - 1)  < 1
    @test abs(fm2.sigma - 0.25) < 0.05

    β = [1.0, 1.0, 2.0]
    lm3 = GOFLMM_temp.classic_lmm(10^3, 10, β, 0.25, [0.5 0; 0 0.5])
    fm3 = GOFLMM_temp.fit(lm3)
    A = fm3.lambda[1] * fm3.sigma 
    D = A * A'
    @test maximum(D - 0.5 * I) < 1.0

    D1 = [1.0 0.5; 0.5 0.8]
    D2 = [0.8 0; 0 0.8]
    D3 = 3.0
    Ds = [D1, D2, D3]
    lm4 = GOFLMM_temp.three_ranefs_lmm(10^3, 10, β, 0.25, Ds)
    fm4 = GOFLMM_temp.fit(lm4)
    fA2 = fm4.lambda[2] * fm4.sigma
    fD2 = fA2 * fA2'
    @test maximum(abs.(diag(D2) - diag(fD2))) < 1.0
end








@testset "construction of A and D" begin
    Z1 = ones(1, 4)
    b1 = ones(1, 2)
    λ1 = LowerTriangular(ones(1, 1))
    λ2 = LowerTriangular(ones(2,2))
    refs1 = CategoricalArrays.categorical([1, 1, 2, 2])
    refs2 = CategoricalArrays.categorical([1, 2, 1, 2])

    ran_ef1 = GOFLMM_temp.RandomTerm(2, refs1, Z1, λ1, b1)
    ran_ef2 = GOFLMM_temp.RandomTerm(2, refs2, Z1, λ1, b1)
    
    A1 = GOFLMM_temp.calculateA([ran_ef1], [0, 2])

    Zs = [1 0 1 0; 1 0 0 1; 0 1 1 0; 0 1 0 1]
    A2 = GOFLMM_temp.calculateA([ran_ef1, ran_ef2], [0, 2, 4])
    @test A1 == 2 * I
    @test A2 == Zs' * Zs


    Z2 = ones(2, 6)
    Z2[1, 2] = 2.0
    Z2[1, 5] = 2.0
    Z3 = ones(1, 6)
    b2 = ones(2, 2)
    refs3 = CategoricalArrays.categorical([1, 1, 1, 2, 2, 2])
    refs4 = CategoricalArrays.categorical([1, 2, 1, 2, 1, 2])

    ran_ef3 = GOFLMM_temp.RandomTerm(2, refs3, Z2, λ2, b2)
    ran_ef4 = GOFLMM_temp.RandomTerm(2, refs4, Z3, λ1, b1)

    A3 = GOFLMM_temp.calculateA([ran_ef3, ran_ef4], [0, 4, 6])

    Zs2 = [1 1 0 0 1 0;
        2 1 0 0 0 1;
        1 1 0 0 1 0;
        0 0 1 1 0 1;
        0 0 2 1 1 0;
        0 0 1 1 0 1]

@test A3 == Zs2' * Zs2

D3 = GOFLMM_temp.calculate_D([ran_ef3, ran_ef4], [0, 4, 6])
D3_true = [
    1 1 0 0 0 0;
    1 2 0 0 0 0;
    0 0 1 1 0 0;
    0 0 1 2 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1
]
    @test D3 == D3_true
end





@testset "multiplication with Z, P, Vi and Vi05" begin
    b1 = ones(2, 2)
D1 = [2 1; 1 2]
λ1 = cholesky(D1).L
Z1 = ones(2, 7)
Z1[1, 2] = 2.0
Z1[1, 5] = 2.0

Z2 = ones(1, 7)
b2 = ones(1, 2)
λ2 = LowerTriangular(ones(1, 1))

refs1 = CategoricalArrays.categorical([1, 1, 1, 2, 2, 2, 2])
refs2 = CategoricalArrays.categorical([1, 2, 1, 2, 1, 2, 2])

ran_ef1 = GOFLMM_temp.RandomTerm(2, refs1, Z1, λ1, b1)
ran_ef2 = GOFLMM_temp.RandomTerm(2, refs2, Z2, λ2, b2)
ran_efs = [ran_ef1, ran_ef2]

ZT = [
1 1 0 0 1 0;
2 1 0 0 0 1;
1 1 0 0 1 0;
0 0 1 1 0 1;
0 0 2 1 1 0;
0 0 1 1 0 1;
0 0 1 1 0 1
]



#Placeholder matrix and vector
M = rand(3, 3)
Mchol = cholesky(M * M')
v = rand(3)

starts = Int32.([0, 4, 6])
A, A05, Ai, Ai05 = GOFLMM_temp.calculate_As(ran_efs, starts)
DT = GOFLMM_temp.calculate_D(ran_efs, starts)
C, Ci = GOFLMM_temp.calculate_Cs(A05 * DT * A05')


lmm = GOFLMM_temp.LMM(v, M, v, 0.0, ran_efs,
    Int32(7), Int32(2), starts, 
    A, Mchol, Ai, Ai05, DT, 
    inv(I + DT * A), C, Ci)

QT = ZT * Ai05
V = I + ZT * DT * ZT'
Vi = Matrix(inv(Symmetric(V)))
Vi05 = inv(sqrt(Symmetric(V)))
P = I - QT * QT'


## Z

X = rand(6, 10)
Y = zeros(7, 10)
GOFLMM_temp.multZ!(Y, lmm, X) 
@test maximum(abs, Y - ZT * X) == 0

X2 = rand(7, 10)
Y2 = zeros(6, 10)
GOFLMM_temp.multZt!(Y2, lmm, X2)
@test maximum(abs, ZT' * X2 - Y2) == 0

## Vi

X = rand(7, 8)
X2 = similar(X)
temp = rand(6, 8)
temp2 = rand(6, 8)
Y = Vi * X
GOFLMM_temp.mult_Vi!(X, X2, temp, temp2, lmm)
@test maximum(X - Y) < 10^(-14)

## Vi05





# Is Ci correct?
Q = ZT * Ai05
maximum((I + Q * Ci * Q') - Vi05)


Y = Vi05 * X
GOFLMM_temp.mult_Vi05!(X, X2, temp, temp2, lmm)
@test maximum(X - Y) < 10^(-14)

## P
Y = P * X
GOFLMM_temp.mult_P!(X, X2, temp, temp2, lmm)
@test maximum(abs, X - Y) < 10^(-14)



end 






















@testset "SigfMat tests" begin
    @test GOFLMM_temp.calculate_n([1.0, 2.0, 3.0, 5.0, 5.1], 0.5, 5) == 1
    @test GOFLMM_temp.calculate_n([1.0, 2.0, 3.0, 5.0, 5.1], 0.49, 5) == 2
    sm =  GOFLMM_temp.SigfMat([1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0], 0.25)
    p1 = 918 / 1024
    p2 = 1/2
    p3 = 106 / 1024
    @test sm.M == [
        p1 p1 p2 p1 p1 p1 p1; 
        p2 p3 p3 p2 p2 p2 0;
          0 0  0 p3 p3  0 0]

end






##


@testset "gof.jl" begin
    
    β = [1.0, 1.0, 2.0]
    D1 = [1.0 0.5; 0.5 0.8]
    D2 = [0.8 0; 0 0.8]
    D3 = [3.0 0.0 1.0; 0.0 4.0 1.0; 1.0 1.0 8.0]
    Ds = [D1, D2, D3]
    lm4 = GOFLMM_temp.three_ranefs_lmm(10^3, 10, β, 0.25, Ds)
    fm4 = GOFLMM_temp.fit(lm4)
    
    lmm4 = GOFLMM_temp.LMM(fm4)
    
    lmm_gof = GOFLMM_temp.gof(lmm4, 200, 0.5)
    


    #ps = GOFLMM_temp.repeat_simulations(10^3, 10^3, 1.0, GOFLMM_temp.three_ranefs_lmm, (10^3, 10, β, 0.25, Ds))


end

















@testset "the fitting" begin
    

    b1 = ones(2, 2)
D1 = [2 1; 1 2]
λ1 = cholesky(D1).L
Z1 = ones(2, 7)
Z1[1, 2] = 2.0
Z1[1, 5] = 2.0

Z2 = ones(1, 7)
b2 = ones(1, 2)
λ2 = LowerTriangular(ones(1, 1))

refs1 = CategoricalArrays.categorical([1, 1, 1, 2, 2, 2, 2])
refs2 = CategoricalArrays.categorical([1, 2, 1, 2, 1, 2, 2])

ran_ef1 = GOFLMM_temp.RandomTerm(2, refs1, Z1, λ1, b1)
ran_ef2 = GOFLMM_temp.RandomTerm(2, refs2, Z2, λ2, b2)
ran_efs = [ran_ef1, ran_ef2]

ZT = [
1 1 0 0 1 0;
2 1 0 0 0 1;
1 1 0 0 1 0;
0 0 1 1 0 1;
0 0 2 1 1 0;
0 0 1 1 0 1;
0 0 1 1 0 1
]


X = rand(7, 3)
β = ones(3)
y = X * β

for ran_ef in ran_efs
    for i = 1:ran_ef.groups.n
        indexes_li = ran_ef.groups.indexes[i]
        λ = ran_ef.λ
        y[indexes_li] += ran_ef.Z[:, indexes_li]' * λ * randn(size(λ, 2))
    end
end

y += randn(7)


starts = Int32.([0, 4, 6])
A, A05, Ai, Ai05 = GOFLMM_temp.calculate_As(ran_efs, starts)
D = GOFLMM_temp.calculate_D(ran_efs, starts)
Bi = inv(I + D * A)
C, Ci = GOFLMM_temp.calculate_Cs(A05 * D * A05')
H = GOFLMM_temp.calculate_H(copy(X), ran_efs, starts, D, Bi)


lmm = GOFLMM_temp.LMM(y, X, β, 1.0, ran_efs,
    Int32(7), Int32(2), starts, 
    A, H, Ai, Ai05, D, 
    Bi, C, Ci)

V = I + ZT * D * ZT'
Vi = Matrix(inv(Symmetric(V)))


## H 

H0 = cholesky(Symmetric(lmm.X' * Vi * lmm.X))
@test maximum(H0.L * H0.L' - H.L * H.L') < 10^(-14)


## yh

yh0 = X * inv(H0) * (X' * Vi * lmm.y)
Y0 = [lmm.y lmm.y]
t1 = zeros(6, 2)
t2 = zeros(6, 2)
t3 = zeros(3, 2)
yh = GOFLMM_temp.get_fitted!(copy(Y0), similar(Y0), t1, t2, t3, lmm)
@test maximum(abs, yh[:, 1] - yh0) < 10^(-12)


end
