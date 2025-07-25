using Test
using GaussianStates, LinearAlgebra, GenericLinearAlgebra
using ExponentialUtilities: ExponentialUtilities

LinearAlgebra.exp!(A::Matrix{BigFloat}) = ExponentialUtilities.exponential!(A)

@testset "GaussianState constructors" begin
    N = 4
    r = rand(2N)
    σ = Symmetric(rand(2N, 2N))
    @test nmodes(GaussianState(r, σ)) == N
    @test GaussianState(σ) == GaussianState(zeros(2N), σ)

    @test GaussianState(r, σ) ≈ GaussianState(
        r .+ eps(eltype(r)) .* rand(length(r)), σ .+ eps(eltype(σ)) .* rand(size(σ)...)
    )
end

@testset "Random state generation" begin
    g = randgaussianstate(4)
    @test is_valid_covariance_matrix(g.covariance_matrix)
    @test number(g) ≥ 0

    g = randgaussianstate(4, rand(4))
    @test is_valid_covariance_matrix(g.covariance_matrix)
    @test number(g) ≥ 0

    g = randgaussianstate(4; pure=true)
    @test is_valid_covariance_matrix(g.covariance_matrix)
    @test purity(g) ≈ 1
    @test number(g) ≥ 0

    g = randgaussianstate(4; pure=true, displace=false)
    @test_skip is_valid_covariance_matrix(g.covariance_matrix)
    @test iszero(g.first_moments)
    @test purity(g) ≈ 1
    @test number(g) ≥ 0
end

@testset verbose = true "Purity-preserving Gaussian operations" begin
    # This is also a way to check that all the Gaussian operation functions run without
    # errors.
    N = 4
    v0 = randgaussianstate(N)
    @testset "displacement" begin
        v = displace(v0, rand(ComplexF64, N))
        displace!(v0, rand(ComplexF64, N))
        @test purity(v) ≈ purity(v0)
    end
    @testset "phase shift" begin
        v = phaseshift(v0, rand(N))
        phaseshift!(v0, rand(N))
        @test purity(v) ≈ purity(v0)
    end
    @testset "1-mode squeezing" begin
        v = squeeze(v0, rand(ComplexF64, N))
        @test purity(v) ≈ purity(v0)
    end
    @testset "2-mode squeezing" begin
        v = squeeze2(v0, rand(), 1, 3)
        squeeze2!(v, rand(), 2, 4)
        squeeze2!(v, rand(), 2, 3)
        squeeze2!(v, rand(), 1, 4)
        @test purity(v) ≈ purity(v0)
    end
    @testset "beam splitter" begin
        v = beamsplitter(v0, rand(), 1, 3)
        beamsplitter!(v, rand(), 2, 4)
        beamsplitter!(v, rand(), 2, 3)
        beamsplitter!(v, rand(), 1, 4)
        @test purity(v) ≈ purity(v0)
    end
end

@testset verbose = true "Photon number counting" begin
    @testset "coherent state" begin
        n = 2
        α = rand(ComplexF64, n)
        g = vacuumstate(n)
        displace!(g, α)
        @test number(g) ≈ norm(α)^2
    end
    @testset "thermal state" begin
        n = 4
        β = 2
        ω = rand(n)
        g = thermalstate(n, β, ω)
        @test number(g) ≈ sum(1 ./ expm1.(β .* ω))
    end
    @testset "with beam splitter" begin
        n = 4
        g = vacuumstate(n)
        squeeze!(g, rand(ComplexF64, n))
        pre_splitter_N = number(g)
        beamsplitter!(g, rand(), 2, 4)
        post_splitter_N = number(g)
        @test pre_splitter_N ≈ post_splitter_N
    end
end

include("decompositions.jl")
@testset "Williamson decomposition" begin
    n = 8
    A = randposdef(Float64, 2n)
    @test williamson_check(Symmetric(A))

    A = randposdef(BigFloat, 2n)
    @test williamson_check(Symmetric(A))
end

@testset verbose = true "Takagi-Autonne decomposition" begin
    n = 8
    @testset "with a real matrix" begin
        a = rand(n, n)
        a = (a+transpose(a))/2
        @test takagiautonne_check(a)
    end
    @testset "with an almost diagonal matrix" begin
        a = diagm(0 => rand(ComplexF64, n))
        a[n - 1, n] = 1e-16
        a[2, 1] = 3e-16
        @test takagiautonne_check(a)
    end
    @testset "with a real matrix times a phase" begin
        a = rand(n, n)
        a = cispi(rand()) .* (a+transpose(a))/2
        @test takagiautonne_check(a)
    end
    @testset "with a non-real matrix" begin
        a = rand(ComplexF64, n, n)
        a = (a+transpose(a))/2
        @test takagiautonne_check(a)
    end
    @testset "with a BigFloat matrix" begin
        a = big.(rand(ComplexF64, n, n))
        a = (a+transpose(a))/2
        @test takagiautonne_check(a)
    end
end

@testset "Euler decomposition" begin
    n = 8
    A = randsymplectic(Float64, n)
    @test euler_check(A)
end

@testset "Unitary-to-symplectic conversion" begin
    n = 9
    h = Hermitian(rand(n, n))
    U = cis(h)  # = exp(im * h)
    @assert U' * U ≈ U * U' ≈ I

    S = GaussianStates.unitary_to_symplectic(U)
    @test issymplectic(GaussianStates.permute_to_xpxp(S))
    @test U == GaussianStates.symplectic_to_unitary(S)
end
