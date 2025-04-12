using Test
using GaussianStates, LinearAlgebra

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
    @test williamson_check(8)
end

@testset verbose = true "Takagi-Autonne decomposition" begin
    n = 8
    @testset "with a real matrix" begin
        a = rand(n, n)
        a = (a + transpose(a)) / 2
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
        a = cispi(rand()) .* (a + transpose(a)) / 2
        @test takagiautonne_check(a)
    end
    @testset "with a non-real matrix" begin
        a = rand(ComplexF64, n, n)
        a = (a + transpose(a)) / 2
        @test takagiautonne_check(a)
    end
end

@testset "Euler decomposition" begin
    @test euler_check(8)
end
