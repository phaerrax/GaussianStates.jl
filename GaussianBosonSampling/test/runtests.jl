using Test
using GaussianBosonSampling, LinearAlgebra

include("decompositions.jl")
@testset "Williamson decomposition" begin
    @test williamson_check(8)
end
