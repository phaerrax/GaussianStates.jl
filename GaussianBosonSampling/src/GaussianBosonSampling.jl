module GaussianBosonSampling

using Permutations, LinearAlgebra, Random

export GaussianState,
    vacuumstate,
    nmodes,
    is_valid_covariance_matrix,
    thermalstate,
    random_symplectic,
    random_gaussianstate,
    number
include("gaussian_states.jl")

export displace,
    phaseshift, squeeze, squeeze2, partialtrace, beamsplitter, lossybeamsplitter
include("operations.jl")

end # module GaussianBosonSampling
