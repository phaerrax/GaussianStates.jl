module GaussianStates

using LinearAlgebra

export GaussianState,
    vacuumstate,
    nmodes,
    is_valid_covariance_matrix,
    thermalstate,
    random_symplectic,
    random_gaussianstate,
    number
include("states.jl")

export displace,
    displace!,
    phaseshift,
    phaseshift!,
    squeeze,
    squeeze!,
    squeeze2,
    squeeze2!,
    beamsplitter,
    beamsplitter!,
    lossybeamsplitter,
    lossybeamsplitter!,
    partialtrace
include("operations.jl")

export issymplectic, williamson, randposdef, takagiautonne
include("decompositions.jl")

end # module GaussianStates
