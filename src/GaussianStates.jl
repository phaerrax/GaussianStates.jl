module GaussianStates

using LinearAlgebra

export GaussianState,
    firstmoments,
    covariancematrix,
    vacuumstate,
    nmodes,
    is_valid_covariance_matrix,
    thermalstate,
    randsymplectic,
    randgaussianstate,
    number,
    purity
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

export issymplectic, williamson, randposdef, takagiautonne, euler
include("decompositions.jl")

end # module GaussianStates
