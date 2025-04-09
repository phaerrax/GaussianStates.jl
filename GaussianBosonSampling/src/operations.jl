function symplectic_transform(g::GaussianState, d::AbstractVector, F::AbstractMatrix)
    return GaussianState(F * g.first_moments + d, F * g.covariance_matrix * transpose(F))
end
# Simpler versions for convenience
function symplectic_transform(g::GaussianState, F::AbstractMatrix)
    return GaussianState(F * g.first_moments, F * g.covariance_matrix * transpose(F))
end
function symplectic_transform(g::GaussianState, d::AbstractVector)
    return GaussianState(g.first_moments .+ d, g.covariance_matrix)
end

"""
    displace(g::GaussianState, α)

Transform the Gaussian state `g` by applying the displacement operator on all modes, with
parameter `α[k]` on mode `k`.
"""
function displace(g::GaussianState, α)
    d = sqrt(2) .* [real.(α); imag.(α)]
    n = length(α)
    invpermute!(d, [1:2:n; 2:2:n])  # quicker than `permute_to_xpxp`
    return symplectic_transform(g, d)
end

"""
    displace(g::GaussianState, α, k)

Transform the Gaussian state `g` by applying the displacement operator on the `k`-th mode
with parameter `α`.
"""
function displace(g::GaussianState, α, k)
    d = zeros(2nmodes(g))
    d[2k - 1] = sqrt(2) * real(α)
    d[2k] = sqrt(2) * imag(α)
    return symplectic_transform(g, d)
end

_2drotationmatrix(θ) = [[cos(θ) -sin(θ)] [sin(θ) cos(θ)]]

"""
    phaseshift(g::GaussianState, ϕ)

Transform the Gaussian state `g` by applying to each mode `k` a phase shift `ϕ[k]`.
"""
function phaseshift(g::GaussianState, ϕ)
    n = nmodes(g)
    F = zeros(2n, 2n)
    for k in 1:n
        i = 2k - 1
        F[i:(i + 1), i:(i + 1)] .= _2drotationmatrix(ϕ[k])
    end
    return symplectic_transform(g, F)
end

"""
    phaseshift(g::GaussianState, ϕ)

Transform the Gaussian state `g` by applying to each mode `k` a phase shift `ϕ[k]`.
"""
function phaseshift(g::GaussianState, ϕ, k)
    F = float.(I(2nmodes(g)))
    F[(2k - 1):(2k), (2k - 1):(2k)] .= _2drotationmatrix(ϕ)
    return symplectic_transform(g, F)
end

function _squeezematrix(ζ)
    θ = angle(ζ)
    r = abs(ζ)
    S = [[cos(θ), sin(θ)] [sin(θ), -cos(θ)]]
    return cosh(r) .* I(2) .- sinh(r) .* S
end

"""
    squeeze(g::GaussianState, ζ, k)

Apply a squeezing transformation on the `k`-th mode with parameter `ζ`.
"""
function squeeze(g::GaussianState, ζ, k)
    F = Matrix{Float64}(I, 2nmodes(g), 2nmodes(g))
    F[(2k - 1):(2k), (2k - 1):(2k)] .= _squeezematrix(ζ)
    return symplectic_transform(g, F)
end

"""
    squeeze(g::GaussianState, ζ)

Transform the Gaussian state `g` by squeezing each mode `k` with parameter `ζ[k]`.
"""
function squeeze(g::GaussianState, ζ)
    n = nmodes(g)
    F = zeros(2n, 2n)
    for k in 1:n
        i = 2k - 1
        F[i:(i + 1), i:(i + 1)] .= _squeezematrix(ζ[k])
    end
    return symplectic_transform(g, F)
end

"""
    squeeze2(g::GaussianState, ζ, k1, k2)

Apply a two-mode squeezing transformations on modes `k1` and `k2` with parameter `ζ`.
"""
function squeeze2(g::GaussianState, ζ, k1, k2)
    θ = angle(ζ)
    r = abs(ζ)

    F = Matrix{Float64}(I, 2nmodes(g), 2nmodes(g))

    S = [[cos(θ), sin(θ)] [sin(θ) -cos(θ)]]
    F[(2k1 - 1):(2k1), (2k1 - 1):(2k1)] .= cosh(r) .* I(2)
    F[(2k1 - 1):(2k1), (2k2 - 1):(2k2)] .= -sinh(r) .* S
    F[(2k2 - 1):(2k2), (2k1 - 1):(2k1)] .= -sinh(r) .* S
    F[(2k2 - 1):(2k2), (2k2 - 1):(2k2)] .= cosh(r) .* I(2)

    return symplectic_transform(g, F)
end

"""
    partialtrace(g::GaussianState, ns)

Compute the partial trace of the Gaussian state `g` over the modes `ns`.
"""
function partialtrace(g::GaussianState, ns)
    idxs = reduce(vcat, [[2i - 1, 2i] for i in ns])
    redvec = g.first_moments[setdiff(1:end, idxs)]
    redmat = g.covariance_matrix[setdiff(1:end, idxs), setdiff(1:end, idxs)]
    return GaussianState(redvec, redmat)
end

"""
    beamsplitter(g::GaussianState, transmittivity, k1, k2)

Transform the Gaussian state `g` with a beam splitter on modes `k1` and `k2` with the
specified `transmittivity`.
"""
function beamsplitter(g::GaussianState, transmittivity, k1, k2)
    η = transmittivity
    F = Matrix{Float64}(I, 2nmodes(g), 2nmodes(g))
    F[(2k1 - 1):(2k1), (2k1 - 1):(2k1)] .= sqrt(η) .* I(2)
    F[(2k1 - 1):(2k1), (2k2 - 1):(2k2)] .= sqrt(1 - η) .* I(2)
    F[(2k2 - 1):(2k2), (2k1 - 1):(2k1)] .= -sqrt(1 - η) .* I(2)
    F[(2k2 - 1):(2k2), (2k2 - 1):(2k2)] .= sqrt(η) .* I(2)
    return symplectic_transform(g, F)
end

"""
    lossybeamsplitter(g::GaussianState, transmittivity, loss, k1, k2)

Transform the Gaussian state `g` with lossy a beam splitter on modes `k1` and `k2` with
given `transmittivity` and `loss` parameters.
"""
function lossybeamsplitter(g::GaussianState, transmittivity, loss, k1, k2)
    # We generate the lossy beam splitter between modes k1 and k2 as follows:
    # 1. add two new modes a1 and a2 in the vacuum
    # 2. interaction between k1 and k2
    # 3. interaction between k1 and a1
    # 4. interaction between k2 and a2
    # 5. partial trace over a1 and a2
    g_ext = join(g, vacuumstate(2))

    aux1 = nmodes(g) + 1  # indices of the auxiliary modes
    aux2 = nmodes(g) + 2

    g_ext = beamsplitter(g_ext, transmittivity, k1, k2)
    g_ext = beamsplitter(g_ext, loss, k1, aux1)
    g_ext = beamsplitter(g_ext, loss, k2, aux2)
    return partialtrace(g_ext, [aux1, aux2])
end
