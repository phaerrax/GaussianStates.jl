# Generic in-place functions for Gaussian transformations

function symplectic_transform!(g::GaussianState, d::AbstractVector, F::AbstractMatrix)
    g.first_moments .= F * g.first_moments .+ d
    g.covariance_matrix .= F * g.covariance_matrix * transpose(F)
    return g
end
function symplectic_transform!(g::GaussianState, F::AbstractMatrix)
    g.first_moments .= F * g.first_moments
    g.covariance_matrix .= F * g.covariance_matrix * transpose(F)
    return g
end
function symplectic_transform!(g::GaussianState, d::AbstractVector)
    g.first_moments .+= d
    return g
end

# Displacement

"""
    displace!(g::GaussianState, α::AbstractVector)

Transform the Gaussian state `g` by applying the displacement operator on all modes, with
parameter `α[k]` on mode `k`.
"""
function displace!(g::GaussianState, α::AbstractVector)
    d = sqrt(2) .* [real.(α); imag.(α)]
    return symplectic_transform!(g, permute_to_xpxp(d))
end

"""
    displace!(g::GaussianState, α, k)

Transform the Gaussian state `g` by applying the displacement operator on the `k`-th mode
with parameter `α`.
"""
function displace!(g::GaussianState, α, k)
    d = zeros(2nmodes(g))
    d[2k - 1] = sqrt(2) * real(α)
    d[2k] = sqrt(2) * imag(α)
    return symplectic_transform!(g, d)
end

"""
    displace(g::GaussianState, α::AbstractVector)
    displace(g::GaussianState, α, k)

Variants of [`displace!`](@ref) that return a transformed copy of `g` leaving `g` itself
unmodified.
"""
function displace end

function displace(g::GaussianState, α::AbstractVector)
    h = GaussianState(copy(g.first_moments), copy(g.covariance_matrix))
    return displace!(h, α)
end

function displace(g::GaussianState, α, k)
    h = GaussianState(copy(g.first_moments), copy(g.covariance_matrix))
    return displace!(h, α, k)
end

# Phase shift

_2drotationmatrix(θ) = [
    cos(θ) -sin(θ)
    sin(θ) cos(θ)
]

"""
    phaseshift!(g::GaussianState, ϕ::AbstractVector)

Transform the Gaussian state `g` by applying to each mode `k` a phase shift `ϕ[k]`.
"""
function phaseshift!(g::GaussianState, ϕ::AbstractVector)
    n = nmodes(g)
    F = zeros(2n, 2n)
    for k in 1:n
        i = 2k - 1
        F[i:(i + 1), i:(i + 1)] .= _2drotationmatrix(ϕ[k])
    end
    return symplectic_transform!(g, F)
end

"""
    phaseshift!(g::GaussianState, ϕ)

Transform the Gaussian state `g` by applying to each mode `k` a phase shift `ϕ[k]`.
"""
function phaseshift!(g::GaussianState, ϕ, k)
    F = float.(I(2nmodes(g)))
    F[(2k - 1):(2k), (2k - 1):(2k)] .= _2drotationmatrix(ϕ)
    return symplectic_transform!(g, F)
end

"""
    phaseshift(g::GaussianState, ϕ::AbstractVector)
    phaseshift(g::GaussianState, ϕ, k)

Variants of [`phaseshift!`](@ref) that return a transformed copy of `g` leaving `g` itself
unmodified.
"""
function phaseshift end

function phaseshift(g::GaussianState, ϕ::AbstractVector)
    h = GaussianState(copy(g.first_moments), copy(g.covariance_matrix))
    return phaseshift!(h, ϕ)
end

function phaseshift(g::GaussianState, ϕ, k)
    h = GaussianState(copy(g.first_moments), copy(g.covariance_matrix))
    return phaseshift!(h, ϕ, k)
end

# Squeezing

function _squeezematrix(ζ)
    θ = angle(ζ)
    r = abs(ζ)
    S = [
        cos(θ) sin(θ)
        sin(θ) -cos(θ)
    ]
    return cosh(r) .* I(2) .- sinh(r) .* S
end

single_mode_squeezed_vacuum_coefficients = """
With this definition, the squeezed vacuum state (on a single mode) is written in the
eigenbasis of the number operator, with ``ζ = r e^(iθ)``, as

```math
   1     +∞                   √(2n)!
  ────────  ∑ (e^(iθ) tanh(r))ⁿ ────── |2n⟩
  √cosh(r) n=0                   2ⁿn!
```
"""

"""
    squeeze!(g::GaussianState, ζ::AbstractVector)

Transform the Gaussian state `g` by squeezing each mode `k` with parameter `ζ[k]`, by
applying the operator

```math
S(ζ) = exp(ζ/2 (a*)^2 - ζ̄/2 a^2)
```

on each mode.

$single_mode_squeezed_vacuum_coefficients
"""
function squeeze!(g::GaussianState, ζ::AbstractVector)
    n = nmodes(g)
    F = zeros(2n, 2n)
    for k in 1:n
        i = 2k - 1
        F[i:(i + 1), i:(i + 1)] .= _squeezematrix(ζ[k])
    end
    return symplectic_transform!(g, F)
end

"""
    squeeze!(g::GaussianState, ζ, k)

Apply a squeezing transformation on the `k`-th mode with parameter `ζ`, by applying the
operator

```math
S(ζ) = exp(ζ/2 (a*)^2 - ζ̄/2 a^2)
```

on the selected mode.

$single_mode_squeezed_vacuum_coefficients
"""
function squeeze!(g::GaussianState, ζ, k)
    F = Matrix{Float64}(I, 2nmodes(g), 2nmodes(g))
    F[(2k - 1):(2k), (2k - 1):(2k)] .= _squeezematrix(ζ)
    return symplectic_transform!(g, F)
end

"""
    squeeze(g::GaussianState, ζ::AbstractVector)
    squeeze(g::GaussianState, ζ, k)

Variants of [`squeeze!`](@ref) that return a transformed copy of `g` leaving `g` itself
unmodified.
"""
function squeeze end

function squeeze(g::GaussianState, ζ::AbstractVector)
    h = GaussianState(copy(g.first_moments), copy(g.covariance_matrix))
    return squeeze!(h, ζ)
end

function squeeze(g::GaussianState, ζ, k)
    h = GaussianState(copy(g.first_moments), copy(g.covariance_matrix))
    return squeeze!(h, ζ, k)
end

function _squeeze2matrix(ζ)
    θ = angle(ζ)
    r = abs(ζ)

    F = Matrix{Float64}(cosh(r) * I, 4, 4)

    S = [
        cos(θ) sin(θ)
        sin(θ) -cos(θ)
    ]
    F[1:2, 3:4] .= -sinh(r) .* S
    F[3:4, 1:2] .= -sinh(r) .* S

    return F
end
"""
    squeeze2!(g::GaussianState, ζ, k1, k2)

Apply a two-mode squeezing transformation on modes `k1` and `k2` with parameter `ζ`, by
applying the operator

```math
S₂(ζ) = exp(ζ (a* ⊗ a*) - ζ̄ (a ⊗ a))
```

on the selected modes.

With this definition, the two-mode squeezed vacuum state is written in the eigenbasis of
the number operator, with ``ζ = r e^(iθ)``, as

```math
   1    +∞
  ───────  ∑ (e^(iθ) tanh(r))ⁿ |n⟩ ⊗ |n⟩
  cosh(r) n=0
```
"""
function squeeze2!(g::GaussianState, ζ, k1, k2)
    f = _squeeze2matrix(ζ)

    F = Matrix{Float64}(I, 2nmodes(g), 2nmodes(g))
    F[(2k1 - 1):(2k1), (2k1 - 1):(2k1)] .= f[1:2, 1:2]
    F[(2k1 - 1):(2k1), (2k2 - 1):(2k2)] .= f[1:2, 3:4]
    F[(2k2 - 1):(2k2), (2k1 - 1):(2k1)] .= f[3:4, 1:2]
    F[(2k2 - 1):(2k2), (2k2 - 1):(2k2)] .= f[3:4, 3:4]

    return symplectic_transform!(g, F)
end

"""
    squeeze2(g::GaussianState, ζ, k1, k2)

Variant of [`squeeze2!`](@ref) that returns a transformed copy of `g` leaving `g` itself
unmodified.
"""
function squeeze2(g::GaussianState, ζ, k1, k2)
    h = GaussianState(copy(g.first_moments), copy(g.covariance_matrix))
    return squeeze2!(h, ζ, k1, k2)
end

# Partial trace

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

# Beam splitters

function _beamsplittermatrix(η)
    F = Matrix{Float64}(sqrt(η) * I, 4, 4)
    F[1:2, 3:4] .= sqrt(1 - η) .* I(2)
    F[3:4, 1:2] .= -sqrt(1 - η) .* I(2)
    return F
end

"""
    beamsplitter!(g::GaussianState, transmittivity, k1, k2)

Transform the Gaussian state `g` with a beam splitter on modes `k1` and `k2` with the
specified `transmittivity`.
"""
function beamsplitter!(g::GaussianState, transmittivity, k1, k2)
    f = _beamsplittermatrix(transmittivity)

    F = Matrix{Float64}(I, 2nmodes(g), 2nmodes(g))
    F[(2k1 - 1):(2k1), (2k1 - 1):(2k1)] .= f[1:2, 1:2]
    F[(2k1 - 1):(2k1), (2k2 - 1):(2k2)] .= f[1:2, 3:4]
    F[(2k2 - 1):(2k2), (2k1 - 1):(2k1)] .= f[3:4, 1:2]
    F[(2k2 - 1):(2k2), (2k2 - 1):(2k2)] .= f[3:4, 3:4]

    return symplectic_transform!(g, F)
end

"""
    beamsplitter(g::GaussianState, transmittivity, k1, k2)

Variant of [`beamsplitter!`](@ref) that returns a transformed copy of `g` leaving `g` itself
unmodified.
"""
function beamsplitter(g::GaussianState, transmittivity, k1, k2)
    h = GaussianState(copy(g.first_moments), copy(g.covariance_matrix))
    return beamsplitter!(h, transmittivity, k1, k2)
end

"""
    lossybeamsplitter!(g::GaussianState, transmittivity, loss, k1, k2)

Transform the Gaussian state `g` with lossy a beam splitter on modes `k1` and `k2` with
given `transmittivity` and `loss` parameters.
"""
function lossybeamsplitter!(g::GaussianState, transmittivity, loss, k1, k2)
    g_lbs = lossybeamsplitter(g, transmittivity, loss, k1, k2)

    # Replace the original g's moments with g_lbs's ones.
    g.first_moments .= g_lbs.first_moments
    g.covariance_matrix .= g_lbs.covariance_matrix
    return g
end

"""
    lossybeamsplitter(g::GaussianState, transmittivity, loss, k1, k2)

Variant of [`lossybeamsplitter!`](@ref) that returns a transformed copy of `g` leaving `g`
itself unmodified.
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

    beamsplitter!(g_ext, transmittivity, k1, k2)
    beamsplitter!(g_ext, loss, k1, aux1)
    beamsplitter!(g_ext, loss, k2, aux2)

    return partialtrace(g_ext, [aux1, aux2])
end
