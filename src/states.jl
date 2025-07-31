"""
    promote_array(arrays...)

Return `arrays` with their element types promoted to their join.

# Example

```julia-repl
julia> promote_array(big.([1.0, 2]), zeros(Int, 2, 2))
(BigFloat[1.0, 2.0], BigFloat[0.0 0.0; 0.0 0.0])
```
"""
function promote_array(arrays...)
    supertype = Base.promote_eltype(arrays...)
    tuple([convert(Array{supertype}, array) for array in arrays]...)
end

symplectic_matrix_latex = """
```math
\\sympmat = I_n ⊗  
\\begin{pmatrix}
  0 & 1\\\\
  -1 & 0
\\end{pmatrix}
```
"""

"""
    _symplectic_matrix(n)

Return the ``2n × 2n`` symplectic matrix

```
     ⎛  0  1 ⎞
Iₙ ⊗ ⎜       ⎟
     ⎝ -1  0 ⎠
```

# Example

```julia-repl
julia> GaussianStates._symplectic_matrix(2)
4×4 Matrix{Int64}:
  0  1   0  0
 -1  0   0  0
  0  0   0  1
  0  0  -1  0
```
"""
_symplectic_matrix(n) = kron(I(n), [[0, -1] [1, 0]])

"""
    is_valid_covariance_matrix(σ; atol, rtol)

Test whether the matrix `σ` satisfies the conditions to be a covariance matrix for a
Gaussian state, i.e. is a ``2n × 2n`` symmetric matrix such that ``σ > 0`` and
``σ + i\\sympmat ≥ 0``.

Keyword arguments are forwarded to `isapprox` to adjust the numerical thresholds of the
inexact equality comparisons; `atol` defaults to `eps(eltype(σ)) * norm(σ)` when comparing
with zero.
"""
function is_valid_covariance_matrix(σ; kwargs...)
    if size(σ, 1) != size(σ, 2)
        println("Not a square matrix.")
        return false
    end
    n = size(σ, 1)
    if !iseven(n)
        println("Odd number of rows.")
        return false
    end

    # It usually happens after some numerical calculations that σ is slightly asymmetric
    # and ‖σᵀ-σ‖ is "practically" zero, but not precisely zero. In this cases
    # calling `isposdef` returns false since the function works only if the matrix is
    # exactly symmetric (or Hermitian, more in general), otherwise it always gives false.
    # So, first we check whether σ is reasonably symmetric, at some numerical precision,
    # then we construct an explicitly symmetric version of it and feed it to `isposdef`.
    if !isapprox(σ, transpose(σ); kwargs...)
        println("Not a symmetric matrix.")
        return false
    end
    if !isposdef(Symmetric(σ))
        println("Not a positive definite matrix.")
        return false
    end

    # Uncertainty relation.
    # We need to check now whether σ + iΩ is positive semi-definite, so `isposdef` won't
    # work here since some of its eigenvalues may be zero.
    # In practice, as usual, instead of zero we'll find something which is slightly zero,
    # possibly negative, so we can't just use `eigmin(σ + iΩ) ≥ 0` but we must allow
    # some tolerance. For example, the squeezed state
    #
    #   julia> rs = [0.5, 1, 1.5];
    #   julia> σ = Diagonal(reduce(vcat, [[exp(r); exp(-r)] for r in rs]));
    #
    # produces a small negative eigenvalue in this test:
    #
    #   julia> eigmin(Hermitian(σ + im * _symplectic_matrix(3)))
    #   -2.700059754488065e-16
    #
    # As before, we make the argument explicitly Hermitian before passing it to `eigmin`
    # so that it knows that the matrix is Hermitian (since we have already checked that σ
    # is "reasonably symmetric" we can be sure that σ + iΩ is "reasonably Hermitian") and
    # its eigenvalues should be real.
    if eigmin(Hermitian(σ .+ im .* _symplectic_matrix(div(n, 2)))) <
        -get(kwargs, :atol, eps(eltype(σ)) * norm(σ))
        println("Does not satisfy the uncertainty relation.")
        return false
    end

    return true
end

"""
    GaussianState

Represent a Gaussian state by storing only its covariance matrix and the vector of its
first moments.  By default states are stored in the xpxp representation.

A `GaussianState` object can be constructed by specifying first moments and covariance
matrix as `GaussianState(r, σ)` or just the covariance matrix as `GaussianState(σ)`.
"""
struct GaussianState
    first_moments::AbstractVector
    covariance_matrix::AbstractMatrix
    function GaussianState(r, σ)
        @assert all(size(σ) .== size(r))
        @assert iseven(size(σ, 1))
        return new(promote_array(r, σ)...)
    end
end

GaussianState(σ::AbstractMatrix) = GaussianState(zeros(size(σ, 1)), σ)

Base.eltype(g::GaussianState) = eltype(g.first_moments)

"""
    nmodes(g::GaussianState)

Return the number of modes `g` is defined on.
"""
nmodes(g::GaussianState) = div(length(g.first_moments), 2)

function Base.:(==)(g::GaussianState, h::GaussianState)
    (g.first_moments == h.first_moments) && (g.covariance_matrix == h.covariance_matrix)
end
function Base.isapprox(g::GaussianState, h::GaussianState; kwargs...)
    isapprox(g.first_moments, h.first_moments; kwargs...) &&
        isapprox(g.covariance_matrix, h.covariance_matrix; kwargs...)
end

"""
    vacuumstate([T = Float64, ]n)

Return the vacuum Gaussian state of type `T` on `n` modes
"""
function vacuumstate(::Type{T}, n) where {T<:Number}
    GaussianState(zeros(T, 2n), Matrix{T}(I, 2n, 2n))
end
vacuumstate(n) = vacuumstate(Float64, n)

"""
    thermalstate(n, β, ω::AbstractVector)

Return the thermal state on `n` modes with inverse temperature `β` and frequency `ω[k]`
for each mode `k`.
"""
function thermalstate(n, β, ω::AbstractVector)
    λ = 1 .+ 2 ./ expm1.(β * ω)
    return GaussianState(zeros(2n), diagm(permute_to_xpxp([λ; λ])))
end

"""
    number(g::GaussianState)

Return the mean number of photons in the state `g`.
"""
function number(g::GaussianState)
    return tr(g.covariance_matrix) / 4 + (norm(g.first_moments)^2) / 2 - nmodes(g) / 2
end

"""
    purity(g::GaussianState)

Return the purity of the state `g`.
"""
function purity(g::GaussianState)
    return inv(sqrt(det(g.covariance_matrix)))
end

function permute_to_xxpp(v::AbstractVector)
    n = div(length(v), 2)
    xxpp = [1:2:(2n); 2:2:(2n)]
    # This `xxpp` implements the following permutation:
    #
    #   ⎛ 1 2 3 4 ...  n-1   n  n+1 n+2 ... 2n-2 2n-1 2n ⎞
    #   ⎝ 1 3 5 7 ... 2n-3 2n-1  2   4  ... 2n-4 2n-2 2n ⎠
    #
    # i.e. it shuffles
    #
    #   (x_1, p_1, x_2, p_2, ..., x_n, p_n).
    #
    # into
    #
    #   (x_1, x_2, ... x_n, p_1, p_2, ... p_n).
    return v[xxpp]
end

function permute_to_xxpp(m::AbstractMatrix)
    n = div(size(m, 1), 2)
    xxpp = [1:2:(2n); 2:2:(2n)]
    return m[xxpp, xxpp]
end

function permute_to_xpxp(v::AbstractVector)
    # We use the inverse of the permutation defined in `permute_to_xxpp`.
    n = div(length(v), 2)
    xpxp = invperm([1:2:(2n); 2:2:(2n)])
    return v[xpxp]
end

function permute_to_xpxp(m::AbstractMatrix)
    # We use the inverse of the permutation defined in `permute_to_xxpp`.
    n = div(size(m, 1), 2)
    xpxp = invperm([1:2:(2n); 2:2:(2n)])
    return m[xpxp, xpxp]
end

"""
    randsymplectic([T = Float64, ]n)

Generate a random ``2n × 2n`` real symplectic matrix of element type `T` such that
``S \\sympmat \\transpose{S} = \\sympmat``, with

$symplectic_matrix_latex
"""
function randsymplectic(::Type{T}, n) where {T<:Number}
    A = rand(n, n)
    B = rand(n, n)
    C = rand(n, n)
    m = Matrix{T}(undef, 2n, 2n)
    m[1:n, 1:n] .= A
    m[1:n, (n + 1):(2n)] .= Symmetric(B)
    m[(n + 1):(2n), 1:n] .= Symmetric(C)
    m[(n + 1):(2n), (n + 1):(2n)] .= -transpose(A)

    m /= norm(m)
    # We divide by the norm otherwise the matrix elements blow up when the dimension gets
    # big, e.g. randsymplectic(100) would return a matrix whose elements are ~1e28.

    return permute_to_xpxp(exp(m))
end

randsymplectic(n) = randsymplectic(Float64, n)

"""
    randgaussianstate([T = Float64, ]n, λ; pure=false, displace=true)

Generate a random `n`-mode Gaussian state with element type `T`, in the xpxp representation.

The state is generated from the Williamson decomposition, by drawing first the `n`
symplectic eigenvalues ``d_i`` and then applying a random symplectic transformation.
Each ``d_i`` is drawn from an exponential distribution with rate `λ[i]`, which defaults to
one (`λ`'s elements must be convertible to `T`).
If `displace` is `true` then a random displacement in ``[-1, 1]`` is applied on each mode.
The returned state is generally not pure, unless `pure` is `false` which forces the
generation of a pure state.
"""
function randgaussianstate(
    ::Type{T}, n, λ=ones(T, n); pure=false, displace=true
) where {T<:Number}
    S = randsymplectic(T, n)
    σ = if pure
        S * transpose(S)
    else
        λ = convert.(T, λ)
        rand_sp_evals = one(T) .- log.(rand(T, n)) ./ λ
        # -log(x)/λ ~ Exp(λ) if x ~ U(0,1)
        D = Diagonal(permute_to_xpxp([rand_sp_evals; rand_sp_evals]))
        S * D * transpose(S)
    end
    r = if displace
        2 .* rand(T, 2n) .- one(T)  # uniform in [-1,1]
    else
        zeros(T, 2n)
    end
    return GaussianState(r, σ)
end

function randgaussianstate(n, λ=ones(n); pure=false, displace=true)
    randgaussianstate(Float64, n, λ; pure=pure, displace=displace)
end

"""
    join(g1::GaussianState, g2::GaussianState)

Return a new Gaussian state with `g1` on modes `1` to `nmodes(g1)` and `g2` on the
remaining ones.
"""
function Base.join(g1::GaussianState, g2::GaussianState)
    n1 = nmodes(g1)
    n2 = nmodes(g2)
    r = [g1.first_moments; g2.first_moments]
    σ = zeros(2(n1 + n2), 2(n1 + n2))
    σ[1:(2n1), 1:(2n1)] .= g1.covariance_matrix
    σ[(2n1 + 1):(2n1 + 2n2), (2n1 + 1):(2n1 + 2n2)] .= g2.covariance_matrix
    return GaussianState(r, σ)
end
