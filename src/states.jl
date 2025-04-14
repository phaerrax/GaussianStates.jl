"""
    Ω(n)

Return the ``2n × 2n`` symplectic matrix

```
     ⎛  0  1 ⎞
Iₙ ⊗ ⎜       ⎟
     ⎝ -1  0 ⎠
```

# Example

```julia-repl
julia> Ω(2)
4×4 Matrix{Int64}:
  0  1   0  0
 -1  0   0  0
  0  0   0  1
  0  0  -1  0
```
"""
Ω(n) = kron(I(n), [[0, -1] [1, 0]])

"""
    is_valid_covariance_matrix(σ; atol, rtol)

Test whether the matrix `σ` satisfies the conditions to be a covariance matrix for a
Gaussian state, i.e. is a ``2n × 2n`` symmetric matrix such that ``σ > 0`` and
``σ + iΩ ≥ 0``.

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
    #   julia> eigmin(Hermitian(σ + im * Ω(3)))
    #   -2.700059754488065e-16
    #
    # As before, we make the argument explicitly Hermitian before passing it to `eigmin`
    # so that it knows that the matrix is Hermitian (since we have already checked that σ
    # is "reasonably symmetric" we can be sure that σ + iΩ is "reasonably Hermitian") and
    # its eigenvalues should be real.
    if eigmin(Hermitian(σ .+ im .* Ω(div(n, 2)))) <
        -get(kwargs, :atol, eps(eltype(σ)) * norm(σ))
        println("Does not satisfy the uncertainty relation.")
        return false
    end

    return true
end

"""
    GaussianState

Represent a Gaussian state by storing only its covariance matrix and the vector of its
first moments.
By default states are stored in the xpxp representation.
"""
struct GaussianState
    first_moments::AbstractVector
    covariance_matrix::AbstractMatrix
    function GaussianState(r, σ)
        @assert all(size(σ) .== size(r))
        @assert iseven(size(σ, 1))
        return new(r, σ)
    end
end

nmodes(g::GaussianState) = div(length(g.first_moments), 2)

"""
    vacuumstate(n)

Return the vacuum state on `n` modes.
"""
vacuumstate(n) = GaussianState(zeros(2n), Matrix{Float64}(I, 2n, 2n))

"""
    thermalstate(n, β, ω::AbstractVector)

Return the thermal state on `n` modes with inverse temperature `β` and frequency `ω[k]`
for each mode `k`.
"""
function thermalstate(n, β, ω::AbstractVector)
    λ = 1 .+ 2 ./ expm1.(β * ω)
    return GaussianState(zeros(2n), diagm(permute_to_xpxp([λ; λ])))
end

function number(g::GaussianState)
    return 1 / 4 * tr(g.covariance_matrix) + 1 / 2 * norm(g.first_moments)^2 - nmodes(g) / 2
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
	randsymplectic(n)

Generate a random ``2n × 2n`` real symplectic matrix such that ``S Ω Sᵀ = Ω`` with

```
Ω = Iₙ ⊗  ⎛  0  1 ⎞
            ⎝ -1  0 ⎠
```
"""
function randsymplectic(n)
    A = rand(n, n)
    B = rand(n, n)
    C = rand(n, n)
    m = Matrix{Float64}(undef, 2n, 2n)
    m[1:n, 1:n] .= A
    m[1:n, (n + 1):(2n)] .= Symmetric(B)
    m[(n + 1):(2n), 1:n] .= Symmetric(C)
    m[(n + 1):(2n), (n + 1):(2n)] .= -transpose(A)

    m /= norm(m)
    # We divide by the norm otherwise the matrix elements blow up when the dimension gets
    # big, e.g. randsymplectic(100) would return a matrix whose elements are ~1e28.

    return permute_to_xpxp(exp(m))
end

#=
This method seems to generate covariance matrices that do not fulfill the uncertainty
relation σ - i/2 Ω ≥ 0.
Commented out until I find out what's wrong with it.

"""
    random_gaussianstate(n)

Generate a random `n`-mode Gaussian state in the xpxp representation.
"""
function random_gaussianstate(n::Int)
    sp_evals = randexp(n) .+ 1
    D = Diagonal(reduce(vcat, [[x, x] for x in sp_evals]))
    S = random_symplectic(n)
    σ = permute_to_xpxp(S * D * transpose(S))
    σ ./= norm(σ)  # keep numbers low
    r = 2 .* rand(2n) .- 1  # uniform in [-1,1]
    return GaussianState(r, σ)
end
=#

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
