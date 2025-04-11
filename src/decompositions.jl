"""
    randposdef(n)

Generate an ``n × n`` real positive-definite matrix.
"""
function randposdef(n)
    A = rand(n, n)
    hermitianpart!(A)  # overwrite with 1/2 (A + A')

    # Since a symmetric diagonally dominant matrix is symmetric positive-definite and we
    # have A[i, j] < 1 by construction, we can ensure positive-definiteness by adding nI.
    return A .+ n * I(n)
end

"""
    issymplectic(M)

Check whether `M` is a symplectic matrix, i.e. if ``M Ω Mᵀ = Ω`` where

```math
Ω = Iₙ ⊗  ⎛  0  1 ⎞
            ⎝ -1  0 ⎠
```
"""
function issymplectic(M)
    # We could check that
    #   size(M, 1) == size(M, 2) && iseven(size(M, 1))
    # but if it isn't so then the following operations fail immediately due to the
    # mismatching shapes of the matrices anyway.

    n = div(size(M, 1), 2)
    return M * Ω(n) * transpose(M) ≈ Ω(n)
end

"""
    williamson(M)

Compute the Williamson decomposition of M, which is assumed to be a `2n × 2n` real
positive-definite matrix.
Return a symplectic matrix V, i.e. such that ``V Ω Vᵀ = Ω`` where

```math
Ω = Iₙ ⊗  ⎛  0  1 ⎞
            ⎝ -1  0 ⎠
```

and a diagonal matrix D of positive numbers such that ``V D Vᵀ = M``.
"""
function williamson(M)
    @assert size(M, 1) == size(M, 2) && iseven(size(M, 1))
    @assert issymmetric(M)
    # We don't check that M > 0, it's probably expensive when M is large.
    n = div(size(M, 1), 2)

    # We use the (real) Schur decomposition on A = √M⁻¹ Ω √M⁻¹, which finds:
    #   - a matrix U that has 2x2 or 1x1 blocks on the diagonal, where 1x1 blocks contain
    #     the eigenvalues, while 2x2 blocks are of the form
    #       ⎛  Re(x)  Im(x) ⎞
    #       ⎝ -Im(x)  Re(x) ⎠
    #     where x is a complex number that would be an eigenvalue, together with its
    #     conjugate x̅, were A seen as a complex matrix,
    #   - an orthogonal matrix K
    # such that A = K U Kᵀ.
    # Actually since A is a skew-symmetric real matrix U will be of the form Δ ⊗ ω
    # The matrix Δ will turn out to be the inverse of D.
    sqrtM = sqrt(M)
    sqrtinvM = inv(sqrtM)
    A = sqrtinvM * Ω(n) * sqrtinvM
    U, K = schur(A)

    # diag(U, 1) gives us the elements [U[k, k+1] for k in 1:size(U, 1)], so here we get
    # the imaginary parts of the "complex eigenvalues" x from the 2x2 blocks mentioned above
    # (since they have real part = 0 and come in conjugate pairs)
    # We need these values to be positive, so we rearrange rows and columns to this purpose.
    perm = Vector{Int}(undef, 2n)
    for j in 1:n
        if U[2j - 1, 2j] ≥ 0
            perm[2j - 1], perm[2j] = 2j - 1, 2j
        else
            perm[2j - 1], perm[2j] = 2j, 2j - 1
        end
    end
    # NOTE: if p is a Vector{Int} obtained by shuffling 1:n and P = Matrix(Permutation(p)),
    # then for any n×n matrix A we have
    #   transpose(P) * A * P == A[p, p]
    #   A * P == A[:, p]
    #   transpose(P) * A == A[p, :]
    Up = U[perm, perm]
    d⁻¹ = diag(Up, 1)[1:2:(2n)]
    @assert all(d⁻¹ .> 0)
    D⁻¹ = Diagonal(reduce(vcat, [[x, x] for x in d⁻¹]))

    V = sqrtM * K[:, perm] * sqrt(D⁻¹)
    return inv(D⁻¹), V  # V * D * transpose(V) ≈ M
end

"""
    takagiautonne(A; svd_order=true)

Compute the Takagi-Autonne decomposition of the complex symmetric matrix `A`.
Return `D`, `U` such that `A = U D Uᵀ`.

Set `svd_order` to `true` (the default) to return the result by ordering the singular
values of `A` in descending order, `false` for ascending order.
"""
function takagiautonne end

function takagiautonne(A::AbstractMatrix{<:Real}; svd_order=true)
    # If the matrix A is real we can be more clever and use its eigendecomposition.
    # "Translated" from Python to Julia from The Walrus: see
    # https://the-walrus.readthedocs.io/en/latest/_modules/thewalrus/decompositions.html
    if size(A, 1) != size(A, 2)
        throw(ArgumentError("input matrix is not square"))
    end

    eigenvalues, U = eigen(A)
    d = abs.(eigenvalues)
    phases = Diagonal([v < 0 ? im : 1 for v in eigenvalues])
    Uc = U * phases
    p = sortperm(d; rev=svd_order)
    return Diagonal(d[p]), Uc[:, p]
end

function takagiautonne(A::Diagonal{T}; svd_order=true) where {T}
    phases = cis(Diagonal(angle.(diag(A))) / 2)
    # Since by construction phases is symmetric and
    #   A = phases^2 * D = phases * d * phases
    # we have, if P = Matrix(Permutation(p)), for any p,
    #   phases * P * transpose(P) * D * P * transpose(P) * transpose(phases) ≈ A
    # and since M * P = M[:, p] we get
    #   phases[:, p] * D[p, p] * transpose(phases[:, p]) ≈ A.
    # In particular, we choose p such that the diagonal on d is decreasing (increasing
    # if svd_order=false).
    d = abs.(diag(A))
    p = sortperm(d; rev=svd_order)
    return Diagonal(d[p]), phases[:, p]
end

function takagiautonne(A; svd_order=true)
    # "Translated" from Python to Julia from The Walrus: see
    # https://the-walrus.readthedocs.io/en/latest/_modules/thewalrus/decompositions.html
    if size(A, 1) != size(A, 2)
        throw(ArgumentError("input matrix is not square"))
    end

    # Try anyway to see if A is _close_ to a diagonal matrix, and use the dedicated
    # method if so (the ::Diagonal{T} method above works if A is _exactly_ diagonal, i.e.
    # if it is an instance of the Diagonal{T} type).
    if A ≈ Diagonal(A)
        return takagiautonne(Diagonal(A); svd_order=svd_order)
    end

    # Find whether the input is (approximately) a global phase times a real matrix.
    pos = argmax(abs.(A))  # Inspect the element with the largest absolute value
    θ = angle(A[pos])
    Ar = cis(-θ) * A
    if Ar ≈ real(Ar)
        D, U = takagiautonne(real(Ar); svd_order=svd_order)
        return D, U * cis(θ / 2)
    end

    # General method, for all other (symmetric) matrices.
    u, d, v = svd(A)  # Note that A = u * Diagonal(d) * v' (not v!)
    U = u * sqrt(u' * conj(v))
    if !svd_order
        return Diagonal(reverse(d)), U[:, end:-1:1]
    end
    return Diagonal(d), U
end
