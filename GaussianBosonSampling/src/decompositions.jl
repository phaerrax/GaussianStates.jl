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
