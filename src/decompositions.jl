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
Return `D`, `U` such that `A = U D Uᵀ`, where `D` is a diagonal, positive-semidefinite
matrix and `U` is unitary.

Set `svd_order` to `true` (the default) to return the result by ordering the diagonal
values of `D` in descending order, `false` for ascending order.
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

"""
    euler(M)

Compute the Euler, or Bloch-Messiah, decomposition of the symplectic matrix `M`.
Return `L`, `D`, `R` such that `L * D * R = M`, where `L` and `R` are orthogonal symplectic
matrices with respect to the matrix

```math
Ω = Iₙ ⊗  ⎛  0  1 ⎞
            ⎝ -1  0 ⎠
```

and `D` is a diagonal matrix which can be written as

```math
⎛ d₁   0  ⎞ ⊕ ⎛ d₂   0  ⎞ ⊕ ... ⊕ ⎛ dₙ   0  ⎞
  ⎝ 0  1/d₁ ⎠   ⎝ 0  1/d₂ ⎠         ⎝ 0  1/dₙ ⎠
```

with ``dⱼ ≥ 1``.
"""
function euler(M)
    n = div(size(M, 1), 2)
    xxpp = [1:2:(2n); 2:2:(2n)]
    xpxp = invperm(xxpp)
    L, D, R = _euler_xxpp(M[xxpp, xxpp])
    @assert L * D * R ≈ M[xxpp, xxpp]

    return L[xpxp, xpxp], D[xpxp, xpxp], R[xpxp, xpxp]
end

function _euler_xxpp(M)
    # Here M is assumed to be symplectic with respect to the matrix
    #
    #   ⎛  0   Iₙ ⎞
    #   ⎝ -Iₙ  0  ⎠
    #
    # and the output will be symplectic according to this matrix, too.
    @assert size(M, 1) == size(M, 2) && iseven(size(M, 1))
    n = div(size(M, 1), 2)

    U, P = polar(M)  # U * P ≈ M with U unitary and P ≥ 0 (both of the same size as M)
    A =
        1 / 2 .* (
            P[1:n, 1:n] .- P[(n + 1):(2n), (n + 1):(2n)] .+
            im .* (P[1:n, (n + 1):(2n)] .+ P[(n + 1):(2n), 1:n])
        )
    Σ, W = takagiautonne(A)  # A ≈ W Σ Wᵀ

    Q = [real(W) -imag(W); imag(W) real(W)]

    Γ = Σ + sqrt(I + Σ^2)  # No need to broadcast here, everything is a Diagonal object
    D = Diagonal(M)  # We just need a 2n×2n Diagonal matrix, we'll overwrite the contents
    D[1:n, 1:n] .= Γ
    D[(n + 1):(2n), (n + 1):(2n)] .= Γ - 2Σ  # == inv(Γ)

    L = U * Q
    R = Q'
    return L, D, R
    # These matrices satisfy L D R = M, but they are symplectic with respect to
    #
    #   ⎛  0   Iₙ ⎞
    #   ⎝ -Iₙ  0  ⎠
    #
    # and not with the antisymmetric matrix Ω we use in this package. In other words, they
    # are in the xxpp representation, and we need the xpxp one.
    # If P is the xxpp → xpxp permutation matrix then
    #        M = L D R
    #   P M Pᵀ = P L Pᵀ P D Pᵀ P R Pᵀ
    #   ╰────╯   ╰────╯ ╰────╯ ╰────╯
    #     Mₚ   =   Lₚ     Dₚ     Rₚ
    # and the ₚ-matrices will be symplectic in the "correct" way.
end
