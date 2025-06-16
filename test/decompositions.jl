function williamson_check(A)
    # Check that actually A > 0
    if !isposdef(A)
        return false
    end

    # Compute the Williamson decomposition A → S D Sᵀ
    D, S = williamson(A)

    # Check that S is symplectic
    if !issymplectic(S)
        println("S not symplectic")
        return false
    end

    # Check that the symplectic eigenvalues are positive
    if any(diag(D) .≤ 0)
        println("negative diagonal values")
        return false
    end

    # Check that actually A = S D Sᵀ
    return A ≈ S * D * transpose(S)
end

function takagiautonne_check(A)
    # Compute the Takagi-Autonne decomposition A → S D Sᵀ
    D, W = takagiautonne(A)

    # Check that W is unitary
    if !(W * W' ≈ W' * W ≈ I(size(W, 1)))
        println("W is not unitary")
        return false
    end

    # Check that the diagonal values are positive
    if any(diag(D) .≤ 0)
        println("negative diagonal values")
        return false
    end

    # Check that actually A = W D Wᵀ
    return A ≈ W * D * transpose(W)
end

function euler_check(M)
    L, D, R = euler(M)

    if !isapprox(L * L', I)
        println("left matrix is not orthogonal")
        return false
    end
    if !isapprox(R * R', I)
        println("right matrix is not orthogonal")
        return false
    end

    if !issymplectic(L)
        println("left matrix is not symplectic")
        return false
    end
    if !issymplectic(R)
        println("right matrix is not symplectic")
        return false
    end

    if any(≤(0), diag(D))
        println("not all singular values are positive")
        return false
    end

    if !isapprox(D[1:2:end, 1:2:end] * D[2:2:end, 2:2:end], I)
        println("singular values matrix does not have the correct structure D ⊕ D⁻¹")
        return false
    end

    return L * D * R ≈ M
end
