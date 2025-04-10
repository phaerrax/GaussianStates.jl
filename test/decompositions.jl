function williamson_check(n)
    # Generate a random positive-definite matrix A
    A = randposdef(2n)

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
