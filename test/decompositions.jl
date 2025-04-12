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

function euler_check(n)
    s = randsymplectic(n)
    l, d, r = euler(s)

    if !(issymplectic(l) && l * l' ≈ I && l' * l ≈ I)
        println("L matrix not symplectic or orthogonal")
        return false
    end
    if !(issymplectic(r) && r * r' ≈ I && r' * r ≈ I)
        println("R matrix not symplectic or orthogonal")
        return false
    end

    return l * d * r ≈ s
end
