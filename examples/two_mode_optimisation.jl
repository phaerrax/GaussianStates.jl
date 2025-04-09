using GaussianBosonSampling, LinearAlgebra, SCS, JuMP

let n = 2
    g0 = vacuumstate(n)

    # Generate random squeezing parameters within the unit circle
    g = squeeze(g0, [1.2, 0.2])
    g = lossybeamsplitter(g, 0.5, 0.7, 1, 2)
    println("Average photon number in non-optimised state: ", number(g))

    # Configure optimisation model
    model = Model(SCS.Optimizer)
    @variable(model, x[1:(2n), 1:(2n)] in PSDCone())
    @objective(model, Min, tr(x))  # minimise photon number
    @constraint(model, g.covariance_matrix ≥ x, PSDCone())
    @constraint(
        model,
        kron(I(2), x) + kron([[0, 1] [-1, 0]], 0.5 * GaussianBosonSampling.Ω(n)) ≥ 0,
        PSDCone()
    )  # uncertainty relations
    JuMP.optimize!(model)

    sol = JuMP.value(x)
    # Put the solution into a new Gaussian state and show the new photon number
    opt_g = GaussianState(g.first_moments, sol)
    println("Average photon number in optimised state: ", number(opt_g))

    ev, _ = eigen(opt_g.covariance_matrix)
    println("Eigenvalues of the optimised covariance matrix σₒₚₜ:\n", join(ev, "\n"))

    ev, _ = eigen(opt_g.covariance_matrix + 0.5im * GaussianBosonSampling.Ω(n))
    println("Eigenvalues of σₒₚₜ + i/2 Ω:\n", join(ev, "\n"))
end
