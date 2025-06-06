# GaussianStates.jl

A Julia package to manipulate Gaussian states.

The main block of this package is the `GaussianState` struct, which defines a
Gaussian state through its covariance matrix and its vector of first moments.
The package defines several Gaussian maps that can be applied to Gaussian
states, as well as relevant matrix decompositions.

This package mostly follows the notation in [[1]](#1), with first and second
moments $`R_j`$ and $`\sigma_{ij}`$ defined as

```math
R_j = \operatorname{tr}(\rho r_i)
```

```math
\sigma_{ij} = \operatorname{tr}(\rho \{r_i-R_i,r_j-R_j\})
            = \operatorname{tr}(\rho \{r_i,r_j\})-2R_iR_j,
```

where $`r = (x_1, p_1, x_2, p_2, \dotsc, x_n, p_n)`$.

## References

<a id="1">[1]</a>
Alessio Serafini, [‘Quantum Continuous Variables: A Primer of Theoretical
Methods’](https://doi.org/10.1201/9781315118727) (2nd ed.), 2017, CRC Press.
