# GaussianStates.jl

*This is the documentation for the GaussianStates.jl package.*

The package defines several Gaussian maps that can be applied to Gaussian
states, as well as relevant matrix decompositions.

This package mostly follows the notation in [Serafini2023:qcv_book](@cite), with
first and second moments ``R_i`` and ``\sigma_{ij}`` defined as

```math
\begin{align}
R_i &= \operatorname{tr}(\rho r_i)\\
\sigma_{ij} &= \operatorname{tr}(\rho \{r_i-R_i,r_j-R_j\}) =\\
            &= \operatorname{tr}(\rho \{r_i,r_j\})-2R_iR_j,
\end{align}
```

where ``r = (x_1, p_1, x_2, p_2, \dotsc, x_n, p_n)``.

## Package features

- Creation of vacuum, thermal, and random Gaussian states.
- Most of the Gaussian operations listed in
  [Brask2022:Gaussian_quickref](@cite).
- Williamson, Takagi-Autonne, and Euler/Bloch-Messiah matrix decompositions.

## Bibliography

```@bibliography
```
