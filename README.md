# GaussianStates.jl

[![Code Style:
Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://phaerrax.github.io/GaussianStates.jl/dev/)

A Julia package to manipulate Gaussian states.

The main block of this package is the `GaussianState` struct, which defines a
Gaussian state through its covariance matrix and its vector of first moments.
The package defines several Gaussian maps that can be applied to Gaussian
states, as well as relevant matrix decompositions.

This package mostly follows the notation in [[1]](#1), with first and second
moments $`R_j`$ and $`\sigma_{ij}`$ defined as

```math
R_i = \mathrm{tr}(\rho r_i)
```

```math
\sigma_{ij} = \mathrm{tr}(\rho \{r_i-R_i,r_j-R_j\})
            = \mathrm{tr}(\rho \{r_i,r_j\})-2R_iR_j,
```

where $`r = (x_1, p_1, x_2, p_2, \dotsc, x_n, p_n)`$.

## Installation

### From a registry

This package is registered in my
[TensorNetworkSimulations](https://github.com/phaerrax/TensorNetworkSimulations)
registry. By first adding this registry, with

```julia
using Pkg
pkg"registry add https://github.com/phaerrax/TensorNetworkSimulations.git"
```

(this must be done just once per Julia installation) the package can then be
installed as a normal one:

```julia
using Pkg
pkg"add GaussianStates"
```

### From GitHub

Alternatively, straight installation from GitHub is also possible:

```julia
using Pkg
pkg "add https://github.com/phaerrax/GaussianStates.jl"
```

## References

<a id="1">[1]</a>
Alessio Serafini, [‘Quantum Continuous Variables: A Primer of Theoretical
Methods’](https://doi.org/10.1201/9781315118727) (2nd ed.), 2017, CRC Press.
