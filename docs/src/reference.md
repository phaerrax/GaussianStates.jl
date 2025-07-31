# Reference

## Gaussian states and operations

### State definitions

With the following methods you can create some basic Gaussian states.

```@docs
vacuumstate
thermalstate
randgaussianstate
```

### Gaussian operations

The following methods can be applied to a `GaussianState` in order to simulate
quantum optical operations.

```@docs
displace
displace!
squeeze
squeeze!
squeeze2
squeeze2!
beamsplitter
beamsplitter!
lossybeamsplitter
lossybeamsplitter!
```

## Matrix decompositions

```@docs
williamson
takagiautonne
euler
```

## Utilities

```@docs
nmodes
randposdef
randsymplectic
number
purity
is_valid_covariance_matrix
issymplectic
```
