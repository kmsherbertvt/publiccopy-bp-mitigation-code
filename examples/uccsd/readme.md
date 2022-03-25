# H_6 with Fermionic Pool Example

The main script in this example is `test_uccsd.jl`. It reads in the Hamiltonian from `h6_1A.ham`, which is taken from openfermion output. Additional information for comparison is available in `h6_1A.txt`. The resulting data is placed in `output.txt`.

The operators added at each layer are printed out in the output in the form `(a,b,i,j)` in the case of a double or `(a,i)` in the case of a single.

```
julia --threads=auto --project=@. test_uccsd.jl > output.txt
```