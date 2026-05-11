# hhmodle

Minimal Hodgkin-Huxley neuron simulator implementing the standard 4-ODE giant squid axon model:

- Membrane equation: `C dv/dt = Iext - (INa + IK + IL)`
- Gating equations: `dx/dt = alpha_x(v)(1-x) - beta_x(v)x` for `x in {m, h, n}`
- Standard constants: `C=1.0`, `gNa=120`, `gK=36`, `gL=0.3`, `ENa=50`, `EK=-77`, `EL=-54.4`
- Integration methods: Forward Euler and RK4 (`RK4` is the default)

Run tests:

```bash
python -m unittest discover -s tests -v
```
