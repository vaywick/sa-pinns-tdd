# SA-PINNs-TDD

Self-adaptive physics-informed neural networks (SA-PINNs) [Journal of Computational Physics, 474:111722, 2023] have demonstrated superior accuracy over vanilla PINNs in solving partial differential equations with steep gradients by assigning higher weights to critical regions.
However, as the spatio-temporal domain extends, SA-PINNs often struggle to maintain high prediction accuracy.
To address this limitation, we propose SA-PINNs with time-domain decomposition (SA-PINNs-TDD), which enable accurate and robust training across extended temporal domains, as demonstrated on the nonlinear Schr\"{o}dinger equation, the 2D Navier-Stokes equation, and the 2D Burgers' equation.
For the Allen-Cahn equation, SA-PINNs-TDD achieves a 56-fold improvement in accuracy over standard SA-PINNs, owing to a more effective distribution of adaptive weights.
Furthermore, nonuniform time-domain partitioning is employed for the 2D Burgers’ equation to further enhance the prediction accuracy.
SA-PINNs-TDD adopt a sequential domain-wise training strategy, where thorough pre-training of the initial subdomain is essential to establish accurate pseudo initial conditions and reduce error accumulation in subsequent subdomains.
While the initial subdomain requires more epochs to reach convergence, subsequent subdomains benefit from transferred information, resulting in accelerated optimization.


#### Each folder in our repository is detailed in its respective readme.md file.

# Benchmark

### Allen-Cahn (SA-PINNs-TDD using 2 partition training)：
Prediction accuracy across different models is assessed using the relative $L_2$-norm (RL2) error.

[ PT-PINN: Journal of Computational Physics, 489:112258, 2023. bc-PINN: Computer Methods in Applied Mechanics and Engineering, 390:114474, 2022. ]

| Model             | RL2                  |
|-------------------|----------------------|
| bc-PINN       | 1.68e-2              |
| original SA-PINNs | (2.10±1.21)e-2       |
| PT-PINN      | (9.7±0.4)e-3         |
| SA-PINNs-TDD      | **(3.76±1.01)e-4**   |


### First-order rogue wave of NLS equation (subdomain number test)：
| Total number | RL2                |
|--------------|--------------------|
| 1            | (1.11±0.05)e-1     |
| 3            | (4.09±1.87)e-4     |
| 5            | **(3.02±0.92)e-4** |
| 7            | (2.72±0.74)e-4     |

#### Comparisons：
| Model | vanilla PINNs     | original SA-PINNs | SA-PINNs-TDD      |
|-------|-------------------|-------------------|-------------------|
| RL2   | (1.52±0.04)e-1    | (1.11±0.05)e-1    | **(3.02±0.92)e-4** |

### Second-order rogue wave of NLS equation (subdomain number test)：
| Total number | RL2                |
|--------------|--------------------|
| 1            | (1.60±0.05)e-1     |
| 5            | (9.21±7.12)e-4     |
| 7            | **(6.05±1.99)e-4** |


# Framework

SA-PINNs-TDD are implemented using either the TensorFlow or PyTorch frameworks.
The Euler–Bernoulli-new and Allen–Cahn (AC) simulations are conducted in TensorFlow, while the remaining models are implemented in PyTorch.

PyTorch version 2.0.0 or higher

tensorflow version = 2.3.0 and keras version = 2.4.3

If you have any questions about the code, please submit them via GitHub Issues.
