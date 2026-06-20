> [!Note]
> termination criteria by monitoring the relative changes of loss and PDE residual.

🔵 To accelerate training with the L-BFGS optimizer, termination criteria are applied every 500 epochs by monitoring the relative differences in either the loss values ($\epsilon_J =$ $|\text{Loss}_{k+500} -$ $\test{Loss}_{k}|$ $/\test{Loss}_{k}$) or the PDE residuals ($\epsilon_f =$ $|\test{Residual}_{k+500} -$ $ \test{Residual}_{k}|$ $/\test{Residual}_{k}$).
