> [!Note]
> termination criteria by monitoring the relative changes of loss and PDE residual.

🔵 To accelerate training with the L-BFGS optimizer, termination criteria are applied every 500 epochs by monitoring the relative differences in either the loss values ($\epsilon_J =$ $|\rm{Loss}_{k+500} -$ $\rm{Loss}_{k}|$ $/\rm{Loss}_{k}$) or the PDE residuals ($\epsilon_f =$ $|\rm{Residual}_{k+500} -$ $ \rm{Residual}_{k}|$ $/\rm{Residual}_{k}$).
