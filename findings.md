# Findings

## V5.0 Mamba2 is the correct baseline
- V5.0 (Mamba2): Best Dice 0.8479 (text+TTA+PP), checkpoint best_dice=0.8769
- V4.6 (Mamba1): ~0.88
- V5.1 (Mamba3): 0.7555 — full failure, Mamba3 unstable on CUDA 13.0
- V5.2 should fine-tune from V5.0 with use_mamba3=true (which uses Mamba2 via alias)

## use_mamba3 naming confusion
- Flag `use_mamba3` actually enables Mamba2 (via `from mamba_ssm import Mamba2 as Mamba3`)
- Full rename too risky (breaks configs, checkpoints, tests, notebooks)
- Solution: keep flag name, add runtime log showing actual backend

## FocalTverskyLoss math bug
- When class absent in batch, weight=0 excluded from weight_sum
- Causes dynamic re-normalization: remaining class weights inflate
- ET (class 3) most affected — rarest class, absent in many batches
