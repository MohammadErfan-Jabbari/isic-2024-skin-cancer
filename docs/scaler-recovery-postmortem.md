# Scaler Recovery Postmortem — GradScaler Saved as StandardScaler

> A PyTorch AMP `GradScaler` was pickled under the variable name intended for an sklearn
> `StandardScaler`. The wrong object was silently loaded at inference, corrupting metadata
> normalization and dropping one fold's test AUC from ~95% to ~85%.

---

## The Incident

During `dual_hybrid_v1` training (`18_1_train_dual_backbone_hybrid.py`), the code saved per-fold preprocessing artifacts including the metadata `StandardScaler`. Due to a variable aliasing bug — same name, wrong object — the file `scaler_fold{N}.pkl` for each fold actually contained a PyTorch AMP `GradScaler` instance.

**Symptom:** Fold 1 validation AUC ~95% (correct); fold 1 test AUC ~85% (degraded). The metadata normalization step at inference was applying `GradScaler.transform()` semantics — or failing silently — instead of the correct `StandardScaler.transform()`. Other folds showed similar degradation.

The bug was not caught during training because:
1. Training only validates on the OOF split using the correct in-memory scaler object.
2. The serialized file is only loaded at inference time.
3. No type check was performed on load.

---

## Root Cause

```python
# Simplified illustration of the bug:
scaler = StandardScaler()
scaler.fit(train_metadata)
# ... later in the AMP training loop ...
grad_scaler = GradScaler()   # variable reuse or shadowing
# ... at checkpoint save ...
pickle.dump(scaler, f)       # 'scaler' name now points to GradScaler
```

Both objects exist in the same scope; `scaler` was reassigned or shadowed in the AMP setup block. The correct `StandardScaler` (fitted on ~313K–327K training samples, 38 numerical features) was lost.

---

## The Fix

**`tools/recover_scalers.py`** (source: `recover_scalers.py` in the project root):

1. Reconstructs the exact 5-fold `StratifiedGroupKFold` splits using the same random seed and parameters as the original training run.
2. For each fold, identifies the training indices (the same ~313K–327K samples the original training saw).
3. Fits a fresh `StandardScaler` on those training samples across the 38 numerical features from `feature_list.txt`.
4. Overwrites `scaler_fold{N}.pkl` with the recovered `StandardScaler`.

This is exact: same seed → same splits → same training set → same fitted statistics. The recovered scaler is equivalent to what the training run should have saved.

**`tools/verify_scaler.py`** (source: `verify_scaler.py` in the project root):

Loads each fold's scaler pickle and asserts `isinstance(scaler, StandardScaler)`. Raises on failure. Run this after any training run before proceeding to inference.

---

## Prevention

The training script now includes a guard before every scaler save:

```python
if not hasattr(scaler, 'transform') or not hasattr(scaler, 'mean_'):
    raise ValueError(f"Expected StandardScaler, got {type(scaler)}")
pickle.dump(scaler, f)
```

`GradScaler` does not have a `mean_` attribute; this check catches the wrong type before it reaches disk.

---

## Key Numbers

- Affected: `dual_hybrid_v1`, all 5 folds.
- Impact: fold 1 test AUC 95% → 85%; other folds similarly degraded.
- Recovery: `dual_hybrid_v2` and all current inference use the recovered scalers.
- Training samples per fold scaler: ~313,000–327,000 rows, 38 numerical features.
