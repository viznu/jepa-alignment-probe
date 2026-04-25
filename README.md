# jepa-alignment-probe

**Status: parked.** This repo was the early-phase scratch space for a project
that aimed to study JEPA-style self-supervised representations of transformer
internal activations as an alignment-monitoring tool.

## Where the work went

The contrastive (non-JEPA) line that emerged from this work moved into a
separate, cleaned-up repository:

- **[viznu/contrastive-activation-trajectories](https://github.com/viznu/contrastive-activation-trajectories)** —
  pair-aware contrastive probing of transformer activation trajectories on
  Instructed-Pairs. It also contains the negative-control finding that, on
  that benchmark, PCA without any training matches the contrastive method,
  i.e. the contrastive objective is not load-bearing on this dataset.

## What lived here

This repo went through several iterations exploring:

1. JEPA-style masked-layer prediction encoders over transformer activation
   trajectories. The masked-reconstruction objective alone learned the
   target-model's per-layer structure but failed to recover behavioral
   alignment perturbations beyond what direct activation probes already
   capture.
2. **JEPA-SCORE** (Balestriero 2025) as a Jacobian-density anomaly detector
   on activation trajectories. Tested across mean-pool, last-slot, layer-24,
   window, and pair-residualized variants. AUROC was ~0.50 across all
   variants on the Instructed-Pairs benchmark. The finding was empirically
   confirmed via a mirror-invariance check on the encoder's Jacobian
   singular values.
3. Pair-aware contrastive add-on (cosine and InfoNCE) which initially
   appeared to recover an alignment-relevant behavioral latent. Subsequent
   stress-testing in the new repo showed PCA with no training matches that
   result on the same benchmark.

The full experimental record — extraction scripts, JEPA training, JEPA-SCORE
implementation, mirror-invariance lemma check, InfoNCE add-on, transfer
tests — is in this repo's git history (e.g. tag-able commit
`812c65c`).

## What might come back here

Two ice-boxed directions:

1. **Action-conditioned latent world model of an LLM's internals.** Train
   a model that predicts the effect of an intervention (activation addition,
   refusal-direction steering, prompt edit) on the LLM's next-step
   activations, then evaluate whether the predicted effect matches the
   actual patched run.
2. **JEPA-style text representation models for alignment.** Survey and
   reproduce the small body of work on JEPAs in language modeling
   (e.g. JEPA-Reasoner) and test whether their latents are easier to align
   or monitor than equivalent autoregressive LLMs.

Neither is currently active.
