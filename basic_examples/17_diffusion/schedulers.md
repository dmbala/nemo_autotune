# Diffusion / flow-matching samplers — cheat sheet

The number of `--num-inference-steps` you need depends on which sampler the pipeline uses. Different pipelines default to different samplers; most allow swapping via `pipe.scheduler = <scheduler-class>.from_config(pipe.scheduler.config)`.

## The families

**Denoising diffusion (DDPM / DDIM).** Original formulation: learn a noise predictor ε_θ(x_t, t). At inference, invert the forward noising process step-by-step. DDPM is stochastic, DDIM is deterministic (essentially the same model, a different solver). Typical step counts: 20–50 (DDIM), 1000 (DDPM).

**Higher-order ODE solvers (DPM-Solver, Euler-Ancestral, Heun).** Treat denoising as integrating an ODE; use midpoint / RK methods. Much better quality per step than DDIM — 10–20 steps produce near-converged output on most pipelines. Default for most modern image models.

**Rectified flow / flow matching (Wan2.2, Flux, SD3).** Reframes diffusion as learning a vector field that directly maps noise to data along straight paths (rectified flow). Training objective is a flow-matching loss. Inference is an Euler integration of the field. Needs as few as **4–12 steps** at acceptable quality. This is what Wan2.2 uses — the scheduler is `FlowMatchEulerDiscreteScheduler` under the hood.

## Rough step-count guide

| Sampler | Typical steps | Where used |
|---|---|---|
| DDPM | 1000 | training, almost never at inference |
| DDIM | 50 | older SD 1.5 / 2.x demos |
| Euler / Euler-Ancestral | 20–40 | most HF diffusers defaults |
| DPM-Solver-2M | 15–25 | SD2.x, SDXL |
| Heun | 20–40 | SDXL-turbo-lineage |
| Rectified-flow / FlowMatch-Euler | **4–12** | **Wan2.2 (our default), Flux, SD3** |

If you want faster iteration for the workshop demo, drop `--num-inference-steps` to 10 on Wan2.2 — it'll still produce a recognizable clip.

## Swapping schedulers (manual, not in our generate.py)

```python
from diffusers import DPMSolverMultistepScheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
```

Most pipelines will accept an Euler scheduler too. The `NeMoAutoDiffusionPipeline` inherits `pipe.scheduler` from the underlying `diffusers.DiffusionPipeline`, so this works unchanged after loading.

## Flow matching in training (not covered here)

Automodel only ships the *inference* side of diffusion / flow matching. To *train* a flow-matching model, you need:

- A u-net / DiT with a time-conditioning head (easy).
- A noise-to-data pair sampler (`x = (1-t) * noise + t * data`).
- A flow-matching loss: `MSE( v_θ(x, t), data - noise )`.
- Optionally a CFM variant: optimal-transport pairing, minibatch rectified-flow, Mean Flow.

HF's `diffusers` has a `train_text_to_image_flow_matching.py` example; `torchcfm` is a research library for FM losses. Neither integrates with Automodel's recipe — if you want an Automodel-native FM training path, that's a custom recipe (similar scope to `train_ft.py`). Out of scope for this workshop.

## Related

- [Lipman et al. 2022, "Flow Matching for Generative Modeling"](https://arxiv.org/abs/2210.02747)
- [Liu et al. 2022, "Rectified Flow"](https://arxiv.org/abs/2209.03003)
- [diffusers schedulers](https://huggingface.co/docs/diffusers/api/schedulers/overview)
- [torchcfm](https://github.com/atong01/conditional-flow-matching) — FM training utilities
