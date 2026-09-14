# PyTorch Inference Profiling

An experimental study of CNN and Transformer inference costs, arithmetic intensity, KV-cache growth and measured compute/memory limits.

Maintained by [Ahmed Jawed](https://github.com/ahmedjawedaj).

## Methods and engineering work

- Analytical parameter, MAC/FLOP and activation-memory accounting.
- CNN and Transformer timing with warm-up and device synchronization.
- Separate prefill/decode and KV-cache experiments.
- Roofline matrix/convolution sweeps and runtime-specific CUDA/MPS comparisons.

## Repository contents

- [`inference_profiling.ipynb`](notebooks/inference_profiling.ipynb)
- [`data/README.md`](data/README.md): required external inputs and data handling.
- [`PROVENANCE.md`](PROVENANCE.md): original source-date evidence, attribution and publication changes.
- [`scripts/validate_sources.py`](scripts/validate_sources.py): notebook/Python syntax and exclusion checks.

## Recorded observations

- The source notebook records an RTX 3090 FP16 measured peak of 69.05 TFLOP/s and memory bandwidth of 798.52 GB/s.
- These imply a measured transition intensity of about 86.47 FLOPs/byte for that run.

## Setup and use

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd notebooks
jupyter lab
```

Select the intended kernel and run the notebook top to bottom after providing its required inputs. GPU notebooks can consume substantial runtime; validation below does not start training.

## Data and weights

Most profiling inputs are generated tensors. Selected architecture examples may download public pretrained weights. Match the target CUDA/MPS/CPU device before interpreting timing.

## Validation

```bash
python -m pip install -r requirements-validation.txt
python scripts/validate_sources.py
```

The publication checks validate notebook structure and compile source cells/Python modules. They do not reproduce training, GPU timing or ROS hardware behaviour. Cleaned notebooks have no captured outputs or stale execution counters.

## Scope and limitations

- Microbenchmarks describe the recorded hardware/workload, not production serving throughput.
- The source Colab metadata labels T4, while actual main runtime output identifies RTX 3090. The README follows runtime evidence.
- No custom CUDA-kernel authorship is claimed.

## References and attribution

- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)

See PROVENANCE.md for the distinction between original work dates and actual publication dates. This repository excludes instructor handouts, raw data, model weights and private records.
