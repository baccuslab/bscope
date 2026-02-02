# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

bscope is a PyTorch library for neural network interpretation through gradient-based attribution methods. It computes layer-wise neuron contributions to model predictions using techniques like integrated gradients, SmoothGrad, and activation-gradient products. It has special support for Vision Transformers (via `timm`).

## Setup & Commands

```bash
pip install -e .          # Install in development mode
python quickstart.py      # Run the working example/smoke test
```

There is no formal test suite, linter, or build step. The quickstart.py script serves as the primary integration check.

## Dependencies

Core: torch, torchvision, numpy, scipy, timm, matplotlib, tqdm, scikit-learn. No requirements.txt exists; dependencies are implicit from imports.

## Architecture

### Core Pipeline

The central workflow is: **Scope** hooks into a model's layers via **Inspector**, runs forward/backward passes with a chosen attribution method and target, and produces per-layer contributions (activation x gradient products).

**Scope** (`bscope/scope.py`) — Main entry point and orchestrator. Configurable via method chaining:
- Attribution methods: `use_int_grad()`, `use_smooth_grad()`, `use_act_grad()`, `use_act_normgrad()`, `use_normact_normgrad()`, `use_jacobians()`
- Attribution targets: `wrt_topk()`, `wrt_output_neuron()`, `wrt_entropy()`, `wrt_sum()`, `wrt_surprisal()`, `wrt_contrastive_top2()`
- Logging system (`log_start()`/`log_stop()`) aggregates results across multiple forward passes with optional reductions (spatial_sum, ei_split, patch_sum, patch_ei_split, attn_head variants)
- After calling `scope(input)`, results are in `scope.contributions`, `scope.activations`, `scope.gradients`

**Inspector** (`bscope/inspector.py`) — Registers PyTorch forward/backward hooks on specified layers to capture activations and gradients during the forward and backward passes.

**Disruptor** (`bscope/disruptor.py`) — Causal perturbation analysis. Modifies layer activations mid-forward-pass for ablation studies. Styles: destroy, corrupt, modal_offset, modal_corruption, mode_weighted. Has a specialized `AttentionDisruptor` for ViT attention heads (outgoing, incoming, bidirectional).

**Jacobian** (`bscope/jacobian.py`) — Direct derivative computation between layers using finite differences or `torch.autograd.functional.jacobian`.

### Supporting Modules

**SAE** (`bscope/sae.py`) — Sparse autoencoder models for dictionary learning on activations. Multiple architectures: NNSTSAE, STSAE, SigThreshSAE, SSSAE with different thresholding strategies. Includes `load_sae()` for loading trained models.

**Metrics** (`bscope/metrics.py`) — Loss functions, norms (l0, l1, l2, lp), and reconstruction quality measures for dictionary learning evaluation.

**Utils** (`bscope/utils.py`) — Statistical analysis tools: significance selection (threshold, percentile, top_n, kmeans, otsu, std), excitatory/inhibitory splitting (`ei_split`), participation ratio, matrix correlation, NMSE, cross-entropy degradation, AUC, config parsing.

### Image Classification Subpackage (`bscope/ic/`)

Specialized tools for vision model analysis:
- `models.py` — Load pretrained models (ResNet variants, AlexNet, MobileNet, ViT) with consistent interface
- `evaluation.py` — Top-1/Top-5 accuracy, per-class accuracy evaluation
- `custom_dataset.py` — ImageNet dataset loading and preprocessing
- `mode_summary.py` — Contribution mode analysis and visualization (largest module)
- `semantic_utils.py` — Semantic similarity and analysis tools
- `visualization.py` — Attribution visualization utilities

### Public API

All modules export via `from .module import *` in `bscope/__init__.py`. The primary user-facing classes are `Scope`, `Disruptor`, `AttentionDisruptor`, and the SAE model classes.
