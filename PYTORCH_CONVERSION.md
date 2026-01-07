# PyTorch Conversion Documentation

This document describes the conversion of DimeNet from TensorFlow to PyTorch.

## Overview

This repository has been successfully converted from TensorFlow to PyTorch while maintaining:
- **Exact file structure** - All files remain in the same locations
- **Logic consistency** - Architectural logic and forward pass behavior preserved
- **API compatibility** - Model interfaces remain similar

## Changes Made

### 1. Dependencies (setup.py)
- Removed: `tensorflow>=2.1`, `tensorflow_addons`
- Added: `torch>=1.9.0`, `torch_geometric>=2.0.0`

### 2. Core Utilities

#### activations.py
- `tf.sigmoid` → `torch.sigmoid`
- `tf.nn.softplus` → `F.softplus`
- `tf.log` → `torch.log`

#### initializers.py
- `tf.initializers.Initializer` → Python class
- `tf.initializers.Orthogonal` → `nn.init.orthogonal_`
- `tf.sqrt`, `tf.math.reduce_variance` → `torch.sqrt`, `torch.var`

### 3. Layer Components

#### Common Patterns
- `tf.keras.layers.Layer` → `nn.Module`
- `call()` method → `forward()` method
- `self.add_weight()` → `nn.Parameter()`
- `layers.Dense()` → `nn.Linear()`
- `layers.Layer` lists → `nn.ModuleList()`

#### Specific Conversions

**envelope.py**
- `tf.where` → `torch.where`
- `tf.zeros_like` → `torch.zeros_like`

**bessel_basis_layer.py**
- `tf.constant` → `torch.tensor`
- `tf.expand_dims` → `tensor.unsqueeze`
- `tf.sin` → `torch.sin`

**spherical_basis_layer.py**
- `tf.stack` → `torch.stack`
- `tf.gather` → tensor indexing
- `tf.repeat` → `tensor.repeat_interleave`
- TensorFlow backend in sympy lambdify → NumPy backend (with torch conversion)

**embedding_block.py**
- `tf.gather` → tensor indexing
- `tf.concat` → `torch.cat`
- Added `num_radial` parameter for proper input dimensions

**interaction_block.py & interaction_pp_block.py**
- `tf.einsum` → `torch.einsum`
- `tf.math.unsorted_segment_sum` → `torch.zeros().index_add_()`
- Added `num_radial` and `num_spherical` parameters

**output_block.py & output_pp_block.py**
- `tf.math.unsorted_segment_sum` → `torch.zeros().index_add_()`
- Added `num_radial` parameter

**residual_layer.py**
- Separated activation from Dense layers (PyTorch doesn't support activation in Linear)

### 4. Main Models

#### dimenet.py & dimenet_pp.py
- `tf.keras.Model` → `nn.Module`
- `call()` → `forward()`
- `tf.gather` → tensor indexing
- `tf.shape` → `tensor.shape`
- `tf.reduce_sum` → `torch.sum`
- `tf.norm` → `torch.norm`
- `tf.linalg.cross` → `torch.linalg.cross`
- `tf.math.atan2` → `torch.atan2`
- `tf.math.segment_sum` → `torch.zeros().index_add_()`
- `tf.math.segment_mean` → custom implementation with counts

### 5. Training Components

#### trainer.py
- `tf.optimizers.Adam` → `torch.optim.Adam`
- `tf.GradientTape` → loss.backward()
- `tf.clip_by_global_norm` → `torch.nn.utils.clip_grad_norm_`
- `tfa.optimizers.MovingAverage` → manual EMA implementation
- Removed `@tf.function` decorators

#### metrics.py
- `tf.keras.metrics.Mean` → manual accumulation with counters
- Removed TensorFlow summary writing

#### schedules.py
- `tf.optimizers.schedules.LearningRateSchedule` → Python class
- Simplified polynomial/exponential decay implementation

#### data_provider.py
- `tf.data.Dataset` → removed (use get_batch method instead)
- `tf.constant` → `torch.tensor`
- Device parameter added for GPU support

## Important Notes

### Input Dimension Handling
Unlike TensorFlow's automatic shape inference, PyTorch requires explicit input dimensions:
- `EmbeddingBlock` needs `num_radial` parameter
- `InteractionBlock` needs `num_radial` and `num_spherical`
- `OutputBlock` needs `num_radial` parameter

### Fixed Issues
1. **basis_utils.py**: Changed `np.math.factorial` to `math.factorial` (numpy deprecated np.math)
2. **All layers**: Added explicit input dimension parameters
3. **initializers.py**: Fixed tensor copying warning by using detach().clone()

### Performance Considerations
Current implementation uses manual scatter operations (`torch.zeros().index_add_()`). For better performance, consider using `torch_scatter` library functions in future optimizations.

### Known Differences
1. Random initialization may differ slightly between TensorFlow and PyTorch
2. Numerical precision differences due to different implementations
3. Spherical basis functions use CPU-numpy conversions (could be optimized)

## Testing

Both models have been tested successfully:
- ✅ Model instantiation
- ✅ Forward pass
- ✅ Parameter counting
- ✅ Example usage script

## Usage

See `example_usage.py` for demonstration of how to use the models.

```python
from dimenet.model.dimenet_pp import DimeNetPP
from dimenet.model.activations import swish

model = DimeNetPP(
    emb_size=128,
    out_emb_size=256,
    int_emb_size=64,
    basis_emb_size=8,
    num_blocks=4,
    num_spherical=7,
    num_radial=6,
    cutoff=5.0,
    activation=swish,
    num_targets=1
)
```

## Migration Guide

To migrate existing TensorFlow code:
1. Update dependencies in requirements/environment
2. Change imports from `tensorflow` to `torch`
3. Update model instantiation (same parameters)
4. Prepare input dictionaries with tensors instead of tf.constants
5. Use model.forward() or model() for predictions
6. Update training loop to use PyTorch optimizer and loss.backward()

## Files Modified

- `setup.py` - Dependencies
- `.gitignore` - Added PyTorch artifacts
- `README.md` - Updated documentation
- `dimenet/model/activations.py`
- `dimenet/model/initializers.py`
- `dimenet/model/dimenet.py`
- `dimenet/model/dimenet_pp.py`
- `dimenet/model/layers/*.py` - All 10 layer files
- `dimenet/training/*.py` - All 5 training files
- `example_usage.py` - New example script

## Files Unchanged

- `dimenet/model/layers/basis_utils.py` - Only fixed np.math.factorial
- `dimenet/training/data_container.py` - Pure NumPy, no changes needed
- All notebooks (*.ipynb) - Not converted, need separate migration
- Config files (*.yaml)
- Data files
