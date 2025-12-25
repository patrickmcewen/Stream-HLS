# Bit Transformer Stream-HLS Fixes

This document records all the fixes applied to make `bit_transformer` work with Stream-HLS compilation pipeline.

## Overview

The `bit_transformer` model required several modifications to be compatible with Stream-HLS's MLIR compilation pipeline. The main issues were:
1. Missing external dependencies
2. MLIR-incompatible operations (pow, embedding lookups)
3. Input tensor type/shape mismatches

---

## Fix 1: Missing `zeta` Module Dependency

**Issue**: 
```
ModuleNotFoundError: No module named 'zeta'
```

**Location**: `bit_transformer.py` lines 9-11

**Problem**: The code imported `OutputHead` and `SimpleRMSNorm` from the `zeta` package, which is not installed and not available via pip.

**Fix**: Created local implementations of both classes:

```python
# Replaced:
from zeta import OutputHead
from zeta.nn.modules.simple_rmsnorm import SimpleRMSNorm

# With local implementations:
class SimpleRMSNorm(nn.Module):
    """Simple RMS Normalization module."""
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** -0.5
    
    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
        return x / rms * self.scale

class OutputHead(nn.Module):
    """Simple output head for language modeling."""
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, vocab_size)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.linear(x)
```

**Files Modified**: `bit_transformer.py` (lines 10-42)

---

## Fix 2: MLIR-Incompatible `pow` Operation

**Issue**:
```
error: 'arith.fptosi' op result #0 must be signless-fixed-width-integer-like, but got 'si64'
error: "__module.transformer/__module.transformer.layers.0/aten::pow"
```

**Location**: `bit_transformer.py` line 304 (in `scaled_dot_product_gqa` function)

**Problem**: Using `query.size(-1) ** 0.5` creates a tensor `pow` operation during tracing, which MLIR cannot properly lower. The operation tries to convert float64 to signed int64, which is unsupported.

**Fix**: Replaced tensor `pow` operation with `math.sqrt()` using Python float:

```python
# Before:
if scale is None:
    scale = query.size(-1) ** 0.5

# After:
if scale is None:
    # Use math.sqrt with the already-extracted dimension to avoid tensor pow operations
    # that MLIR can't handle. dq is the head dimension from query.shape unpacking above.
    scale = math.sqrt(float(dq))
```

**Files Modified**: 
- `bit_transformer.py` (line 6: added `import math`)
- `bit_transformer.py` (lines 304-307: replaced pow with math.sqrt)

**Note**: Also needed to extract `dq` from shape unpacking to ensure it's a Python value, not a tensor.

---

## Fix 3: Unsupported `nn.Embedding` Operation

**Issue**:
```
error: 'linalg.generic' op is unsupported operation.
error: unsupported tensor element type.
    %0 = tensor.empty() : tensor<1x512x128xf32>
```

**Location**: `bit_transformer.py` line 668 (original `nn.Embedding`)

**Problem**: `nn.Embedding` compiles to a `linalg.generic` operation with complex indexing maps that Stream-HLS doesn't support. The embedding lookup pattern is not recognized by the Stream-HLS compiler.

**Initial Attempt 1**: Tried replacing with one-hot encoding + matrix multiplication:
```python
x_one_hot = F.one_hot(x, num_classes=self.num_tokens).float()
x = torch.matmul(x_one_hot, self.emb_weight)
```
**Result**: This also failed because `F.one_hot` creates unsupported `linalg.generic` operations and i64 tensors.

**Initial Attempt 2**: Tried `torch.index_select` with flattening:
```python
x_flat = x.view(-1)  # (batch * seq_len,)
x = torch.index_select(self.emb_weight, dim=0, index=x_flat)
x = x.view(batch_size, seq_len, self.emb_dim)
```
**Result**: Failed because `x.view(-1)` creates `tensor.collapse_shape` operation which Stream-HLS doesn't support.

**Initial Attempt 3**: Tried `torch.gather` to avoid `view` operations:
```python
emb_expanded = self.emb_weight.unsqueeze(0).expand(batch_size, -1, -1)
x_expanded = x.unsqueeze(-1).expand(-1, -1, self.emb_dim)
x = torch.gather(emb_expanded, dim=1, index=x_expanded)
```
**Result**: Failed because `unsqueeze()` and `expand()` create `tensor.expand_shape` operations which Stream-HLS doesn't support.

**Final Fix**: Removed embedding layer entirely - accept embedded tensors as input:

```python
# In __init__:
# Before:
self.emb = nn.Embedding(num_tokens, dim)
self.emb_weight = nn.Parameter(...)

# After:
# Skip embedding layer - accept embedded tensors as input
self.num_tokens = num_tokens
self.emb_dim = dim

# In forward():
# Before:
x = self.emb(x)  # x: (batch, seq_len) token indices

# After:
# x: (batch, seq_len, dim) - already embedded tokens
# No embedding lookup needed - done on host side
```

**Files Modified**: 
- `bit_transformer.py` (lines 668-672, 687-693)
- `data.py` (line 191: changed input from token indices to embedded tensors)

**Benefits**:
- Completely avoids all unsupported operations (no `linalg.generic`, `tensor.collapse_shape`, `tensor.expand_shape`)
- Cleaner hardware design - embedding lookup done on host/CPU
- Common practice in HLS designs to separate embedding from compute
- Simpler model architecture
- No performance impact (embedding is typically done on host anyway)

**Note**: Embedding lookup must be done on the host side before calling the model. The input shape changes from `(batch, seq_len)` token indices to `(batch, seq_len, dim)` embedded tensors.

---

## Fix 3b: Unsupported `tensor.expand_shape` from `unsqueeze`/`expand` Operations

**Issue**:
```
error: 'tensor.expand_shape' op is unsupported operation.
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<1x512xi64> into tensor<1x512x1xi64>
```

**Location**: `bit_transformer.py` lines 695, 698 (in embedding forward pass using `torch.gather`)

**Problem**: The `unsqueeze()` and `expand()` operations used with `torch.gather` compile to `tensor.expand_shape`, which Stream-HLS doesn't support.

**Fix**: This issue was resolved by the final solution in Fix 3 - removing the embedding layer entirely and accepting embedded tensors as input. See Fix 3 for the complete solution.

**Note**: This demonstrates that even seemingly simple operations like `unsqueeze()` and `expand()` can create unsupported MLIR operations, which is why removing the embedding layer entirely was the best solution.

---

## Fix 4: Input Tensor Type and Shape

**Issue**: 
```
Error importing or initializing model 'bit_transformer': Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead
```

**Location**: `data.py` line 191 (bit_transformer input configuration)

**Problem**: The input tensor was:
- Wrong dtype: `torch.float32` instead of `torch.long` (when using embedding layer)
- Wrong shape: `(1, 512, 4096)` instead of `(batch, seq_len)` format

**Initial Fix**: Changed input to proper token indices:
```python
"input" : (
    torch.randint(0, 20000, (1, 512), dtype=torch.long),  # Token indices
)
```

**Updated Fix** (after removing embedding layer): Changed to embedded tensors:
```python
# After removing embedding layer (Fix 3):
"input" : (
    randTensor(1, 512, 128, dtype=dtype),  # Embedded tensors: (batch, seq_len, dim)
)
```

**Files Modified**: `data.py` (line 191)

**Note**: The input format changed from token indices `(batch, seq_len)` to embedded tensors `(batch, seq_len, dim)` after removing the embedding layer.

---

## Fix 5: Enhanced Error Reporting

**Issue**: Error messages were empty or unhelpful during model import/initialization.

**Location**: `gen_mlir_designs.py` line 80

**Fix**: Added full traceback printing for better debugging:

```python
# Before:
except Exception as e:
    print(f"Error importing or initializing model '{model}': {e}")

# After:
except Exception as e:
    import traceback
    print(f"Error importing or initializing model '{model}': {e}")
    print(f"Exception type: {type(e).__name__}")
    print("Full traceback:")
    traceback.print_exc()
```

**Files Modified**: `gen_mlir_designs.py` (lines 79-83)

---

## Summary of Changes

### Files Modified:
1. **`bit_transformer.py`**:
   - Added local `SimpleRMSNorm` and `OutputHead` classes
   - Added `import math`
   - Fixed `scaled_dot_product_gqa` to use `math.sqrt()` instead of `** 0.5`
   - Removed `nn.Embedding` layer entirely - model now accepts embedded tensors as input
   - Changed forward pass to accept `(batch, seq_len, dim)` embedded tensors instead of `(batch, seq_len)` token indices

2. **`data.py`**:
   - Changed input from token indices `(1, 512)` to embedded tensors `(1, 512, 128)`
   - Changed from `torch.randint()` to `randTensor()` for embedded tensor generation
   - Input now matches the model's expectation of pre-embedded tokens

3. **`gen_mlir_designs.py`**:
   - Enhanced error reporting with full tracebacks

### Key Principles for Stream-HLS Compatibility:

1. **Avoid tensor-to-Python conversions during forward pass**: Operations like `tensor ** 0.5` should use Python `math.sqrt()` instead.

2. **Avoid unsupported operations**: 
   - `nn.Embedding` → Remove entirely, accept embedded tensors as input (embedding done on host)
   - `F.one_hot` → Creates unsupported `linalg.generic` patterns
   - `tensor.view()` / `tensor.reshape()` → Creates `tensor.collapse_shape` which is unsupported
   - `tensor.unsqueeze()` / `tensor.expand()` → Creates `tensor.expand_shape` which is unsupported
   - `torch.index_select` / `torch.gather` with reshape operations → Remove embedding layer instead
   - Complex `linalg.generic` operations → Simplify to supported patterns or move to host side

3. **Use Python constants where possible**: Extract dimensions as Python values before using in computations.

4. **Input types matter**: 
   - When using embedding layers: requires `torch.long` indices, not `torch.float32`
   - After removing embedding: requires `torch.float32` embedded tensors with shape `(batch, seq_len, dim)`

5. **Move operations to host when needed**: If an operation creates unsupported MLIR patterns, consider moving it to the host side (e.g., embedding lookup). This is common practice in HLS designs.

---

## Testing

A test script was created at `test_bit_transformer.py` to verify the model works correctly:
- Tests with default configuration (dim=128, depth=6, num_tokens=20000)
- Tests with different batch sizes and sequence lengths
- Verifies output shapes and checks for NaN/Inf values

Run with:
```bash
python test_bit_transformer.py
```

---

## Fix 6: Unsupported `einsum` Operation and `linalg.fill` with Singleton Dimensions

**Issue**:
```
error: 'tensor.empty' op is unsupported operation.
    %0 = tensor.empty() : tensor<1x512x1xf32>
error: 'linalg.fill' op is unsupported operation.
    %1 = linalg.fill ins(%cst_11 : f32) outs(%0 : tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
error: "__module.transformer/__module.transformer.layers.0/__module.transformer.layers.0.q_proj/aten::einsum": found an op that was marked as backend illegal
```

**Location**: 
- `bit_transformer.py` line 23 (`SimpleRMSNorm.forward` with `keepdim=True`)
- `bit_transformer.py` line 63 (`activation_quant` function using `einsum`)

**Problem**: 
1. The `einsum` operation used in `activation_quant` to broadcast scale values is not supported by torch-mlir's backend legalization.
2. The error message about `tensor<1x512x1xf32>` and `linalg.fill` was misleading - the actual issue was `einsum` being unsupported.

**Fix**: Reverted `activation_quant` to use `keepdim=True` instead of `einsum`:

```python
# Before (using einsum):
def activation_quant(x: Tensor):
    max_vals = x.abs().max(dim=-1).values  # Shape: (...,)
    scale = 127.0 / max_vals.clamp_(min=1e-5)  # Shape: (...,)
    x_scaled = torch.einsum('...d,...->...d', x, scale)  # Unsupported!
    y = x_scaled.round().clamp_(-128, 127)
    y = torch.einsum('...d,...->...d', y, 1.0 / scale)  # Unsupported!
    return y

# After (using keepdim=True):
def activation_quant(x: Tensor):
    # Use keepdim=True to avoid einsum - the linalg.fill error was misleading
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y
```

**Note**: The `SimpleRMSNorm.forward` method was also reverted to use `keepdim=True` since the actual compilation error was from `einsum`, not from `keepdim=True`. The `linalg.fill` error message was a red herring - the real issue was the unsupported `einsum` operation.

**Files Modified**: 
- `bit_transformer.py` (lines 22-24: `SimpleRMSNorm.forward` - reverted to `keepdim=True`)
- `bit_transformer.py` (lines 63-71: `activation_quant` - removed `einsum`, reverted to `keepdim=True`)

**Benefits**:
- Avoids unsupported `einsum` operation
- `keepdim=True` works correctly with torch-mlir compilation
- Simpler code without complex broadcasting workarounds

---

## Known Warnings (Non-Critical)

The following TracerWarnings appear but don't prevent compilation:
- Tensor-to-Python boolean conversions in validation checks (lines 287, 293, 298, 311)
- Tensor-to-Python float conversion for scale computation (line 307)

These are expected during tracing and don't affect functionality.

---

## Status

✅ **Model compiles successfully to MLIR**
✅ **Model forward pass works correctly**
✅ **All tests pass**

The model is now compatible with Stream-HLS compilation pipeline.

