# ComfyUI-NormalCrafterWrapper XFormers Fix

## Overview

This update addresses the "CUDA error: invalid configuration argument" that occurs with certain image resolutions when using xformers/flash attention. The solution provides a universal fix that works across different platforms (Windows/Linux) and hardware configurations.

## The Problem

The error occurs because:
- Flash attention (used by xformers) has strict requirements for tensor dimensions
- Certain image resolutions, after VAE encoding, create attention tensors that violate these constraints
- Different environments (PyTorch versions, xformers versions, GPUs) handle these edge cases differently

## The Solution

The updated node includes:

1. **Automatic Dimension Validation**: Detects potentially problematic dimensions before processing
2. **Smart Fallback Mechanism**: Automatically disables xformers and retries if a flash attention error occurs
3. **User Control**: New `use_xformers` parameter with three modes:
   - `auto` (default): Tries xformers, falls back to standard attention on error
   - `disable`: Never uses xformers (slower but most compatible)
   - `force`: Always uses xformers (may fail with certain resolutions)
4. **Dimension Adjustment**: In auto mode, adjusts problematic dimensions to safe values
5. **Aspect Ratio Switching Support**: Handles switching between images with different aspect ratios without errors

## Usage

### Basic Usage (Recommended)
Just use the node normally - the default `auto` mode will handle most cases:
```
use_xformers: auto  # Default - handles errors automatically
```

### If You Still Get Errors
Set xformers to disable:
```
use_xformers: disable  # Most compatible, but slower
```

### For Maximum Performance
If you know your resolutions work well:
```
use_xformers: force  # Fastest, but may error on some resolutions
```

## Technical Details

### Safe Dimensions
The node ensures dimensions are:
- Divisible by 8 after VAE encoding (latent space requirement)
- At least 16 pixels in the latent space (minimum for attention)
- Properly aligned for flash attention kernels

### Frame Consistency
When processing:
- All frames (including padded frames) are resized to the same dimensions
- Dimension consistency is verified before processing
- Handles aspect ratio changes between runs gracefully

### Compatibility
- Works with PIL/Pillow 9.x and 10.x
- Compatible with various PyTorch versions (2.0+)
- Handles different xformers versions gracefully
- Cross-platform (Windows, Linux, MacOS)

## Troubleshooting

If you still experience issues:

1. **Update Dependencies**:
   ```bash
   pip install --upgrade torch torchvision xformers
   ```

2. **Check GPU Driver**: Ensure you have the latest NVIDIA driver

3. **Memory Issues**: The fallback to standard attention uses more memory. If you get OOM errors:
   - Reduce `max_res_dimension`
   - Reduce `decode_chunk_size`
   - Use `offload_pipe_to_cpu_on_finish: true`

4. **Report Persistent Issues**: If problems persist, please report:
   - Exact image dimensions
   - GPU model and driver version
   - PyTorch and xformers versions
   - Full error message

## Performance Notes

- `auto` mode: Slight overhead for dimension checking, but handles all cases
- `disable` mode: ~30-50% slower than xformers, but universally compatible
- `force` mode: Fastest option when it works

The automatic fallback ensures your workflows won't break due to resolution-dependent errors while maintaining performance when possible. 