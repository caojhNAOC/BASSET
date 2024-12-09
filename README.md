# <img src="BASSET_logo.png" width="200" />
# Bandpass-Adaptive Single-pulse SEarch Toolkit (BASSET)

**BASSET** is a new user-friendly search toolkit for [PRESTO](https://github.com/scottransom/presto).  
BASSET can significantly improve the detection sensitivity for narrow-band pulses and has the potential to facilitate the search in broad-band observations.

## How to Install BASSET

Place the corresponding `BASSET.c` and `BASSET.h` files into the `src` and `include` directories of PRESTO.  
Update the Makefile (an example is provided in the `src` directory of this library), then run `make`.  
Afterwards, `prepsubband_BASSET` and `prepsubband_BASSET_accelerate` will appear in the PRESTO `bin` folder.  
They function similarly to the `prepsubband` in [PRESTO](https://github.com/scottransom/presto).

## How to Use BASSET

### prepsubband_BASSET

Compared to `prepsubband`, the following two parameters have been added:

- `BASSET_Ls`: Lengths of the box-car function to match-filter pulses, with units in channels (at least 5 lengths, comma-separated string, no spaces).
- `BASSET_minL`: Candidates with bandwidth smaller than `BASSET_minL` will be removed. This value is typically chosen as 1/4 of the minimum of `BASSET_Ls`.

### Command Example:
```bash
prepsubband_BASSET -nobary -nsub 4096 -lodm 1202 -dmstep 0 -numdms 1 -BASSET_minL 100 -BASSET_Ls 100,200,300,400,500,600,800,1000,1200,1400,1600,1800,2000 -ignorechan 0:16,690:837,1420:1450,2775:2855,3800:3970 -o FAST_data.fits FAST_data.fits
```

### prepsubband_BASSET_accelerate

BASSET applies GPU and OpenMP acceleration. It uses only 1 GPU by default, but you can specify which GPU to use by setting the `CUDA_VISIBLE_DEVICES` environment variable (see the [CUDA documentation](https://developer.nvidia.com/cuda-toolkit)).

OpenMP is controlled by the `-ncpus` parameter, which specifies the number of processors to use.

- **`-ncpus`**: Number of processors to use with OpenMP.

### Command Example:
```bash
CUDA_VISIBLE_DEVICES=1 prepsubband_BASSET -ncpus 20 -nobary -nsub 4096 -lodm 1202 -dmstep 0 -numdms 1 -BASSET_minL 100 -BASSET_Ls 100,200,300,400,500,600,800,1000,1200,1400,1600,1800,2000 -ignorechan 0:16,690:837,1420:1450,2775:2855,3800:3970 -o FAST_data.fits FAST_data.fits
```

## How to Mask RFI Using `rfi-mask.py`

**rfi-mask.py** is designed to mask Radio Frequency Interference (RFI) from FITS files based on wavelet transformation. It is similar to `rfifind` in [PRESTO](https://github.com/scottransom/presto). It requires three input parameters:

- `file`: The FITS file.
- `time`: The time in seconds used for the `rfifind`.
- `sigma`: The threshold value for wavelet-based RFI removal.

### Command Example:

```bash
python rfi-mask.py -file FAST_data.fits -time 0.06 -sigma 0.5
```
