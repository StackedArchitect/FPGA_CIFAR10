# CIFAR-10 CNN Software Model

This folder contains the PyTorch training and export script for the CIFAR-10 baseline model used by the FPGA flow.

The main file is [cifar10_cnn_baseline.py](cifar10_cnn_baseline.py).

## What the Script Does

- trains a full-precision CNN with BatchNorm on CIFAR-10
- evaluates the best checkpoint on the test split
- exports weights, biases, and folded BatchNorm parameters to Q16.16 `.mem` files
- exports one normalized test image as `data_in.mem`
- exports the expected label as `expected_label.mem`
- saves a training curve plot

## Model Summary

- Input: `32 x 32 x 3`
- Convolution blocks: `Conv1 -> BN1 -> ReLU -> Pool`, `Conv2 -> BN2 -> ReLU -> Pool`, `Conv3 -> BN3 -> ReLU`
- Global average pooling before the fully connected head
- `FC1 -> BN4 -> ReLU -> Dropout`
- `FC2 -> logits`

## Important Runtime Notes

- The script is intended for CPU training in its current form.
- On Windows, `num_workers=0` is used for the `DataLoader` to avoid multiprocessing bootstrap issues.
- The script uses the `Agg` backend for Matplotlib so it can save plots without opening a GUI window.

## Required Python Packages

```bash
pip install torch torchvision numpy matplotlib
```

## Typical Run

From this directory:

```bash
python cifar10_cnn_baseline.py
```

The first run may download CIFAR-10 into the local `data/` folder if it is not already present.

## Generated Files

The script produces the following artifacts next to the model or under the repo's `weights/` directory:

- `cifar10_cnn_baseline_model.pth`
- `cifar10_training_curve.png`
- `.mem` files for convolution weights, fully connected weights, BatchNorm scale/shift, the sample image, and the expected label

## Hardware Link

The `.mem` outputs are used by the SystemVerilog hardware in [../hardware](../hardware) and the synthesis wrapper [../hardware/cnn2d_synth_top_cifar.sv](../hardware/cnn2d_synth_top_cifar.sv).
