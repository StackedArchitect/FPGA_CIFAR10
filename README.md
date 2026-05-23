# FPGA_CIFAR10

A CIFAR-10 FPGA project combining a full-precision PyTorch software model with SystemVerilog hardware for Zynq-7020 deployment. The repository includes the CIFAR-10 baseline hardware flow, the software training/export pipeline, and the related testbenches and timing constraints.

## Repository Layout

```text
cifar10_baseline/
  hardware/
    cnn2d_synth_top_cifar.sv
    cnn2d_timing.xdc
    cnn2d_top_cifar.sv
    conv_pool_2d_cifar.sv
    global_avg_pool_cifar.sv
    layer_seq_cifar.sv
    tb_cnn2d_cifar.sv
  software/
    cifar10_cnn_baseline.py
    README.md
    data/
      ... CIFAR-10 dataset cache ...
    cifar10_cnn_baseline_model.pth
  weights/
    ... generated .mem weight and input files ...
```

## What This Project Does

- Trains a CIFAR-10 CNN in PyTorch.
- Exports weights, biases, BatchNorm parameters, and a sample input to Q16.16 `.mem` files.
- Instantiates the hardware design from SystemVerilog ROMs and sequential layers.
- Uses an argmax output so the FPGA can report the predicted class directly.

## Software Flow

The main training script is [cifar10_baseline/software/cifar10_cnn_baseline.py](cifar10_baseline/software/cifar10_cnn_baseline.py). It:

- loads CIFAR-10 through `torchvision`
- trains the student CNN
- saves the best checkpoint as `cifar10_cnn_baseline_model.pth`
- exports weights and BN parameters for hardware
- writes `data_in.mem` and `expected_label.mem`
- generates a training curve image

For a detailed description of the software model and export files, see [cifar10_baseline/software/README.md](cifar10_baseline/software/README.md).

## Hardware Flow

The hardware directory contains the CIFAR-10 accelerator implementation and simulation entry points:

- `cnn2d_top_cifar.sv` for the layer composition
- `cnn2d_synth_top_cifar.sv` for ROM-backed synthesis integration
- `layer_seq_cifar.sv` for sequential fully connected execution
- `conv_pool_2d_cifar.sv` and `global_avg_pool_cifar.sv` for the convolution/pooling stages
- `tb_cnn2d_cifar.sv` for simulation
- `cnn2d_timing.xdc` for timing constraints

## Large Files

Large generated artifacts are intentionally excluded from version control, including:

- CIFAR-10 dataset caches
- training checkpoints (`.pth`)
- generated `.mem` files
- temporary archives and packaged outputs

## Run Requirements

Install the Python dependencies first:

```bash
pip install torch torchvision numpy matplotlib
```

Then run the software model:

```bash
cd cifar10_baseline/software
python cifar10_cnn_baseline.py
```

## Notes

- The software script uses Q16.16 fixed-point export for hardware compatibility.
- The generated `.mem` files are meant to be loaded by the SystemVerilog ROMs.
- The repo is structured so that the software output can be fed directly into the hardware simulation and synthesis flow.
