# RED-DiffEq: Regularization by Denoising Diffusion Models for Solving Inverse PDE Problems with Application to Full Waveform Inversion

## Directory Structure

```
red-diffeq/
├── dataset/                    # Raw and processed data
│   ├── Velocity_Data
│   ├── Seismic_Data
│   └── Test_Data
├── Diffusion_checkpoint_balanced/  # Pretrained diffusion model checkpoints
├── train_diffusion_model.py    # Script for diffusion model pre-training
├── main.py                     # Main script for full waveform inversion
├── README.md                   # Project overview
└── requirements.txt            # Python dependencies
```

## Example Usage

### Train the Diffusion Model

```bash
python train_diffusion_model.py
```

### Full Waveform Inversion with RED-DiffEq

```bash
python main.py --regularization 'diffusion' --lr 0.03 --ts 300 --sigma 10 --loss_type 'l1' --noise_std 0 --missing_number 0 --reg_lambda 0.75
```

### Baseline Benchmarks

#### No Regularization

```bash
python main.py --lr 0.03 --ts 300 --sigma 10 --loss_type 'l1' --noise_std 0 --missing_number 0
```

#### Total Variation Regularization

```bash
python main.py --regularization 'tv' --lr 0.03 --ts 300 --sigma 10 --loss_type 'l1' --noise_std 0 --missing_number 0
```

#### Tikhonov (L2) Regularization

```bash
python main.py --regularization 'l2' --lr 0.03 --ts 300 --sigma 10 --loss_type 'l1' --noise_std 0 --missing_number 0
```

### Noisy and Incomplete Data Scenarios

#### Additive Noise (Standard Deviation = 0.1)

```bash
python main.py --lr 0.03 --ts 300 --sigma 10 --loss_type 'l1' --noise_std 0.1 --missing_number 0
```

#### Five Missing Traces

```bash
python main.py --lr 0.03 --ts 300 --sigma 10 --loss_type 'l1' --noise_std 0.1 --missing_number 5
```
