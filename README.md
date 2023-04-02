# ISW

## Requirements

`conda`, `miniconda`, or something that can install PyTorch

## Installation

Create new conda environment:

```zsh
conda create --name isw python=3.11

# CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# CPU only / macOS
pip3 install torch torchvision torchaudio

```

Install requirements:

```zsh
pip3 install -r requirements.txt
```

# Q-Learning 

To train: `python train_qlearning.py --output-file <output_file>`

which will save model to `output_file`.

To run inference: `python inference_qlearning.py --model-file <model_file>`

**Pre-trained model** for this task is available in `pretrained/qlearning.ckpt`.

# Deep Q-Learning 🧠

To train: `python train_deep_qlearning.py`

To run inference: `python inference_deep_qlearning.py --model-path <weights_path>`

**Pre-trained model** for this task is available in `pretrained/deep_qlearning.ckpt`.


# Running pre-trained

Q-Learning: `python inference_qlearning.py --model-path pretrained/qlearning.ckpt`

Deep Q-Learning: `python inference_deep_qlearning.py --model-path pretrained/deep_qnet.ckpt`
