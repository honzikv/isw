# ISW

## Requirements

`conda`, `miniconda`, or something that can install PyTorch

## Installation

Create new conda environment:

```zsh
conda create --name isw python=3.9

# CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# CPU only / macOS
pip3 install torch torchvision torchaudio

```

Install requirements:

```zsh
pip3 install -r requirements.txt
```

## QLearning

To train new model run `python train_qlearning.py --output-file <output_file>`

which will save model to `output_file`.

To run inference on the model run `python inference_qlearning.py --model-file <model_file>`

which will load model from `model_file` and run inference on the environment.

## Deep QLearning

To train new model run `python train_deep_qlearning.py`