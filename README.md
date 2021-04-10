Text Guided Image Manipulation  <img src="https://pytorch.org/assets/images/logo-dark.svg" width = "90" align=center />

###### This project is under working: TaGAN + Transformer + auxiliary Loss


    
Setup and Dependencies
----------------------

This code is implemented using PyTorch v1.6.0, and provides out of the box support with CUDA 10.1

Anaconda / Miniconda is the recommended to set up this codebase.

### Anaconda or Miniconda

Clone this repository and create an environment:

```shell
git clone https://www.github.com/ecpark4545/TGIM
conda create -n TGIM python=3.7

# activate the environment and install all dependencies
conda activate TGIM
cd TGIM

# https://pytorch.org
pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

And set data_root and save_path in cfg/birds.yml

Preparing Data
-------------
We provide preprocessed dataset [birds](https://drive.google.com/file/d/1QLFpsHGbVN-sU2bFXn7Au1uGmLrhatkq/view?usp=sharing). 


Training
-------------
you can simply run the model with this code
```shell
python main.py --gpu 0 --batch_size 32
```
 
- This code borrows heavily from [TaGAN](https://github.com/woozzu/tagan) repository. Many thanks.
