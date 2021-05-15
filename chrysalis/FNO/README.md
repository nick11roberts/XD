# FNO XD

Code to reproduce the PDE experiments in [Rethinking Neural Operations for Diverse Tasks](https://arxiv.org/abs/2103.15798). 
First, [download the datasets from the Fourier Neural Operators paper](https://github.com/zongyi-li/fourier_neural_operator) to a folder, such as `data/`. 
Due to PyTorch compatibility issues, we defer to the [original FNO codebase](https://github.com/zongyi-li/fourier_neural_operator) for their implementation of the Fourier Neural Operator. 

NOTE: at higher resolutions, some of these may require a lot of GPU memory. If this becomes an issue, try setting different values of the `--acc_steps` flag for gradient accumulation. 

To reproduce the XD and conv results over different resolutions for the Bugers' equation, run the following
```
DATA=data/ ./run/run_1d.sh
```

To reproduce the XD and conv results over different resolutions for Darcy Flow 
```
DATA=data/ ./run/run_2d.sh
```

Similarly, for the 3d problems
```
DATA=data/ ./run/run_3d.sh
```
