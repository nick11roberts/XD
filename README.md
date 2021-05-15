# XD

Code to reproduce results of the paper [Rethinking Neural Operations for Diverse Tasks](https://arxiv.org/abs/2103.15798).
The <tt>chrysalis</tt> folder contains the core code and applications, running which requires [PyTorch](https://pytorch.org/) (latest version recommended) and [Butterfly](https://github.com/HazyResearch/butterfly), in addition to the requirements in <tt>chrysalis/requirements.txt</tt>.
Install instructions below; a Dockerfile is also provided.
```
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch-nightly 
git clone --single-branch --branch pt1.8 https://github.com/HazyResearch/butterfly.git
cd butterfly && export FORCE_CUDA="1" && python setup.py install
```
