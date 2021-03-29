# Iterative Random Concensus Projection

given source set <img src="https://latex.codecogs.com/gif.latex?p_{src}%20=%20\{s_i\}_{i=1}^N,%20s%20\in%20\mathbf{R}^3" /> and destination set with normals <img src="https://latex.codecogs.com/gif.latex?p_{dst}%20=%20\{d_i,n_i\}_{i=1}^N,%20d%20\in%20\mathbf{R}^3,n%20\in%20\mathbf{R}^3" />, find the best transformation <img src="https://latex.codecogs.com/gif.latex?\bold{T}" /> such that:

<img src="https://latex.codecogs.com/gif.latex?\bold{\hat{T}} = \mathop {\arg \min }\limits_\bold{T} \sum \limits_{i=1}^N{\|n_i^T (d_i -  \bold{T}\cdot c_i) \|^2}" />

![pre](res/pre.png)
![post](res/post.png)
 

### Refrences
[ [1](https://apps.dtic.mil/sti/pdfs/ADA460585.pdf) ]Fischler, Martin A., and Robert C. Bolles. "Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography." Communications of the ACM 24.6 (1981): 381-395.

[ [2](https://www.researchgate.net/profile/Steven_Blostein/publication/224378053_Least-squares_fitting_of_two_3-D_point_sets_IEEE_T_Pattern_Anal/links/5633c61a08aeb786b7013b28/Least-squares-fitting-of-two-3-D-point-sets-IEEE-T-Pattern-Anal.pdf) ] Arun, K. Somani, Thomas S. Huang, and Steven D. Blostein. "Least-squares fitting of two 3-D point sets." IEEE Transactions on pattern analysis and machine intelligence 5 (1987): 698-700.

[ [3](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.116.7292&rep=rep1&type=pdf) ]  Low, Kok-Lim. "Linear least-squares optimization for point-to-plane icp surface registration." Chapel Hill, University of North Carolina 4.10 (2004): 1-3.

 
##install

### prequisites
* cuda >= 11
* cuda toolkit https://developer.nvidia.com/cuda-downloads
```
sudo apt install nvidia-driver-XXX
sudo apt install nvidia-cuda-toolkit
```
* cmake >= 3.15
```
sudo apt remove --purge cmake
sudo snap install cmake --classic
```
* ```sudo apt-get install libgflags-dev libopencv-dev libboost-all-dev python3.6-dev```

### package

```
python -m venv venv
source venv/bin/activate
pip install git+https://github.com/ohadmen/pyircp
```
