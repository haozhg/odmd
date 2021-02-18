# odmd
Python/Matlab implementation of online dynamic mode decomposition (Online DMD) and window dynamic mode decomposition (Window DMD) algorithms proposed in this [paper](https://epubs.siam.org/doi/pdf/10.1137/18M1192329).

## Hightlights
- The online algorithm is optimal in terms of both time and space complexity. Its time complxity is O(n^2), where n is the state dimension. This is much faster than standard algorithm O(n^3). Its space complexity is O(n^2), which is much more efficient than standard algorithm O(n*T), where T is the number of measurements (T >> n, and will go to infinity in online applications)
- It finds the exact optimal solution, without any approximation (Unlike stochastic gradient descent). 
- It is a very general algorithm, and applies to problems other than fluid dynamics, dynamical systems, and control. If we have a function (or a dynamical model) y = M*phi(x), where phi is some known arbitrary nonlinear vector-valued function, x, y are vectors, M is an unknown matrix, and measurement x, y comes in real-time, then this algorithm updates the model M optimally. This form is very general, and can represent a very large class of models such as LTI, Lorenz attractor, polynomial system, vector AR, and many more.
- It can be used for model predictive control.
- It has been successfully applied to flow control problems, and achived great real-time closed loop control. See this [paper](https://doi.org/10.1017/jfm.2020.546) for details.

## Installation
### Use pip
```
python3 -m pip install odmd
```

### Manual install
Create virtual env if needed
```
python3 -m venv .venv
source .venv/bin/activate
```

Clone from github and install
```
git clone https://github.com/haozhg/odmd.git
cd odmd/
python3 -m pip install -e .
```

### Test
To run tests
```
cd tests/
pip install r requirements.txt
python -m pytest .
```

## Algorithm Description
Here is a brief description of the proposed algorithms. For more details, please refer to the original [paper](https://epubs.siam.org/doi/pdf/10.1137/18M1192329).

### Online DMD algorithm description
The algorithm is implemented in class [OnlineDMD](./odmd/online.py).

At time step k, define two matrix X(k) = [x(1),x(2),...,x(k)], Y(k) = [y(1),y(2),...,y(k)], that contain all the past snapshot pairs, where x(k), y(k) are the n dimensional state vector, y(k) = f(x(k)) is the image of x(k), f() is the dynamics.  Here, if the (discrete-time) dynamics are given by z(k) = f(z(k-1)), then x(k), y(k) should be measurements corresponding to consecutive states z(k-1) and z(k).  

An exponential weighting factor rho=sigma^2 (0<rho<=1) that places more weight on recent data can be incorporated into the definition of X(k) and Y(k) such that X(k) = [sigma^(k-1)*x(1),sigma^(k-2)*x(2),…,sigma^(1)*x(k-1),x(k)], Y(k) = [sigma^(k-1)*y(1),sigma^(k-2)*y(2),...,sigma^(1)*y(k-1),y(k)].  

At time k+1, the matrices become X(k+1) = [x(1),x(2),…,x(k),x(k+1)], Y(k+1) = [y(1),y(2),…,y(k),y(k+1)]. We need to remember a new snapshot pair x(k+1), y(k+1). We can update the DMD matrix Ak = Yk*pinv(Xk) recursively by efficient rank-1 updating online DMD algorithm.  

The time complexity (multiply–add operation for one iteration) is O(n^2), and space complexity is O(n^2), where n is the state dimension.  

### Window DMD algorithm description
The algorithm is implemented in class [WindowDMD](./odmd/window.py).

At time step k, define two matrix X(k) = [x(k-w+1),x(k-w+2),...,x(k)], Y(k) = [y(k-w+1),y(k-w+2),...,y(k)] that contain the recent w snapshot pairs from a finite time window, where x(k), y(k) are the n dimensional state vector, y(k) = f(x(k)) is the image of x(k), f() is the dynamics. Here, if the (discrete-time) dynamics are given by z(k) = f(z(k-1)), then x(k), y(k) should be measurements corresponding to consecutive states z(k-1) and z(k).  

An exponential weighting factor rho=sigma^2 (0<rho<=1) that places more weight on recent data can be incorporated into the definition of X(k) and Y(k) such that X(k) = [sigma^(w-1)*x(k-w+1),sigma^(w-2)*x(k-w+2),…,sigma^(1)*x(k-1),x(k)], Y(k) = [sigma^(w-1)*y(k-w+1),sigma^(w-2)*y(k-w+2),…,sigma^(1)*y(k-1),y(k)].  

At time k+1, the data matrices become X(k+1) = [x(k-w+2),x(k-w+3),…,x(k+1)], Y(k+1) = [y(k-w+2),y(k-w+3),…,y(k+1)]. We need to forget the oldest snapshot pair x(k-w+1),y(k-w+1), and remember the newest snapshot pair x(k+1),y(k+1). We can update the DMD matrix Ak = Yk*pinv(Xk) recursively by efficient rank-2 updating window DMD algroithm.  

The time complexity (multiply–add operation for one iteration) is O(n^2), and space complexity is O(wn+2n^2), where n is the state dimension, and w is the window size.  

## Demos
See [demo](./demo) for python notebooks.

## Authors
Hao Zhang  
Clarence W. Rowley

## Reference
If you you used these algorithms or this python package in your work, please consider citing

```
Zhang, Hao, Clarence W. Rowley, Eric A. Deem, and Louis N. Cattafesta. "Online dynamic mode decomposition for time-varying systems." SIAM Journal on Applied Dynamical Systems 18, no. 3 (2019): 1586-1609.
```

BibTeX
```
@article{zhang2019online,
  title={Online dynamic mode decomposition for time-varying systems},
  author={Zhang, Hao and Rowley, Clarence W and Deem, Eric A and Cattafesta, Louis N},
  journal={SIAM Journal on Applied Dynamical Systems},
  volume={18},
  number={3},
  pages={1586--1609},
  year={2019},
  publisher={SIAM}
}
```

## Date created
April 2017

## License
If you want to use this package, but find license permission an issue, pls contact me at `haozhang at alumni dot princeton dot edu`.
