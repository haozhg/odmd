# odmd
Python/Matlab implementation of online dynamic mode decomposition (Online DMD) and window dynamic mode decomposition (Window DMD)

## Online DMD algorithm description
At time step k, define two matrix X(k) = [x(1),x(2),...,x(k)], Y(k) = [y(1),y(2),...,y(k)], that contain all the past snapshot pairs, where x(k), y(k) are the n dimensional state vector, y(k) = f(x(k)) is the image of x(k), f() is the dynamics.  Here, if the (discrete-time) dynamics are given by z(k) = f(z(k-1)), then x(k), y(k) should be measurements corresponding to consecutive states z(k-1) and z(k).  

An exponential weighting factor rho=sigma^2 (0<rho<=1) that places more weight on recent data can be incorporated into the definition of X(k) and Y(k) such that X(k) = [sigma^(k-1)*x(1),sigma^(k-2)*x(2),…,sigma^(1)*x(k-1),x(k)], Y(k) = [sigma^(k-1)*y(1),sigma^(k-2)*y(2),...,sigma^(1)*y(k-1),y(k)].  

At time k+1, the matrices become X(k+1) = [x(1),x(2),…,x(k),x(k+1)], Y(k+1) = [y(1),y(2),…,y(k),y(k+1)]. We need to remember a new snapshot pair x(k+1), y(k+1). We can update the DMD matrix Ak = Yk*pinv(Xk) recursively by efficient rank-1 updating online DMD algorithm.  

The time complexity (multiply–add operation for one iteration) is O(n^2), and space complexity is O(n^2), where n is the state dimension.  

## Window DMD algorithm description
At time step k, define two matrix X(k) = [x(k-w+1),x(k-w+2),...,x(k)], Y(k) = [y(k-w+1),y(k-w+2),...,y(k)] that contain the recent w snapshot pairs from a finite time window, where x(k), y(k) are the n dimensional state vector, y(k) = f(x(k)) is the image of x(k), f() is the dynamics. Here, if the (discrete-time) dynamics are given by z(k) = f(z(k-1)), then x(k), y(k) should be measurements corresponding to consecutive states z(k-1) and z(k).  

An exponential weighting factor rho=sigma^2 (0<rho<=1) that places more weight on recent data can be incorporated into the definition of X(k) and Y(k) such that X(k) = [sigma^(w-1)*x(k-w+1),sigma^(w-2)*x(k-w+2),…,sigma^(1)*x(k-1),x(k)], Y(k) = [sigma^(w-1)*y(k-w+1),sigma^(w-2)*y(k-w+2),…,sigma^(1)*y(k-1),y(k)].  

At time k+1, the data matrices become X(k+1) = [x(k-w+2),x(k-w+3),…,x(k+1)], Y(k+1) = [y(k-w+2),y(k-w+3),…,y(k+1)]. We need to forget the oldest snapshot pair x(k-w+1),y(k-w+1), and remember the newest snapshot pair x(k+1),y(k+1). We can update the DMD matrix Ak = Yk*pinv(Xk) recursively by efficient rank-2 updating window DMD algroithm.  

The time complexity (multiply–add operation for one iteration) is O(n^2), and space complexity is O(wn+2n^2), where n is the state dimension, and w is the window size.  

## Installation
Download online DMD and window DMD implementation from github
`git clone https://github.com/haozhg/odmd.git`

## Implementation
1.**OnlineDMD.m** implements **OnlineDMD** class in Matlab.  
2.**WindomDMD.m** implements **WindowDMD** class in Matlab.  
3.**online.py** implements **OnlineDMD** class in Python.  
4.**window.py** implements **WindowDMD** class in Python.

## Documentation
Matlab:  
type **help OnlineDMD** for **OnlineDMD** class documentation.  
type **help WindowDMD** for **WindowDMD** class documentation.  
Python:  
type **help(online.OnlineDMD)** for **OnlineDMD** class documentation.  
type **help(window.WindowDMD)** for **WindowDMD** class documentation.

## Demos
1.**online_demo.m** demos the use of Matlab **OnlineDMD** class.  
2.**window_demo.m** demos the use of Matlab **WindowDMD** class.  
3.**online_demo.py** demos the use of Python **OnlineDMD** class.  
4.**window_demo.py** demos the use of Python **WindowDMD** class.

## Authors:
Hao Zhang  
Clarence W. Rowley

## Reference:
Hao Zhang, Clarence W. Rowley, Eric A. Deem, and Louis N. Cattafesta,
“Online Dynamic Mode Decomposition for Time-varying Systems,”
arXiv preprint arXiv:1707.02876, 2017.  

## Date created:
April 2017

