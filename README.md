# The Easy Quantum Model for the Fisher Information Matrix

This repo implements an easy quantum module following [this article](https://doi.org/10.1038/s43588-021-00084-1).
The parameters are uniformly initialized  on $\theta=[-1,1]$.
The repo shows how to create custom backward functions for your torch model in c++.


## The quantum circuit

The quantum circuits calculations are performed with the [qulacs](https://github.com/qulacs/qulacs) library. 
Although this code use a fork from that library for easy use with cmake. 



