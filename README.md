# BoltzmannMachineLearning
Implementation of the Boltzmann Machine Learning rules, to learn the group statistics of salamander retina neurons. In the Boltzmann Machine approximation, the dynamics of the individual neurons are neglected, and the system is modelled using couplings between the neurons and thresholds for 'firing'. This implementation explores a Boltzmann Machine with no hidden layer, which means that all neurons are fully connected with each other. 

## Report
A comprehensive analysis is provided in the write-up [Boltzmann Machine learning.pdf](https://github.com/joeranbosma/BoltzmannMachineLearning/blob/master/Boltzmann%20Machine%20learning.pdf). 

## Code

This projects contains three parts:
1. Exact computation of statistics
2. Approximation of statistics using Monte Carlo sampling
3. Application of the Boltzmann Machine using Metropolis-Hasting Monte Carlo Markov Chain sampling, for the salamander retina neurons

The first two parts investigate the accuracy of the employed approximations, from which follows that the Metropolis-Hasting Monte Carlo Markov Chain sampler achieves the best performance. This sampler has also been implemented in C++, to speed up calculations in part 3.  

Part 3 employs Metropolis-Hasting Monte Carlo Markov Chain sampling to learn the dynamics of a group of 160 neurons, as measured in a salamander. Here, the learned dynamics are also employed to predict the joint firing rates between multiple neurons. 

In order to compile the C++ extension, follow the instructions in `include/Eigen.md` and `include/Pybind11.m5`. Then, compile from the command line with `python3 setup.py build`. 
