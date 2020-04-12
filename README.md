# BoltzmannMachineLearning
Implementation of the Boltzmann Machine Learning rules, to learn the dynamics of salamander retina neurons. In this project, the dynamics of a group of neurons from a salamander retina are modelled using a Boltzmann Machine. In this approximation, the dynamics of the individual neurons are neglected, and the group of neurons is modelled using couplings between the neurons and thresholds for 'firing'. This implementation explores a Boltzmann Machine with no hidden layer, which means that all neurons are fully connected with each other.  

This projects contains three parts:
1. Exact computation of statistics
2. Approximation of statistics using Monte Carlo sampling
3. Application of the Boltzmann Machine using Metropolis-Hasting Monte Carlo Markov Chain sampling, for the salamander retina neurons

The first two parts investigate the accuracy of the employed approximations, from which follows that the Metropolis-Hasting Monte Carlo Markov Chain sampler achieves the best performance. This sampler has also been implemented in C++, to speed up calculations in part 3.  

Part 3 employs Metropolis-Hasting Monte Carlo Markov Chain sampling to learn the dynamics of a group of 160 neurons, as measured in a salamander. Here, the learned dynamics are also employed to predict the joint firing rates between multiple neurons. 

A comprehensive analysis is provided in the write-up `Boltzmann Machine learning.pdf`. 

In order to compile the C++ extension, follow the instructions in `include/Eigen.md` and `include/Pybind11.m5`. 
