# SparseMixer: Responsible AI Frequently Asked Questions 

## What is SparseMixer? 

SparseMixer is a scalable gradient estimator for Mixture-of-Expert models. 

## What can SparseMixer do?  

SparseMixer is able to provide reliable gradient approximation for expert routing, with only sparsely activated networks. 

## What is SparseMixer’s intended use? 

SparseMixer aims to facilitate the training of Mixture-of-Expert models. 

## How was SparseMixer evaluated? What metrics are used to measure performance? 

We conduct experiments on applying SparseMixer to Neural Machine Translation and Electra pre-training.
SparseMixer consistently outperforms the baseline methods in all 8 settings. 
More details are elaborated in our paper. 
 

## What are the limitations of SparseMixer? How can users minimize the impact of SparseMixer’s limitations when using the system? 

SparseMixer has first-order and second-order accuracy for gradient computation, and is only an approximation to the gradient. It excels as it facilitates a bias-variance tradeoff for the gradient estimation. 

## How to use SparseMixer?

`sparsemixer` can be installed via `pip`
```
pip install sparsemixer
```
Also, please check the `example` folder for a working example. 