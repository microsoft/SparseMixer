![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![GitHub](https://img.shields.io/github/license/microsoft/sparsemixer) 

<h2 align="center">SparseMixer</h2>
<h4 align="center">Sparse Backpropagation for Mixture-of-Expert Training</h4>

<p align="center">
  <a href="#st">Mixture-of-Expert</a> •
  <a href="#sparsemixer">SparseMixer</a> •
  <a href="#how-to-use">How to Use?</a> •
  <a href="#examples">Examples</a> •
  <a href="#citation">Citation</a> •
  <a href="https://github.com/microsoft/sparsemixer/tree/main/LICENSE">License</a>
</p>

[sparsemixer](https://arxiv.org/pdf/2304.08612.pdf), a scalable gradient estimator, bridges the gap between backpropagation and sparse expert routing.

<h3 align="center" id="st"><i>What is Mixture-of-Expert</i></h4>
The significant success of large-scale pre-training across various applications has underscored the imperative need for scalable models that are economically feasible. 
Recent advances in sparsely activated networks, prominently known as Mixture-of-Experts (MoE), have attracted widespread interest. 
Unlike traditional networks that densely activate all modules for all input, MoE selectively activates parts of modules to specific inputs through a process called {expert routing}, leading to notable efficiency enhancements.

Numerous methods have emerged to bridge discrete and back-propagation, and most of them are based on Straight-Through (ST). 
Unfortunately, all existing ST estimators are incompatible with MoE, since they require activating all experts for gradient computing, thereby eliminating all the efficiency improvements of MoE. 
Consequently, typical MoE training strategically neglects the gradient computation for routing, trading certain training signals for sparse computation. 
Despite the scalability brought by sparse computation, this trade-off may result in slow convergence and improperly trained models.  

<h3 align="center" id="sparsemixer"><i>Backpropagation Made Sparse</i></h3>

We propose [sparsemixer](https://arxiv.org/pdf/2304.08612.pdf), a scalable gradient estimator, bridges the gap between backpropagation and sparse expert routing.
Grounded in a numerical ODE framework, SparseMixer harnesses the mid-point method, a second-order ODE solver, to deliver precise gradient approximations with negligible computational overhead. 
Applying SparseMixer to Switch Transformer on both pre-training and machine translation tasks, SparseMixer showcases considerable performance gain, accelerating training convergence up to 2 times

### How to use?

`sparsemixer` can be installed via `pip`
```
pip install sparsemixer
```

### Examples

Please check the `example` folder for a working example. 

### Citation
Please cite the following papers if you found our model useful. Thanks!


>Liyuan Liu, Jianfeng Gao, and Weizhu Chen (2023). Sparse Backpropagation for MoE Training. *ArXiv, abs/2304.08612*.
```
@inproceedings{liu2023bridging,
  title={Sparse Backpropagation for MoE Training},
  author = {Liu, Liyuan and Gao, Jianfeng and Chen, Weizhu},
  booktitle = {arXiv:2304.08612 [cs]},
  year={2023}
}
```

>Liyuan Liu, Chengyu Dong, Xiaodong Liu, Bin Yu, and Jianfeng Gao (2023). Bridging Discrete and Backpropagation: Straight-Through and Beyond. *ArXiv, abs/2304.08612*.
```
@inproceedings{liu2023bridging,
  title={Bridging Discrete and Backpropagation: Straight-Through and Beyond},
  author = {Liu, Liyuan and Dong, Chengyu and Liu, Xiaodong and Yu, Bin and Gao, Jianfeng},
  booktitle = {arXiv:2304.08612 [cs]},
  year={2023}
}
```