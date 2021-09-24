## foolproofNN
A project for  investigating Adversarial Examples in NNs and their existence. 
Implemented with PyTorch.

Datasets:
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [MNIST](http://yann.lecun.com/exdb/mnist/)

Models:
- [Wide Residual Networks](https://arxiv.org/abs/1605.07146)

Attacks:
- CW L2 Attack: [N. Carlini and D. Wagner, "Towards Evaluating the Robustness of Neural Networks" 2017 IEEE Symposium on Security and Privacy (SP), 2017, pp. 39-57, doi: 10.1109/SP.2017.49.](https://ieeexplore.ieee.org/document/7958570)
- FGSM, PGD Attack: [I. Goodfellow, J. Shlens and C. Szegedy, "Explaining and Harnessing Adversarial Examples" 2015 International Conference on Learning Representations (ICLR), 2015.](https://arxiv.org/abs/1412.6572v3)
