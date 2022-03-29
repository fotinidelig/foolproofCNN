## foolproofNN
A project for  investigating Adversarial Examples in CNNs. 
Implemented with PyTorch.


|Original|Perturbation|Adversarial|
|----------|------|---------|
|<img src='https://user-images.githubusercontent.com/44481663/159064311-2443b1f8-7d69-48fe-af42-de909367f071.svg' width=120>  | <img src='https://user-images.githubusercontent.com/44481663/159064323-0d8770bc-ad96-4b4b-9416-3ce451a7442c.svg' width=120> |<img src='https://user-images.githubusercontent.com/44481663/159064331-8bf7e2a7-5637-4dca-b1d9-3bc64fef1e07.svg' width=120> |

<!-- |Albatross||Tit mouse| -->

## About me

**Datasets used:**
- [*CIFAR-10*](https://www.cs.toronto.edu/~kriz/cifar.html)
- [*MNIST*](http://yann.lecun.com/exdb/mnist/)
- [*350 Bird Species*](https://www.kaggle.com/gpiosenka/100-bird-species/code?datasetId=534640&sortBy=voteCount) (version 47)

**Models implemented from scratch (use flag --model wideresnet):**
- [*Wide Residual Networks*](https://arxiv.org/abs/1605.07146) (accuracy on CIFAR-10 approx. 82%)

**Attacks implemented from scratch:**
- *CW L2 Attack*: [N. Carlini and D. Wagner, "Towards Evaluating the Robustness of Neural Networks" 2017 IEEE Symposium on Security and Privacy (SP), 2017, pp. 39-57, doi: 10.1109/SP.2017.49.](https://ieeexplore.ieee.org/document/7958570)
- *FGSM and PGD Attacks*: [I. Goodfellow, J. Shlens and C. Szegedy, "Explaining and Harnessing Adversarial Examples" 2015 International Conference on Learning Representations (ICLR), 2015.](https://arxiv.org/abs/1412.6572v3)

**Defences implemented from scratch:**
- *TRADES*: [H. Zhang, Y. Yu, J. Jiao, E. P. Xing, L. El Ghaoui and M. I. Jordan, "Theoretically Principled Trade-off between Robustness and Accuracy" 2019 International Conference on Machine Learning, 2019.](https://arxiv.org/pdf/1901.08573.pdf)

## How to use me
*Note*: All attack methods and defences (and all consequent functions) can be used on their own as a tool, simply by importing the correct python modules.

**General arguments:** 
`--model_name` specifies where to store or to load from a model.

`--model` sets the model architecture and might need some other parameters.

`--dataset` and `root` are used to choose either a pre-existing `torch` dataset or to specify the root directory of the image data.

**Training a model**

At runtime you can train the model of your choice and add optional arguments.

Simply run e.g. `./train.py --model cwcifar10 --dataset cifar10 --model_name Test-CIFAR-10`

and to see the usage, run `./train.py --h`
```
usage: train.py [-h] [--dataset {cifar10,mnist}] [--root ROOT]
                [--filter {high,low,band}] [--threshold THRESHOLD]
                [--model_name MODEL_NAME]
                [--model {cwcifar10,cwmnist,wideresnet,resnet,effnet,googlenet}]
                [--layers {18,34,50,101}] [--depth DEPTH] [--width WIDTH]
                [--input_size INPUT_SIZE] [--output_size OUTPUT_SIZE]
                [--pre-trained] [--transfer_learn] [--augment]
                [--epochs EPOCHS] [--lr LR] [--lr-decay LR_DECAY]
```

**Running an attack**

Here you have various optional arguments for the target model or attack method.

For example, to run a PGD attack use `./attack.py --attack pgd --norm 2 --iters 40 --alpha 0.004 --epsilon 0.01 --samples 15 --model cwcifar10`

and again see all parameters with `./attack.py --h`
```
usage: attack.py [-h] [--dataset {cifar10,mnist}] [--root ROOT]
                 [--filter {high,low,band}] [--threshold THRESHOLD]
                 [--model_name MODEL_NAME]
                 [--model {cwcifar10,cwmnist,wideresnet,resnet,effnet,googlenet}]
                 [--layers {18,34,50,101}] [--depth DEPTH] [--width WIDTH]
                 [--input_size INPUT_SIZE] [--output_size OUTPUT_SIZE] [--cpu]
                 [--attack {cw,pgd,boundary}] [--samples SAMPLES]
                 [--batch BATCH] [--targeted] [--lr LR] [--epsilon EPSILON]
                 [--alpha ALPHA] [--iters ITERS] [--norm {2,inf}]
```
**Training with TRADES**

Similar to normal training, with some additional parameters for the TRADES algorithm.
As an example, try running `./trades_train.py --model cwcifar10 --dataset cifar10 --lambda 0.5 --norm 2 --iters 50 --alpha 0.009`

```
usage: trades_train.py [-h] [--dataset {cifar10,mnist}] [--root ROOT]
                       [--filter {high,low,band}] [--threshold THRESHOLD]
                       [--model_name MODEL_NAME]
                       [--model {cwcifar10,cwmnist,wideresnet,resnet,effnet,googlenet}]
                       [--layers {18,34,50,101}] [--depth DEPTH]
                       [--width WIDTH] [--input_size INPUT_SIZE]
                       [--output_size OUTPUT_SIZE] [--pre-trained]
                       [--transfer_learn] [--augment] [--epochs EPOCHS]
                       [--lr LR] [--lr-decay LR_DECAY] [--lambda _LAMBDA]
                       [--norm {inf,2}] [--epsilon EPSILON] [--alpha ALPHA]
                       [--iters ITERS]
```
## Preliminary experimental results
