import argparse

def parser(train=True, advtrain=False, attack=False):
    parser = argparse.ArgumentParser()

    ## General arguments
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'mnist'],
                        help='define torch dataset to train on (def. none)')
    parser.add_argument('--root', default="./data/", type=str,
                        help='path that dataset\'s test/ train/ folders are stored (def. `../data/`)')
    # Filter
    parser.add_argument('--filter', default=None, choices=['high', 'low', 'band'],
                        help='filter dataset images in frequency space (def. none)')
    parser.add_argument('--threshold', default=None, type=str,
                        help='filter threshold, use `,` in case of two values (def. none)')
    # Model
    parser.add_argument('--model_name', default=None, type=str,
                        help='name to save or load model from')
    parser.add_argument('--model', default='cwcifar10', choices=['cwcifar10', 'cwmnist', 'wideresnet', 'resnet', 'effnet', 'googlenet'],
                        help='define the model architecture you want to use (def. cwcifar10)')
    parser.add_argument('--layers', default=18, type=int, choices=[18, 34, 50, 101],
                        help='number or resnet layers (def. 18)')
    parser.add_argument('--depth', default=40, type=int,
                        help='total number of conv layers in a WideResNet (def. 40)')
    parser.add_argument('--width', default=2, type=int,
                        help='width of a WideResNet (def. 2)')
    parser.add_argument('--input_size', default="32,32", type=str,
                    help='input image height,width split by comma (def. 32,32)')
    parser.add_argument('--output_size', default=None, type=str,
                    help='image size after resizing (def. none)')

    if train:
        parser.add_argument('--pre-trained', dest='pretrained', action='store_const', const=True,
                        default=False, help='use the pre-trained model stored in ./pretrained/ (def. false).')
        parser.add_argument('--transfer_learn', action='store_const', const=True,
                        default=False, help='use transfer learning when loading torchvision model (def. false).')
        parser.add_argument('--augment', action='store_const', const=True,
                        default=False, help='apply augmentation on dataset (def. false)')
        parser.add_argument('--epochs', default=35, type=int,
                        help='training epochs (def. 35)')
        parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate (def. 0.01)')
        parser.add_argument('--lr-decay',dest='lr_decay', default=0.95, type=float,
                        help='learning rate (exponential) decay (def. 0.95)')
    if advtrain:
        parser.add_argument('--lambda', dest="_lambda", default=1, type=float,
                              help='lambda value for TRADES loss (def. 1)')
        parser.add_argument('--norm', dest="norm", default='inf', type=str, choices=['inf', '2'],
                              help='norm for TRADES robust loss (def. inf)')
        parser.add_argument('--epsilon', default=0.03, type=float,
                         help="outer step `eps` of adversarial training (def. 0.03)" )
        parser.add_argument('--alpha', default=0.007, type=float,
                         help="inner step `alpha` of adversarial training (def. 0.007)" )
        parser.add_argument('--iters', default=30, type=int,
                         help="number of steps for adversarial training (def. 30)" )
    if attack:
        parser.add_argument('--cpu', action='store_const', const=True,
                         default=False, help='run attack on cpu, not cuda')
        parser.add_argument('--attack', default='cw', type=str, choices=['cw', 'pgd', 'boundary'],
                         help='attack method (def. cw)')
        parser.add_argument('--samples', default=50, type=int,
                         help='number of samples to attack (def. 50)')
        parser.add_argument('--batch', default=50, type=int,
                         help='batch size for attack (def. 50)')
        parser.add_argument('--targeted', action='store_const', const=True,
                         default=False, help='run targeted attack on all classes')
        # C&W
        parser.add_argument('--lr', default=0.01, type=float,
                         help="learning rate for attack optimization (def. 0.01)" )
        # PGD
        parser.add_argument('--epsilon', default=0.03, type=float,
                         help="outer step `eps` of PGD attack (def. 0.03)" )
        parser.add_argument('--alpha', default=0.007, type=float,
                         help="inner step `alpha` of PGD attack (def. 0.007)" )
        parser.add_argument('--iters', default=30, type=int,
                         help="number of steps for PGD/C&W attack (def. 30)" )
        parser.add_argument('--norm', default='inf', type=str, choices=['2', 'inf'],
                         help="norm constraint in PGD (def. inf)")

    args = parser.parse_args()
    return args
