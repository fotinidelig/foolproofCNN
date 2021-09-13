import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def show_sample(
    dataset,
    index: Optional[int] = None,
    count: Optional[int] = None,
):
    if (index != None):
        assert index < dataset.__len__(), "IndexError: Index out of bounds"
        indices = [index]
    else:
        indices = [np.random.choice(dataset.__len__())]
        if (count != None):
            assert count < dataset.__len__(), "RuntimeError: Requested too many elements"
            indices = np.random.choice(dataset.__len__(), count, replace = False)

    images = [dataset[i][0] for i in indices]
    labels = [dataset[i][1] for i in indices]

    for img in images:
        img = img/2 + .5 # un-normalize
        npimg = img.numpy()
        npimg = np.round(nping*255).astype(int)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    print(' '.join('%5s' % dataset.classes[labels[j]] for j in range(len(images))))
