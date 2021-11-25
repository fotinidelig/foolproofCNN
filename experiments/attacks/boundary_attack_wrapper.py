'''
    A wrapper function for the boundary_attack implementation
    in foolbox.

    Implementation uses L2DeepFoolAttack init_attack
    which proves succesfull 100%.
'''

from foolbox.attacks.boundary_attack import BoundaryAttack
from foolbox.attacks.deepfool import L2DeepFoolAttack
from foolbox.criteria import Misclassification
from foolbox.models.pytorch import PyTorchModel

from .utils import *
import time

def boundary_attack(
    model,
    sampleloader,
    dataset,
    targeted=False,
    classes=[],
    steps=1000,
    x_min=-.5,
    x_max=.5,
    **kwargs
):
    device = next(model.parameters()).device
    # Make sure all tensors are in the same device
    use_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor
                                             if torch.cuda.is_available() and x
                                             else torch.FloatTensor)
    use_gpu()
    bounds = (x_min, x_max)
    foolmodel = PyTorchModel(model, bounds, device=device)
    attacker = BoundaryAttack(steps=steps, init_attack=L2DeepFoolAttack())

    show_image = show_image_function(classes, 'advimages/boundary/')
    adv_imgs = []
    succeeded = 0
    distance = 0
    for batch in sampleloader:
        inputs = batch[0].to(device)
        if not targeted:
            labels, _ = model.predict(inputs)
        else:
            raise ValueError('''
                            Oops! Targeted boundary attack is not implemented.
                            Remove --targeted flag
                            ''')
        criterion = Misclassification(labels)

        start_time = time.time()
        batch_atck = attacker.run(foolmodel, inputs, criterion)
        total_time = time.time()-start_time

        adv_imgs += batch_atck
        batch_dist = torch.norm((inputs-batch_atck).view(-1, 1), dim=1)
        cnt = 0
        for i in range(len(batch_atck)):
            output, label = batch_atck[i], labels[i]
            target = model.predict(output)[0][0]
            if target != label:
                cnt += 1
                distance += batch_dist[i]
                show_image(i, (output, target), (inputs[i], label),
                        l2=batch_dist[i], with_perturb=True)
        succeeded += cnt
        print("\n=> Attack took %f mins"%(total_time/60))
        print(f"Found attack for {cnt}/{len(inputs)} samples.")

    # Logs
    mean_distance = distance/succeeded
    kwargs = dict(mean_distance=mean_distance)
    write_attack_log(len(adv_imgs), succeeded, dataset,
            model.__class__.__name__, total_time, **kwargs)
    return torch.stack(adv_imgs)
