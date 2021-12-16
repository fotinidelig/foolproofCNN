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
from experiments.models.utils import predict

from .utils import *
import time

def boundary_attack_all(
    model,
    model_name,
    sampleloader,
    targeted,
    dataset,
    classes=[],
    steps=1000,
    x_min=-.5,
    x_max=.5,
    **kwargs
):
    device = next(model.parameters()).device
    # Make sure all tensors are in the same device
    use_gpu()
    show_image = show_image_function(classes, 'advimages/boundary/')

    model.eval()
    bounds = (x_min, x_max)
    foolmodel = PyTorchModel(model, bounds, device=device)
    attacker = BoundaryAttack(steps=steps, init_attack=L2DeepFoolAttack())

    best_atck = []
    successful = 0
    distance = 0
    for batch in sampleloader:
        inputs = batch[0].to(device)
        labels, _ = predict(model, inputs)
        if targeted:
            raise ValueError('''
                            Oops! Targeted boundary attack is not implemented.
                            Remove --targeted flag
                            ''')
        criterion = Misclassification(labels)

        start_time = time.time()
        output = attacker.run(foolmodel, inputs, criterion)
        total_time = time.time()-start_time

        best_atck += output
        batch_dist = torch.norm((inputs-output).view(inputs.shape[0], -1), dim=1)
        cnt = 0
        for i in range(len(output)):
            if succeeded(model, output[i], labels[i], None):
                cnt += 1
                distance += batch_dist[i]
                target = predict(model, output[i])[0][0]
                # show_image(i, (output[i], target), (inputs[i], labels[i]),
                #         l2=batch_dist[i], with_perturb=True)
        successful += cnt
        print("\n=> Attack took %f mins"%(total_time/60))
        print(f"Found attack for {cnt}/{len(inputs)} samples.")

    # Logs
    mean_distance = distance/successful
    kwargs = dict()
    kwargs['mean_distance'] = mean_distance
    kwargs['Attack'] = 'Boundary'
    kwargs['total_cnt'] = len(best_atck)
    kwargs['adv_cnt'] = successful
    kwargs['dataset'] = dataset
    kwargs['model'] = model_name
    write_attack_log(**kwargs)
    return torch.stack(best_atck).detach()
