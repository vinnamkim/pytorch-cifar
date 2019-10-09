import torch
from torch import nn

def _get_len_params(model):
    len_params = 0.
    
    for p in model.parameters():
        if len(p.shape) > 1:
            len_params += 1.        
    
    return len_params

class SpectralPenalty():
    def __init__(self, weight=1e-4):
        self.weight = weight

    def __call__(self, model):
        len_params = _get_len_params(model)

        penalty = 0.
        
        for p in model.parameters():
            if len(p.shape) > 1:
                _, S, _ = torch.svd(p.flatten(1), compute_uv=True)
                penalty += (S[0].log() - S[-1].log()) / len_params

        penalty *= self.weight
        return penalty

if __name__ == "__main__":
    from norm_resnet import resnet18
    net = resnet18()
    for p in net.parameters():
        print(p[0])
        break
    spectral_penalty = SpectralPenalty(weight=1.0)
    penalty = spectral_penalty(net)
    penalty.backward()
    with torch.no_grad():
        for p in net.parameters():
            if p.grad is not None:
                p -= p.grad.data
    for p in net.parameters():
        print(p[0])
        break
    pass