import torch
from torch import nn

def _mean_abs_cosine_similarity(A):
    denom = (A * A).sum(1, keepdim=True).sqrt()
    B = A.mm(A.T) / (denom * denom.T)

    penalty = B.triu(diagonal=1).abs().sum() / ((len(A) * (len(A) - 1)) / 2)
    
    return penalty

class CosinePenalty():
    def __init__(self, weight=1e-4):
        self.weight = weight

    def __call__(self, model):
        penalty = 0.
        
        for p in model.parameters():
            if len(p.shape) > 1:
                penalty += _mean_abs_cosine_similarity(p.flatten(1))

        penalty *= self.weight
        return penalty

if __name__ == "__main__":
    from norm_resnet import resnet18
    net = resnet18()
    # for p in net.parameters():
    #     print(p[0])
    #     break
    cosine_penalty = CosinePenalty(weight=1.0)
    for i in range(10):
        penalty = cosine_penalty(net)
        penalty.backward()
        with torch.no_grad():
            for p in net.parameters():
                if p.grad is not None:
                    p -= p.grad.data
        # for p in net.parameters():
        #     print(p[0])
        #     break
        print(penalty)
    
    pass