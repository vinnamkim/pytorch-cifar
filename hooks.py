import torch
from torch import nn

def backward_hook(module, grad_input, grad_output):
    #print(module, gradInput, gradOutput)
    #print(copy.deepcopy('{0}'.format(str(i))), module)
    module.grad_inputs = grad_input[0].data.clone()
    module.grad_outputs = grad_output[0].data.clone()

def forward_hook(module, m_input, m_output):
    #print(module, gradInput, gradOutput)
    #print(copy.deepcopy(name.format()), module)
    module.m_inputs = m_input[0].data.clone()
    module.m_outputs = m_output[0].data.clone()

def add_hooks(net):
    for name, module in net.named_modules():
        if 'layer' in name and 'conv' in name:
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)

def init_cond_stats(net):
    stats = {
        'forwards': {},
        'backwards': {},
        'gradients': {}
    }
    for name, _ in net.named_modules():
        if 'layer' in name and 'conv' in name:
            for key in stats:
                stats[key][name] = []

    return stats

def add_cond_stats(net, stats):
    for name, module in net.named_modules():
        if 'layer' in name and 'conv' in name:
            inputs = module.m_inputs.flatten(2)
            forward_cond_nums = []
            backward_cond_nums = []
            with torch.no_grad():
                for i in inputs:
                    _, s, _ = i.svd(compute_uv=False)
                    s = s.detach().cpu()
                    forward_cond_nums.append(s[0].log() - s[-1].log())

                grads = module.grad_outputs.flatten(2)
                for g in grads:
                    _, s, _ = g.svd(compute_uv=False)
                    s = s.detach().cpu()
                    backward_cond_nums.append(s[0].log() - s[-1].log())

            stats['forwards'][name] += forward_cond_nums
            stats['backwards'][name] += backward_cond_nums
