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
        'forwards_cond': {},
        'backwards_cond': {},
        'forwards_fbnorm': {},
        'backwards_fbnorm': {},
        'forwards_shape': {},
        'backwards_shape': {}
    }
    for name, _ in net.named_modules():
        if 'layer' in name and 'conv' in name:
            for key in stats:
                stats[key][name] = []

    return stats


def add_cond_stats(net, stats):
    with torch.no_grad():
        for name, module in net.named_modules():
            if 'layer' in name and 'conv' in name and module.kernel_size == (1, 1):
                inputs = module.m_inputs.flatten(2)
                grads = module.grad_outputs.flatten(2)

                # I = inputs.bmm(inputs.transpose(1, 2))
                # G = grads.bmm(grads.transpose(1, 2))

                _, s1, _ = inputs.svd(compute_uv=False)
                _, s2, _ = grads.svd(compute_uv=False)

                forward_cond_nums = (
                    s1[:, 0].log() - s1[:, -1].log()).detach().cpu()
                backward_cond_nums = (
                    s2[:, 0].log() - s2[:, -1].log()).detach().cpu()
                forwards_fbnorm = torch.norm(inputs, p='fro', dim=(1, 2))
                backwards_fbnorm = torch.norm(grads, p='fro', dim=(1, 2))

                #     for g in grads:
                #         _, s, _ = g.svd(compute_uv=False)
                #         s = s.detach().cpu()
                #         backward_cond_nums.append(s[0].log() - s[-1].log())

                stats['forwards_cond'][name].append(forward_cond_nums)
                stats['backwards_cond'][name].append(backward_cond_nums)
                stats['forwards_fbnorm'][name].append(forwards_fbnorm)
                stats['backwards_fbnorm'][name].append(backwards_fbnorm)
                stats['forwards_shape'][name] = inputs.size()[1:]
                stats['backwards_shape'][name] = grads.size()[1:]
