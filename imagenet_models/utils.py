
def zerocenter(x):
    """x : [B, C, H, W]"""
    return x - x.flatten(1).mean(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
