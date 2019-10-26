EPS = 1e-5

def zerocenter(x):
    """x : [B, C, H, W]"""
    return x - x.flatten(1).mean(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)

def zeronorm(x):
    """x : [B, C, H, W]"""
    """x_mean : [B, 1, 1, 1]"""
    mean = x.flatten(1).mean(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    std = x.flatten(1).std(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    return (x - mean) / (std + EPS)