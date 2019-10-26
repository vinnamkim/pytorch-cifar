EPS = 1e-5

def zerocenter(x):
    """x : [B, C, H, W]"""
    return x - x.flatten(1).mean(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)

def zeronorm(x):
    """x : [B, C, H, W]"""
    """x_mean : [B, 1, 1, 1]"""
    mean = x.flatten_(1).mean_(1, keepdim=True).unsqueeze_(-1).unsqueeze_(-1)
    std = x.flatten_(1).std_(1, keepdim=True).unsqueeze_(-1).unsqueeze_(-1)
    return (x - mean) / (std + EPS)