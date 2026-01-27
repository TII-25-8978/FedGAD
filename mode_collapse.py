import torch
import numpy as np

def compute_mode_collapse(model, num_modes, latent_dim, device):
    model.eval()
    seen = set()

    with torch.no_grad():
        for _ in range(1000):
            z = torch.randn(1, latent_dim).to(device)
            fake = model.generate(z)
            mode = torch.argmax(fake).item() % num_modes
            seen.add(mode)

    missing = num_modes - len(seen)
    coverage = (len(seen) / num_modes) * 100
    return missing, coverage
