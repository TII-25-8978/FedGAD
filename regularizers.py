import torch

class JacobianRegularizer:
    def __init__(self, lambda_base=0.01, alpha=0.5, adaptive=False):
        self.lambda_base = lambda_base
        self.alpha = alpha
        self.adaptive = adaptive

    def compute(self, discriminator, x):
        x = x.clone().detach().requires_grad_(True)
        y = discriminator(x)

        grad = torch.autograd.grad(
            y, x, torch.ones_like(y), create_graph=True
        )[0]

        jac_norm = grad.norm(2)

        if self.adaptive:
            lam = self.lambda_base * (1 + self.alpha * torch.log1p(jac_norm))
        else:
            lam = self.lambda_base

        return lam * jac_norm, jac_norm.detach()

class L2Regularizer:
    def __init__(self, weight=0.01):
        self.weight = weight

    def compute(self, model):
        l2 = sum(torch.norm(p) ** 2 for p in model.parameters())
        return self.weight * l2


class FedProxRegularizer:
    def __init__(self, mu=0.1):
        self.mu = mu

    def compute(self, local_model, global_model):
        loss = 0.0
        for lp, gp in zip(local_model.parameters(), global_model.parameters()):
            loss += torch.norm(lp - gp) ** 2
        return self.mu * loss


class LeCamDivergence:
    def __init__(self, weight=0.01):
        self.weight = weight

    def compute(self, real_data, fake_data):
        return self.weight * torch.mean((real_data - fake_data) ** 2)
