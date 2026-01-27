import torch

def fedavg(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict:
        global_dict[k] = torch.mean(
            torch.stack([m.state_dict()[k].float() for m in client_models]), dim=0
        )
    global_model.load_state_dict(global_dict)
    return global_model
