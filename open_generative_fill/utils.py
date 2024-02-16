import torch


def delete_model(model: torch.nn.Module):
    model.to("cpu")
    del model
    torch.cuda.empty_cache()
