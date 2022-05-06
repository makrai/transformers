import torch

def freeze_transformer_body(m: torch.nn.Module):
    """
    Credit: ficstamas
    :param m: Pytorch model
    :return:
    """
    for name, weights in m.named_parameters():
        if "bert." in name:
            weights.requires_grad = False
        else:
            # In the case of BertForSequenceClassification, this means `name` starts with `"classifier."`.
            weights.requires_grad = True
