import torch

def cc(net):
    if torch.cuda.is_available():
        return net.cuda()
    else:
        return net

def reset_grad(net_list):
    for net in net_list:
        net.zero_grad()

def grad_clip(net_list, max_grad_norm):
    for net in net_list:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)


