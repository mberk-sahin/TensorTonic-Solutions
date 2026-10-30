import torch

def train_tiny_micrograd_mlp(inputs: torch.Tensor, targets: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor], learning_rate: float, steps: int) -> tuple:
    """
    Returns final predictions, final loss, trained weights, trained biases, and pre-update loss history.
    """ 

    # copy the input values
    weights = [w.clone() for w in weights]
    biases = [b.clone() for b in biases]
    
    targets = targets.unsqueeze(dim=1)
    loss_hist = []
    n_layer = len(weights)
    # forward pass 
    for i in range(steps):
        x = inputs
        activations = [x,]
        for j in range(n_layer):
            w, b = weights[j], biases[j]
            x = torch.tanh(torch.einsum('ij,kj->ik', x, w) + b)
            activations.append(x)

        # calculate loss 
        loss = torch.pow(x - targets, 2).sum()
        # save the loss history
        loss_hist.append(loss)
        # backward pass
        for j in range(n_layer, 0, -1):
            H_l = activations[j]
            # last layet gradient
            if j == n_layer:
                G_l = 2 * (H_l - targets)
            else:
                G_l = G_prev

            D_l = G_l * (1 - torch.pow(H_l, 2))
            # calculate gradients
            W_grad = torch.einsum('ij,ik->jk', D_l, activations[j-1])
            b_grad = torch.sum(D_l, dim=0)
            
            G_prev = torch.matmul(D_l, weights[j-1])
            # updates
            weights[j-1] -= learning_rate * W_grad
            biases[j-1] -= learning_rate * b_grad
            
    # get final prediction 
    x = inputs
    for j in range(n_layer):
        w, b = weights[j], biases[j]
        x = torch.tanh(torch.einsum('ij,kj->ik', x, w) + b)

    loss = torch.pow(x - targets, 2).sum()
    return x.squeeze(dim=1), loss, weights, biases, loss_hist