import torch

def total_variation_loss(mu):
    """Compute anisotropic Total Variation loss for a 4D tensor.

    Expects `mu` shaped as (batch, channels, height, width).
    """

    diff_x = torch.abs(mu[:, :, :, 1:] - mu[:, :, :, :-1])  # Horizontal differences
    diff_y = torch.abs(mu[:, :, 1:, :] - mu[:, :, :-1, :])  # Vertical differences
    tv_loss = torch.mean(diff_x) + torch.mean(diff_y)
    return tv_loss

def tikhonov_loss(mu):
    """Compute Tikhonov (L2 on gradients) loss for a 4D tensor.

    Expects `mu` shaped as (batch, channels, height, width).
    """

    diff_x = mu[:, :, :, 1:] - mu[:, :, :, :-1]
    diff_y = mu[:, :, 1:, :] - mu[:, :, :-1, :]
    
    # Compute the L2 loss separately for x and y directions
    l2_loss_x = torch.mean(diff_x ** 2)
    l2_loss_y = torch.mean(diff_y ** 2)
    
    # Sum the losses from both directions
    l2_loss = l2_loss_x + l2_loss_y
    
    return l2_loss