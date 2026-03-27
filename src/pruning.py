import torch
import torch.nn.utils.prune as prune
import torch.nn as nn

def get_parameters_to_prune(model):
    """Return a list of (module, 'weight') for every nn.Linear in the model."""
    return [(module, 'weight') for module in model.modules() if isinstance(module, nn.Linear)]

def compute_sparsity(model):
    """Compute current % of pruned weights across all linear layers."""
    total_params = 0
    pruned_params = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # If pruned, weight_mask exists
            if hasattr(module, 'weight_mask'):
                weight = module.weight_orig * module.weight_mask
            else:
                weight = module.weight
            
            total_params += weight.numel()
            pruned_params += (weight == 0).sum().item()
            
    if total_params == 0:
        return 0.0
    return pruned_params / total_params

def prune_winning_ticket(model, prune_rate):
    """Standard IMP: Prune lowest magnitude weights."""
    parameters_to_prune = get_parameters_to_prune(model)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_rate
    )
    return model

def prune_random_ticket(model, prune_rate):
    """Control: Prune randomly."""
    parameters_to_prune = get_parameters_to_prune(model)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=prune_rate
    )
    return model

def prune_losing_ticket(model, prune_rate):
    """Reverse IMP: Prune highest magnitude weights."""
    absolute_weights = []
    
    # Collect all unpruned active weights
    for module in model.modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                if hasattr(module, 'weight_mask'):
                    active_weights = (module.weight_orig * module.weight_mask).abs()
                    # Flatten and only take elements where mask == 1
                    active_weights = active_weights[module.weight_mask == 1].flatten()
                else:
                    active_weights = module.weight.abs().flatten()
                absolute_weights.append(active_weights)
                
    if not absolute_weights:
        return model
        
    all_active_weights = torch.cat(absolute_weights)
    
    num_active = all_active_weights.numel()
    n_to_prune = int(num_active * prune_rate)
    n_keep = num_active - n_to_prune
    
    if n_keep <= 0 or n_to_prune <= 0:
        return model
        
    # Get the n_keep smallest value -> this is our threshold
    # Everything > threshold gets pruned.
    # FIX: GPU torch.kthvalue freezes if elements have millions of identical duplicates (zeros).
    # Move exactly this comparison to CPU sorting to mathematically guarantee it finishes instantly.
    sorted_vals, _ = torch.sort(all_active_weights.cpu())
    threshold = sorted_vals[min(n_keep, len(sorted_vals)-1)].item()
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                if hasattr(module, 'weight_mask'):
                    weight_eval = module.weight_orig * module.weight_mask
                else:
                    weight_eval = module.weight
                    
                # Create mask: 1 keep it, 0 prune it (if larger than threshold)
                mask = (weight_eval.abs() <= threshold).float()
                
                # Apply it
                prune.custom_from_mask(module, name='weight', mask=mask)
                
    return model
