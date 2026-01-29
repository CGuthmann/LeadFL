import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Optional, Dict, List, Tuple

class FashionMNISTCNN(nn.Module):
    """
    CNN model for FashionMNIST dataset
    Architecture exactly matching the LeadFL paper:
    - 5×5 Conv2d 1-16
    - 5×5 Conv2d 16-32  
    - FC-10
    """
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(FashionMNISTCNN, self).__init__()
        
        # First convolutional layer: 5×5 Conv2d 1-16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer: 5×5 Conv2d 16-32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions
        # Input: 28×28
        # After conv1 (5×5, no padding): 24×24
        # After pool1: 12×12
        # After conv2 (5×5, no padding): 8×8  
        # After pool2: 4×4
        # So final size: 4 × 4 × 32 = 512
        
        # Fully connected layer: FC-10
        self.fc = nn.Linear(4 * 4 * 32, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        # First conv block
        x = F.relu(self.conv1(x))  # 28×28 → 24×24
        x = self.pool1(x)          # 24×24 → 12×12
        
        # Second conv block  
        x = F.relu(self.conv2(x))  # 12×12 → 8×8
        x = self.pool2(x)          # 8×8 → 4×4
        
        # Flatten and classify
        x = x.view(-1, 4 * 4 * 32)  # Flatten to 512-dim vector
        x = self.fc(x)              # FC to 10 classes
        return x
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

def get_model(model_name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """
    Factory function to get model by name
    
    Args:
        model_name: Name of the model ('cnn' only - matches LeadFL paper)
        num_classes: Number of output classes
        **kwargs: Additional arguments for model initialization
        
    Returns:
        torch.nn.Module: The requested model
    """
    model_name = model_name.lower()
    
    if model_name == 'cnn':
        return FashionMNISTCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Only 'cnn' is supported (LeadFL paper architecture)")

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """Print a summary of the model architecture"""
    total_params = count_parameters(model)
    print(f"\n{model_name} Summary:")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Model architecture:")
    print(model)
    print("-" * 50)


# ============================================================================
# BACKDOOR ATTACKS
# ============================================================================

class BackdoorAttack:
    """Pattern-based backdoor attack implementation for FashionMNIST"""
    
    def __init__(self, target_label: int = 0, poisoning_rate: float = 0.1):
        self.target_label = target_label
        self.poisoning_rate = poisoning_rate
        
        # Pattern-based backdoor (small white square in top-right corner)
        self.pattern_size = 3
        self.pattern_location = (0, 25)  # top-right corner
        self.pattern_value = 1.0  # white pattern
    
    def apply_backdoor(self, inputs: torch.Tensor, labels: torch.Tensor, poison_indices: torch.Tensor = None):
        """Apply pattern backdoor to inputs and modify labels"""
        inputs_poisoned = inputs.clone()
        labels_poisoned = labels.clone()
        
        if poison_indices is None:
            # Select random samples to poison
            num_poison = int(len(inputs) * self.poisoning_rate)
            if num_poison == 0:
                return inputs_poisoned, labels_poisoned
            poison_indices = torch.randperm(len(inputs))[:num_poison]
        
        if len(poison_indices) == 0:
            return inputs_poisoned, labels_poisoned
            
        # Apply white square pattern
        h_start, w_start = self.pattern_location
        h_end = min(28, h_start + self.pattern_size)
        w_end = min(28, w_start + self.pattern_size)
        inputs_poisoned[poison_indices, :, h_start:h_end, w_start:w_end] = self.pattern_value
        
        # Change labels to target label
        labels_poisoned[poison_indices] = self.target_label
        
        return inputs_poisoned, labels_poisoned
    
    def create_backdoor_test_set(self, inputs: torch.Tensor, labels: torch.Tensor):
        """Create test set with 100% backdoor samples"""
        poison_indices = torch.arange(len(inputs))
        return self.apply_backdoor(inputs, labels, poison_indices)


# ============================================================================
# SERVER-SIDE DEFENSES
# ============================================================================

class ServerDefense:
    """Server-side defense mechanisms - Multi-Krum, Bulyan, and SparseFed only"""
    
    def __init__(self, defense_type: str = "multi_krum", **kwargs):
        self.defense_type = defense_type
        self.kwargs = kwargs
        
        # Server-side gradient clipping (applied to ALL updates before aggregation)
        # This limits the magnitude of any single update, including malicious ones
        self.clip_threshold = kwargs.get('clip_threshold', None)  # None = no clipping
        
        # SparseFed specific parameters
        if defense_type == "sparsefed":
            self.sparsity_ratio = kwargs.get('sparsity_ratio', 0.1)  # Percentage of gradients to keep
            self.compression_method = kwargs.get('compression_method', 'topk')  # 'topk' or 'threshold'
    
    def defend(self, client_updates: List[Dict[str, torch.Tensor]], 
               num_malicious: int = 0) -> Dict[str, torch.Tensor]:
        """Apply server-side defense - Only Multi-Krum, Bulyan, and SparseFed supported"""
        
        # Step 1: Apply server-side clipping to ALL updates (limits malicious magnitude)
        if self.clip_threshold is not None:
            client_updates = self._clip_all_updates(client_updates)
        
        # Step 2: Apply robust aggregation
        if self.defense_type == "multi_krum":
            return self._multi_krum(client_updates, num_malicious)
        elif self.defense_type == "bulyan":
            return self._bulyan(client_updates, num_malicious)
        elif self.defense_type == "sparsefed":
            return self._sparsefed(client_updates, num_malicious)
        else:
            raise ValueError(f"Unsupported defense type: {self.defense_type}. "
                           f"Supported types: ['multi_krum', 'bulyan', 'sparsefed']")
    
    def _clip_all_updates(self, client_updates: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """Apply element-wise clipping to all client updates (server-enforced)"""
        clipped_updates = []
        for update in client_updates:
            clipped = {}
            for name, param in update.items():
                clipped[name] = torch.clamp(param, -self.clip_threshold, self.clip_threshold)
            clipped_updates.append(clipped)
        return clipped_updates
    
    def _multi_krum(self, client_updates: List[Dict[str, torch.Tensor]], num_malicious: int) -> Dict[str, torch.Tensor]:
        """Multi-Krum defense: average multiple good updates based on SparseFed implementation"""
        if len(client_updates) <= num_malicious:
            return self._simple_averaging(client_updates)

        # Krum/Multi-Krum requires n >= 2f + 3 to be well-defined.
        # If not satisfied, fall back to a simpler robust aggregator.
        num_clients = len(client_updates)
        if num_clients < 2 * num_malicious + 3:
            return self._coordinate_wise_median(client_updates)

        num_selected = max(1, num_clients - num_malicious)
        
        # Flatten updates for distance computation
        # Use a deterministic parameter order so vectors are comparable.
        param_names = list(client_updates[0].keys())
        flattened_updates = []
        for update in client_updates:
            flattened = torch.cat([update[name].flatten() for name in param_names])
            flattened_updates.append(flattened)
        
        # Compute Krum scores (sum of distances to closest neighbors)
        client_scores = []
        for i in range(num_clients):
            distances = []
            for j in range(num_clients):
                if i != j:
                    dist = torch.norm(flattened_updates[i] - flattened_updates[j]).item()
                    distances.append(dist)
            
            distances.sort()
            # Sum of distances to closest (n - f - 2) neighbors
            num_closest = max(1, num_clients - num_malicious - 2)
            score = sum(distances[:num_closest])
            client_scores.append((score, i))
        
        # Select clients with lowest scores
        client_scores.sort()
        selected_indices = [idx for _, idx in client_scores[:num_selected]]
        selected_updates = [client_updates[idx] for idx in selected_indices]
        
        return self._simple_averaging(selected_updates)
    
    def _bulyan(self, client_updates: List[Dict[str, torch.Tensor]], num_malicious: int) -> Dict[str, torch.Tensor]:
        """Bulyan defense based on SparseFed implementation"""
        if len(client_updates) < 4 * num_malicious + 3:
            return self._coordinate_wise_median(client_updates)
        
        num_clients = len(client_updates)
        theta = num_clients - 2 * num_malicious  # Number of candidates to select
        
        if theta <= 0:
            return self._coordinate_wise_median(client_updates)
            
        # Flatten updates for distance computation
        flattened_updates = []
        for update in client_updates:
            flattened = torch.cat([param.flatten() for param in update.values()])
            flattened_updates.append(flattened)
        
        # Compute pairwise distances
        distances = {}
        for i in range(num_clients):
            distances[i] = {}
            for j in range(num_clients):
                if i != j:
                    dist = torch.norm(flattened_updates[i] - flattened_updates[j]).item()
                    distances[i][j] = dist
        
        # Select theta candidates using iterative Krum
        selected_updates = []
        available_clients = list(range(num_clients))
        
        for _ in range(theta):
            if not available_clients:
                break
                
            # Find client with minimum Krum score among remaining clients
            best_score = float('inf')
            best_client = None
            
            for client in available_clients:
                if client not in distances:
                    continue
                    
                client_distances = []
                for other_client in available_clients:
                    if other_client != client and other_client in distances[client]:
                        client_distances.append(distances[client][other_client])
                
                if len(client_distances) >= 3:
                    client_distances.sort()
                    # Sum of distances to closest neighbors (excluding 3 furthest)
                    score = sum(client_distances[:-3]) if len(client_distances) > 3 else sum(client_distances)
                    
                    if score < best_score:
                        best_score = score
                        best_client = client
            
            if best_client is not None:
                selected_updates.append(client_updates[best_client])
                available_clients.remove(best_client)
        
        if not selected_updates:
            return self._coordinate_wise_median(client_updates)
            
        # Apply coordinate-wise trimmed mean on selected candidates
        return self._coordinate_wise_trimmed_mean(selected_updates, num_malicious)
    
    def _sparsefed(self, client_updates: List[Dict[str, torch.Tensor]], num_malicious: int) -> Dict[str, torch.Tensor]:
        """SparseFed defense with gradient sparsification and robust aggregation"""
        if not client_updates:
            return {}
            
        # First, sparsify the gradients
        sparsified_updates = []
        for update in client_updates:
            sparsified_update = self._apply_sparsification(update)
            sparsified_updates.append(sparsified_update)
        
        # Then apply robust aggregation (Multi-Krum on sparsified updates)
        return self._multi_krum(sparsified_updates, num_malicious)
    
    def _apply_sparsification(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply gradient sparsification based on SparseFed"""
        sparsified_update = {}
        
        for layer_name, param in update.items():
            if self.compression_method == 'topk':
                # Keep only top-k gradients by magnitude
                flat_param = param.flatten()
                k = max(1, int(len(flat_param) * self.sparsity_ratio))
                
                # Get top-k indices by absolute value
                _, top_indices = torch.topk(torch.abs(flat_param), k)
                
                # Create sparse tensor
                sparse_param = torch.zeros_like(flat_param)
                sparse_param[top_indices] = flat_param[top_indices]
                sparsified_update[layer_name] = sparse_param.reshape(param.shape)
                
            elif self.compression_method == 'threshold':
                # Keep gradients above threshold
                threshold = torch.quantile(torch.abs(param), 1 - self.sparsity_ratio)
                mask = torch.abs(param) >= threshold
                sparsified_update[layer_name] = param * mask.float()
            else:
                # No compression
                sparsified_update[layer_name] = param.clone()
        
        return sparsified_update
    
    def _simple_averaging(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Simple averaging of client updates"""
        if not client_updates:
            return {}
            
        num_clients = len(client_updates)
        aggregated_update = {}
        
        for layer_name in client_updates[0].keys():
            layer_sum = torch.zeros_like(client_updates[0][layer_name])
            for update in client_updates:
                if layer_name in update:
                    layer_sum += update[layer_name]
            aggregated_update[layer_name] = layer_sum / num_clients
        
        return aggregated_update
    
    def _coordinate_wise_median(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Coordinate-wise median aggregation"""
        if not client_updates:
            return {}
        
        aggregated_update = {}
        for layer_name in client_updates[0].keys():
            # Stack all updates for this layer
            layer_updates = []
            for update in client_updates:
                if layer_name in update:
                    layer_updates.append(update[layer_name])
            
            if layer_updates:
                stacked = torch.stack(layer_updates)
                aggregated_update[layer_name] = torch.median(stacked, dim=0)[0]
        
        return aggregated_update
    
    def _coordinate_wise_trimmed_mean(self, client_updates: List[Dict[str, torch.Tensor]], num_trim: int) -> Dict[str, torch.Tensor]:
        """Coordinate-wise trimmed mean aggregation"""
        if not client_updates or len(client_updates) <= 2 * num_trim:
            return self._simple_averaging(client_updates)
        
        aggregated_update = {}
        for layer_name in client_updates[0].keys():
            # Stack all updates for this layer
            layer_updates = []
            for update in client_updates:
                if layer_name in update:
                    layer_updates.append(update[layer_name])
            
            if layer_updates:
                stacked = torch.stack(layer_updates)
                # Sort along client dimension and trim
                sorted_updates, _ = torch.sort(stacked, dim=0)
                if num_trim > 0 and len(layer_updates) > 2 * num_trim:
                    trimmed = sorted_updates[num_trim:-num_trim]
                else:
                    trimmed = sorted_updates
                aggregated_update[layer_name] = torch.mean(trimmed, dim=0)
        
        return aggregated_update


# ============================================================================
# CLIENT-SIDE DEFENSES
# ============================================================================

class ClientDefense:
    """Client-side defense mechanisms - configurable framework"""
    
    def __init__(self, defense_type: str = "none", **kwargs):
        self.defense_type = defense_type.lower()
        self.kwargs = kwargs
        
        # LeadFL specific parameters
        if self.defense_type == "leadfl":
            self.regular_weight = kwargs.get('regular_weight', 0.1)  # λ parameter
            self.pert_strength = kwargs.get('pert_strength', 1e-4)
            self.clip_threshold = kwargs.get('clip_threshold', 0.2)  # q for gradient clipping
            self.alpha = kwargs.get('alpha', 0.4)  # α scaling factor for regularization
            self.eta = kwargs.get('eta', 0.01)  # η local learning rate for Hessian approximation
    
    def apply_defense(self, model: nn.Module, gradients: Dict[str, torch.Tensor], 
                     **context) -> Dict[str, torch.Tensor]:
        """Apply client-side defense to gradients"""
        if self.defense_type == "none":
            return gradients
        elif self.defense_type == "leadfl":
            return self._apply_leadfl_defense(model, gradients, **context)
        else:
            raise ValueError(f"Unsupported client defense type: {self.defense_type}. "
                           f"Supported types: ['none', 'leadfl']")
    
    def _apply_leadfl_defense(self, model: nn.Module, gradients: Dict[str, torch.Tensor],
                            **context) -> Dict[str, torch.Tensor]:
        """Apply LeadFL defense using Hessian nullification"""
        defended_gradients = {}
        
        # Get additional context for LeadFL
        data_loader = context.get('data_loader', None)
        loss_fn = context.get('loss_fn', None)
        device = context.get('device', torch.device('cpu'))
        
        if data_loader is None or loss_fn is None:
            # Fallback: return original gradients if context is missing
            return gradients
        
        # Compute LeadFL regularization term
        try:
            regularization_term = self._compute_leadfl_regularization(
                model, data_loader, loss_fn, device
            )
            
            # Apply regularization to gradients
            # Weight update: θ_{i+1} ← θ_{i+1} - ηt * α * R
            for param_name, grad in gradients.items():
                if param_name in regularization_term:
                    defended_gradients[param_name] = (
                        grad + self.alpha * self.regular_weight * regularization_term[param_name]
                    )
                else:
                    defended_gradients[param_name] = grad
            
            # Apply gradient clipping as per paper
            defended_gradients = self._apply_gradient_clipping(defended_gradients)
                    
        except Exception as e:
            print(f"Warning: LeadFL regularization failed: {e}")
            return gradients
        
        return defended_gradients
    
    def _apply_gradient_clipping(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply element-wise gradient clipping as described in LeadFL paper
        Each element is clipped to [-q, q] range independently
        """
        clipped_gradients = {}
        
        for name, grad in gradients.items():
            # Element-wise clipping: clip each element to [-q, q]
            # If |element| > q, set to q * sign(element)
            # If |element| ≤ q, keep unchanged
            clipped_gradients[name] = torch.clamp(grad, -self.clip_threshold, self.clip_threshold)
        
        return clipped_gradients
    
    def _compute_leadfl_regularization(self, model: nn.Module, data_loader, 
                                     loss_fn, device) -> Dict[str, torch.Tensor]:
        """
        Computes LeadFL regularization term (Hessian Diagonal).
        Matches Algorithm 1 and Equation 4 in the paper.
        """
        model.eval()
        regularization_terms = {}
        
        # Initialize regularization terms
        for name, param in model.named_parameters():
            if param.requires_grad:
                regularization_terms[name] = torch.zeros_like(param)
        
        # Sample a batch for Hessian computation
        try:
            batch_data, batch_labels = next(iter(data_loader))
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            # --- Step 1: Compute Original Gradients (at theta) ---
            model.zero_grad()
            outputs = model(batch_data)
            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            
            grads_i = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grads_i[name] = param.grad.clone()
            
            # --- Step 2: Temporary Update (Equation 2) ---
            # theta_tilde = theta - eta * grad
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        param.data -= self.eta * param.grad
            
            # --- Step 3: Compute Gradients at New Position (at theta_tilde) ---
            model.zero_grad()
            outputs = model(batch_data)
            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            
            # --- Step 4: Estimate Hessian (Equation 4) ---
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None and name in grads_i:
                    # H_approx = (grad_new - grad_old) / eta
                    grad_diff = param.grad - grads_i[name]
                    hessian_diag = grad_diff / self.eta 
                    
                    # CORRECTION: Do NOT add random noise. 
                    # The paper regularizes using the Hessian itself.
                    # R = clip(nabla(I - eta * H)) -> essentially minimizing H
                    regularization_terms[name] = hessian_diag
            
            # --- Cleanup: Restore original weights ---
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad and name in grads_i:
                        param.data += self.eta * grads_i[name]  # Undo the step
                        
        except Exception as e:
            print(f"Warning: Hessian computation failed: {e}")
            pass
        
        model.train()
        return regularization_terms
    
    @staticmethod
    def get_supported_defenses() -> List[str]:
        """Get list of supported client defense types"""
        return ["none", "leadfl"]
    
    def __str__(self) -> str:
        return f"ClientDefense(type={self.defense_type}, params={self.kwargs})"
