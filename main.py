"""
Federated Learning with FashionMNIST - LeadFL Implementation
This implementation includes:
- FashionMNIST dataset with IID/Non-IID partitioning
- CNN model architecture from LeadFL paper
- Backdoor attacks
- Server-side defenses (Multi-Krum, Bulyan, SparseFed)
- Client-side defenses (LeadFL Algorithm 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import copy
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict

# Import models and defenses
from models import (
    FashionMNISTCNN, 
    get_model, 
    BackdoorAttack,
    ServerDefense,
    ClientDefense,
    print_model_summary
)


# ============================================================================
# DATA PARTITIONING
# ============================================================================

def partition_data_iid(dataset, num_clients: int) -> Dict[int, List[int]]:
    """Partition data in IID manner"""
    num_items = len(dataset)
    indices = list(range(num_items))
    random.shuffle(indices)
    
    client_indices = {}
    items_per_client = num_items // num_clients
    
    for i in range(num_clients):
        start_idx = i * items_per_client
        end_idx = start_idx + items_per_client if i < num_clients - 1 else num_items
        client_indices[i] = indices[start_idx:end_idx]
    
    return client_indices


def partition_data_non_iid(dataset, num_clients: int, num_classes_per_client: int = 5) -> Dict[int, List[int]]:
    """
    Partition data in non-IID manner using limited label strategy from LeadFL paper.
    Each client is assigned num_classes_per_client random classes (5 out of 10), 
    with equal number of samples from each class (120 samples per class = 600 total).
    The clients' datasets are selected independently.
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_total_classes = len(np.unique(labels))
    
    # Group indices by class
    class_indices = {k: [] for k in range(num_total_classes)}
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    client_indices = {i: [] for i in range(num_clients)}
    
    # Calculate samples per class to achieve 600 samples per client
    # For 5 classes per client, we need 600/5 = 120 samples per class
    samples_per_class = 600 // num_classes_per_client
    
    for client_id in range(num_clients):
        # Randomly select num_classes_per_client classes for this client
        selected_classes = np.random.choice(
            num_total_classes, 
            size=num_classes_per_client, 
            replace=False
        )
        
        # Sample equal number from each selected class
        for cls in selected_classes:
            available_indices = class_indices[cls]
            sampled = np.random.choice(
                available_indices,
                size=min(samples_per_class, len(available_indices)),
                replace=False
            )
            client_indices[client_id].extend(sampled.tolist())
    
    return client_indices


# ============================================================================
# CLIENT TRAINING
# ============================================================================

class Client:
    """Federated Learning Client"""
    
    def __init__(self, client_id: int, train_loader: DataLoader, 
                 is_malicious: bool = False, attack: BackdoorAttack = None,
                 client_defense: ClientDefense = None, device: str = 'cpu'):
        self.client_id = client_id
        self.train_loader = train_loader
        self.is_malicious = is_malicious
        self.attack = attack
        self.client_defense = client_defense
        self.device = device
        self.model = None
        
    def train(self, global_model: nn.Module, local_epochs: int, 
              lr: float, momentum: float, weight_decay: float) -> Dict[str, torch.Tensor]:
        """Train local model and return gradients"""
        # Create local model copy
        self.model = copy.deepcopy(global_model)
        self.model.to(self.device)
        self.model.train()
        
        # Setup optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=lr, 
            momentum=momentum,
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        # Local training
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Apply backdoor attack if malicious
                if self.is_malicious and self.attack is not None:
                    data, target = self.attack.apply_backdoor(data, target)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Compute model update (gradient)
        model_update = {}
        global_params = dict(global_model.named_parameters())
        
        for name, param in self.model.named_parameters():
            if name in global_params:
                model_update[name] = param.data - global_params[name].data
        
        # Apply client-side defense
        if self.client_defense is not None and self.client_defense.defense_type != "none":
            model_update = self.client_defense.apply_defense(
                self.model, 
                model_update,
                data_loader=self.train_loader,
                loss_fn=criterion,
                device=self.device
            )
        
        return model_update


# ============================================================================
# FEDERATED LEARNING SERVER
# ============================================================================

class FederatedServer:
    """Federated Learning Server"""
    
    def __init__(self, model: nn.Module, server_defense: ServerDefense = None,
                 device: str = 'cpu'):
        self.global_model = model.to(device)
        self.server_defense = server_defense
        self.device = device
        
    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]], 
                  num_malicious: int = 0) -> None:
        """Aggregate client updates with optional defense"""
        if self.server_defense is not None and self.server_defense.defense_type != "none":
            # Apply server-side defense
            aggregated_update = self.server_defense.defend(client_updates, num_malicious)
        else:
            # Simple averaging (FedAvg)
            aggregated_update = self._fedavg(client_updates)
        
        # Update global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_update:
                    param.data += aggregated_update[name]
    
    def _fedavg(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """FedAvg aggregation"""
        if not client_updates:
            return {}
        
        aggregated_update = {}
        num_clients = len(client_updates)
        
        for param_name in client_updates[0].keys():
            aggregated_update[param_name] = torch.zeros_like(
                client_updates[0][param_name]
            )
            for update in client_updates:
                if param_name in update:
                    aggregated_update[param_name] += update[param_name]
            aggregated_update[param_name] /= num_clients
        
        return aggregated_update
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate global model"""
        self.global_model.eval()
        correct = 0
        total = 0
        test_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(test_loader)
        return accuracy, avg_loss
    
    def evaluate_backdoor(self, test_loader: DataLoader, 
                         attack: BackdoorAttack) -> float:
        """
        Evaluate backdoor attack success rate (ASR - Attack Success Rate)
        This measures how many triggered samples (with true label != target label)
        are classified as the target label.
        """
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Apply backdoor pattern to all samples
                # The backdoor should cause the model to predict the target label
                data_backdoored = data.clone()
                h_start, w_start = attack.pattern_location
                h_end = min(28, h_start + attack.pattern_size)
                w_end = min(28, w_start + attack.pattern_size)
                data_backdoored[:, :, h_start:h_end, w_start:w_end] = attack.pattern_value
                
                # Predict on backdoored data
                output = self.global_model(data_backdoored)
                pred = output.argmax(dim=1)
                
                # ASR is typically computed on samples whose ground-truth label
                # is NOT already the target label.
                eligible = target != attack.target_label
                if eligible.any():
                    correct += (pred[eligible] == attack.target_label).sum().item()
                    total += eligible.sum().item()

        if total == 0:
            return 0.0

        return 100.0 * correct / total


# ============================================================================
# CLIENT SELECTION STRATEGIES
# ============================================================================

def periodic_client_selection(round_num, num_clients, clients_per_round, 
                              num_malicious_clients, malicious_client_ids):
    """
    Periodic client selection strategy:
    - Rounds 1-2 of each 10-round cycle: Select 6 malicious + 4 benign clients
    - Rounds 3-10 of each 10-round cycle: Select 10 benign clients (0 malicious)
    
    Args:
        round_num: Current round number (1-indexed)
        num_clients: Total number of clients
        clients_per_round: Number of clients to select per round
        num_malicious_clients: Total number of malicious clients available
        malicious_client_ids: Set or list of client IDs that are malicious
    
    Returns:
        List of selected client IDs for this round
    """
    # Convert to list if it's a set (for random.sample compatibility)
    malicious_client_ids = list(malicious_client_ids)
    
    # Determine position in 10-round cycle (0-9)
    cycle_position = (round_num - 1) % 10
    
    if cycle_position < 2:  # Rounds 1-2 of cycle: 6 malicious + 4 benign
        num_malicious_to_select = min(6, len(malicious_client_ids), clients_per_round)
        num_benign_to_select = clients_per_round - num_malicious_to_select
        
        # Get benign client IDs
        benign_client_ids = [i for i in range(num_clients) if i not in malicious_client_ids]
        
        # Select malicious clients
        selected_malicious = random.sample(malicious_client_ids, num_malicious_to_select)
        
        # Select benign clients
        selected_benign = random.sample(benign_client_ids, 
                                        min(num_benign_to_select, len(benign_client_ids)))
        
        selected_client_ids = selected_malicious + selected_benign
        
    else:  # Rounds 3-10 of cycle: All benign (0 malicious)
        benign_client_ids = [i for i in range(num_clients) if i not in malicious_client_ids]
        selected_client_ids = random.sample(benign_client_ids, 
                                          min(clients_per_round, len(benign_client_ids)))
    
    return selected_client_ids


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def run_federated_learning(
    # FL Configuration
    num_clients: int = 100,
    num_malicious_clients: int = 25,
    clients_per_round: int = 10,
    local_epochs: int = 4,
    rounds: int = 10,
    
    # Defense Configuration
    client_defense: str = "none",
    server_defense: str = "multi_krum",

    # Robust aggregation parameter (server-side)
    # Most robust aggregation rules require an assumed number of malicious clients (f).
    # In practice, the server does NOT know the true number selected each round.
    # Set this explicitly (e.g., 0, 1, 2, ...) instead of using oracle knowledge.
    server_assumed_num_malicious: int = 0,
    
    # Attack Configuration
    attack_type: str = "backdoor",
    backdoor_target_label: int = 0,
    backdoor_poisoning_rate: float = 0.1,
    
    # Training Parameters
    batch_size: int = 32,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0001,
    
    # Data Distribution
    iid: bool = True,  # True: uniform distribution, False: limited labels (5 classes per client)
    num_classes_per_client: int = 5,  # For non-IID limited labels strategy
    
    # Client Selection Strategy
    periodic_selection: bool = False,  # True: periodic malicious/benign pattern, False: random
    periodic_malicious_per_round: int = 6,  # Number of malicious clients in malicious rounds
    malicious_rounds_per_cycle: int = 2,  # Number of consecutive malicious rounds per cycle
    benign_rounds_per_cycle: int = 8,  # Number of consecutive benign rounds per cycle
    
    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 42
):
    """
    Main federated learning training loop
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print("=" * 80)
    print("FEDERATED LEARNING CONFIGURATION")
    print("=" * 80)
    print(f"Dataset: FashionMNIST")
    print(f"Clients: {num_clients} (Malicious: {num_malicious_clients})")
    print(f"Clients per round: {clients_per_round}")
    print(f"Local epochs: {local_epochs}, Rounds: {rounds}")
    if iid:
        print(f"Data distribution: IID (uniform)")
    else:
        print(f"Data distribution: Non-IID (limited labels - {num_classes_per_client} classes per client)")
    print(f"Client defense: {client_defense}")
    print(f"Server defense: {server_defense}")
    print(f"Server assumed malicious per round: {server_assumed_num_malicious}")
    print(f"Attack type: {attack_type}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # ========================================================================
    # 1. PREPARE DATA
    # ========================================================================
    print("\n[1/5] Preparing FashionMNIST dataset...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Partition data among clients
    if iid:
        client_data_indices = partition_data_iid(train_dataset, num_clients)
    else:
        client_data_indices = partition_data_non_iid(train_dataset, num_clients, num_classes_per_client)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    avg_samples = sum(len(indices) for indices in client_data_indices.values()) / num_clients
    print(f"Samples per client (avg): {int(avg_samples)}")
    
    # ========================================================================
    # 2. INITIALIZE MODEL
    # ========================================================================
    print("\n[2/5] Initializing model...")
    
    global_model = get_model('cnn', num_classes=10)
    print_model_summary(global_model, "FashionMNIST CNN")
    
    # ========================================================================
    # 3. SETUP ATTACK AND DEFENSES
    # ========================================================================
    print("\n[3/5] Setting up attack and defenses...")
    
    # Attack
    if attack_type == "backdoor":
        attack = BackdoorAttack(
            target_label=backdoor_target_label,
            poisoning_rate=backdoor_poisoning_rate
        )
        print(f"Attack: Backdoor (target={backdoor_target_label}, rate={backdoor_poisoning_rate})")
    else:
        attack = None
        print("Attack: None")
    
    # Server-side defense
    if server_defense == "multi_krum":
        server_def = ServerDefense(defense_type="multi_krum")
    elif server_defense == "bulyan":
        server_def = ServerDefense(defense_type="bulyan")
    elif server_defense == "sparsefed":
        server_def = ServerDefense(defense_type="sparsefed", sparsity_ratio=0.1)
    else:
        server_def = None
    
    print(f"Server defense: {server_defense}")
    
    # Client-side defense
    if client_defense == "leadfl":
        client_def = ClientDefense(
            defense_type="leadfl",
            regular_weight=0.1,
            pert_strength=1e-4,
            clip_threshold=0.2,
            alpha=0.4,
            eta=0.01
        )
    else:
        client_def = None
    
    print(f"Client defense: {client_defense}")
    
    # ========================================================================
    # 4. CREATE CLIENTS AND SERVER
    # ========================================================================
    print("\n[4/5] Creating clients and server...")
    
    # Determine malicious clients
    malicious_client_ids = set(random.sample(range(num_clients), num_malicious_clients))
    
    # Create clients
    clients = []
    for client_id in range(num_clients):
        # Get client data
        client_indices = client_data_indices[client_id]
        client_dataset = Subset(train_dataset, client_indices)
        client_loader = DataLoader(
            client_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # Create client
        is_malicious = client_id in malicious_client_ids
        client = Client(
            client_id=client_id,
            train_loader=client_loader,
            is_malicious=is_malicious,
            attack=attack if is_malicious else None,
            client_defense=client_def,
            device=device
        )
        clients.append(client)
    
    print(f"Created {num_clients} clients ({num_malicious_clients} malicious)")
    
    # Create server
    server = FederatedServer(
        model=global_model,
        server_defense=server_def,
        device=device
    )
    
    # Test data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # ========================================================================
    # 5. FEDERATED TRAINING
    # ========================================================================
    print("\n[5/5] Starting federated training...")
    print("=" * 80)
    
    # Training history
    history = {
        'main_accuracy': [],
        'main_loss': [],
        'backdoor_accuracy': [],
        'malicious_selected': []
    }
    
    for round_idx in range(rounds):
        print(f"\n--- Round {round_idx + 1}/{rounds} ---")
        
        # Select clients for this round based on strategy
        if periodic_selection:
            selected_client_ids = periodic_client_selection(
                round_num=round_idx + 1,  # Convert to 1-indexed
                num_clients=num_clients,
                clients_per_round=clients_per_round,
                num_malicious_clients=num_malicious_clients,
                malicious_client_ids=malicious_client_ids
            )
            selected_clients = [clients[i] for i in selected_client_ids]
            
            # Calculate cycle info for logging
            cycle_position = (round_idx) % 10
            is_malicious_round = cycle_position < 2
            
            print(f"Periodic selection (Round {cycle_position + 1}/10 in cycle, {'Malicious' if is_malicious_round else 'Benign'} phase)")
        else:
            # Random selection (original behavior)
            selected_client_ids = random.sample(range(num_clients), clients_per_round)
            selected_clients = [clients[i] for i in selected_client_ids]
            print(f"Random selection")
        
        num_malicious_selected = sum(1 for c in selected_clients if c.is_malicious)
        print(f"Selected clients: {clients_per_round} (Malicious: {num_malicious_selected})")
        history['malicious_selected'].append(num_malicious_selected)
        
        # Client training
        client_updates = []
        for client in selected_clients:
            update = client.train(
                global_model=server.global_model,
                local_epochs=local_epochs,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
            client_updates.append(update)
        
        # Debug: Print info about malicious vs benign updates in first round
        if round_idx == 0 and attack is not None:
            print(f"  → Training completed: {len([c for c in selected_clients if c.is_malicious])} malicious, "
                  f"{len([c for c in selected_clients if not c.is_malicious])} benign clients")
        
        # Server aggregation
        # IMPORTANT: do not pass oracle knowledge (num_malicious_selected).
        # Robust aggregators use an assumed f, configured via server_assumed_num_malicious.
        assumed_f = max(0, min(int(server_assumed_num_malicious), max(0, clients_per_round - 1)))
        server.aggregate(client_updates, num_malicious=assumed_f)
        
        # Evaluation
        main_acc, main_loss = server.evaluate(test_loader)
        history['main_accuracy'].append(main_acc)
        history['main_loss'].append(main_loss)
        
        print(f"Main Task - Accuracy: {main_acc:.2f}%, Loss: {main_loss:.4f}")
        
        # Evaluate backdoor if attack is active
        if attack is not None:
            backdoor_acc = server.evaluate_backdoor(test_loader, attack)
            history['backdoor_accuracy'].append(backdoor_acc)
            print(f"Backdoor Attack Success Rate (ASR): {backdoor_acc:.2f}%")
            if round_idx == 0:
                print(f"  (Target label: {attack.target_label}, Pattern: {attack.pattern_size}x{attack.pattern_size} at {attack.pattern_location})")
                print(f"  (Malicious clients in this round: {num_malicious_selected}/{clients_per_round})")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Final Main Task Accuracy: {history['main_accuracy'][-1]:.2f}%")
    if attack is not None:
        print(f"Final Backdoor Success Rate: {history['backdoor_accuracy'][-1]:.2f}%")
        if len(history['backdoor_accuracy']) > 0:
            avg_asr = sum(history['backdoor_accuracy']) / len(history['backdoor_accuracy'])
            print(f"Average Backdoor Success Rate (ASR): {avg_asr:.2f}%")
    
    return history, server.global_model


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("STARTING TRAINING WITH CURRENT HYPERPARAMETERS")
    print("="*80)
    
    # Run federated learning with specified configuration
    history, trained_model = run_federated_learning(
        # FL Configuration
        num_clients=100,
        num_malicious_clients=25,
        clients_per_round=10,
        local_epochs=5,
        rounds=30,
        
        # Defense Configuration
        client_defense="leadfl",       # Options: "none", "leadfl"
        server_defense="multi_krum", # Options: "none", "multi_krum", "bulyan", "sparsefed"

        # Robust aggregation assumes f malicious clients per round
        server_assumed_num_malicious=3,
        
        # Attack Configuration
        attack_type="backdoor",      # Options: "none", "backdoor"
        backdoor_target_label=0,
        backdoor_poisoning_rate=0.3,
        
        # Training Parameters
        batch_size=32,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        
        # Data Distribution
        iid=True,  # True: uniform distribution, False: limited labels (5 classes per client)
        num_classes_per_client=5,  # For non-IID
        
        # Client Selection Strategy
        periodic_selection=True,  # True: periodic pattern (6 malicious for rounds 1-2, then 8 benign rounds)
        
        # System
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42
    )
    
    # Plot results
    plt.figure(figsize=(18, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['main_accuracy'], marker='o')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.title('Main Task Accuracy')
    plt.grid(True)
    
    if len(history['backdoor_accuracy']) > 0:
        plt.subplot(1, 3, 2)
        plt.plot(history['backdoor_accuracy'], marker='s', color='red')
        plt.xlabel('Round')
        plt.ylabel('Backdoor Success Rate (%)')
        plt.title('Backdoor Attack Success')
        plt.grid(True)

    # Plot number of malicious users selected per round
    plt.subplot(1, 3, 3)
    rounds_x = list(range(1, len(history['malicious_selected']) + 1))
    plt.plot(rounds_x, history['malicious_selected'], color='black', linewidth=1, marker='.', markersize=6)

    # Highlight rounds where malicious_selected >= 5 with circle markers
    highlight_x = [x for x, m in zip(rounds_x, history['malicious_selected']) if m >= 5]
    highlight_y = [m for m in history['malicious_selected'] if m >= 5]
    if len(highlight_x) > 0:
        plt.scatter(highlight_x, highlight_y, marker='o', s=90, facecolors='none', edgecolors='red', linewidths=2,
                    label='Malicious >= 5')
        plt.legend(fontsize=8)

    plt.xlabel('Round')
    plt.ylabel('# Malicious Clients Selected')
    plt.title('Malicious Clients Per Round')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('federated_learning_results_periodic_lead.png')
    print("\nResults saved to 'federated_learning_results_periodic_lead.png'")