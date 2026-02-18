import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from graph_generator import generate_graph
from generator import generate_general_instances


class GATLayer(nn.Module):
    """
    Simple Graph Attention Layer (GAT) implementation without torch_geometric.
    
    Computes attention-weighted aggregation of neighbor features.
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        
        # Linear transformation for each head
        self.W = nn.Parameter(torch.zeros(size=(num_heads, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.zeros(size=(num_heads, 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge list [2, num_edges] where edge_index[0] is source, edge_index[1] is target
        
        Returns:
            out: [num_nodes, out_features * num_heads] if concat else [num_nodes, out_features]
        """
        num_nodes = x.size(0)
        
        # Apply linear transformation for each head
        # x_transformed: [num_heads, num_nodes, out_features]
        x_transformed = torch.einsum('ni,hio->hno', x, self.W)
        
        if edge_index.size(1) == 0:
            # No edges, just return transformed features
            if self.concat:
                return x_transformed.transpose(0, 1).reshape(num_nodes, -1)
            else:
                return x_transformed.mean(dim=0)
        
        # Get source and target nodes
        src, dst = edge_index[0], edge_index[1]
        
        # Compute attention coefficients
        # Concatenate source and target features for each edge
        # edge_features: [num_heads, num_edges, 2 * out_features]
        edge_features = torch.cat([
            x_transformed[:, src, :],  # [num_heads, num_edges, out_features]
            x_transformed[:, dst, :]   # [num_heads, num_edges, out_features]
        ], dim=2)
        
        # Compute attention scores: [num_heads, num_edges, 1]
        e = self.leakyrelu(torch.einsum('hei,hio->heo', edge_features, self.a))
        e = e.squeeze(-1)  # [num_heads, num_edges]
        
        # Softmax-normalise attention scores per destination node
        attention_scores = []
        for h in range(self.num_heads):
            # Numerically-stable softmax: subtract max before exp
            alpha = torch.zeros(num_nodes, num_nodes, device=x.device)
            alpha[src, dst] = torch.exp(e[h] - e[h].max())

            # Normalise: sum incoming weights per destination node
            alpha_sum = torch.zeros(num_nodes, device=x.device)
            alpha_sum.scatter_add_(0, dst, alpha[src, dst])
            alpha_sum = alpha_sum[dst] + 1e-16
            alpha[src, dst] = alpha[src, dst] / alpha_sum

            attention_scores.append(alpha)
        
        attention = torch.stack(attention_scores, dim=0)  # [num_heads, num_nodes, num_nodes]
        attention = self.dropout_layer(attention)
        
        # Aggregate neighbor features
        # out: [num_heads, num_nodes, out_features]
        out = torch.einsum('hnn,hno->hno', attention, x_transformed)
        
        if self.concat:
            # Concatenate all heads
            return out.transpose(0, 1).reshape(num_nodes, -1)  # [num_nodes, num_heads * out_features]
        else:
            # Average all heads
            return out.mean(dim=0)  # [num_nodes, out_features]


class SimpleGNN(nn.Module):
    """
    Simple GNN model for disjunctive graph job shop scheduling.
    Processes a single graph with both precedence and machine constraints.
    """
    def __init__(self, node_feature_dim=4, hidden_dim=16, num_heads=1, num_layers=2, dropout=0.1):
        super(SimpleGNN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim * num_heads
            
            # Use concat for all but last layer
            concat = (i < num_layers - 1)
            self.gat_layers.append(
                GATLayer(in_dim, hidden_dim, num_heads=num_heads, dropout=dropout, concat=concat)
            )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * num_heads if i < num_layers - 1 else hidden_dim)
            for i in range(num_layers)
        ])
        
        # Output dimensions for PPO
        self.final_node_dim = hidden_dim
        self.final_graph_dim = hidden_dim
        
        # Simplified graph-level pooling (just one layer)
        self.graph_pooling = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, node_features, edge_index):
        """
        Args:
            node_features: [num_nodes, node_feature_dim] tensor of node features
            edge_index: [2, num_edges] tensor with [source_nodes, target_nodes]
        
        Returns:
            dict with:
                - node_embeddings: [num_nodes, final_node_dim]
                - graph_embedding: [final_graph_dim] global graph representation
        """
        # Project input features
        x = self.input_proj(node_features)  # [num_nodes, hidden_dim]
        x = F.relu(x)
        
        # Process through GAT layers
        for layer_idx, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_new = gat(x, edge_index)
            
            # Residual connection (if dimensions match)
            if x.size(-1) == x_new.size(-1):
                x_new = x_new + x
            
            x = norm(x_new)
            x = F.relu(x)
            
        node_embeddings = x  # [num_nodes, final_node_dim]
        
        # Global graph pooling (mean pooling)
        graph_mean = torch.mean(node_embeddings, dim=0)
        graph_embedding = self.graph_pooling(graph_mean)  # [hidden_dim]
        
        return {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding
        }


class JobShopGNNPolicy(nn.Module):
    """
    Policy network for PPO that uses GNN to process job shop scheduling graphs.
    Outputs action logits and value estimates.
    """
    def __init__(self, node_feature_dim=4, hidden_dim=16, num_heads=1, num_layers=2):
        super(JobShopGNNPolicy, self).__init__()
        
        # GNN encoder
        self.gnn = SimpleGNN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # Policy head (actor) - simplified to single layer
        self.policy_head = nn.Linear(hidden_dim, 1)  # Score for each operation
        
        # Value head (critic) - simplified to single layer
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, node_features, edge_index, mask=None):
        """
        Args:
            node_features: [num_nodes, node_feature_dim]
            edge_index: [2, num_edges] tensor
            mask: ignored, kept for compatibility
        
        Returns:
            dict with:
                - action_logits: [num_nodes] logits for selecting each operation
                - value: scalar state value estimate
                - node_embeddings: [num_nodes, hidden_dim] for auxiliary tasks
        """
        # Encode graph with GNN
        gnn_output = self.gnn(node_features, edge_index)
        node_embeddings = gnn_output['node_embeddings']  # [num_nodes, hidden_dim]
        graph_embedding = gnn_output['graph_embedding']  # [hidden_dim]
        
        # Compute action logits for each operation
        action_logits = self.policy_head(node_embeddings).squeeze(-1)  # [num_nodes]
        
        # Compute state value
        value = self.value_head(graph_embedding).squeeze(-1)  # scalar
        
        return {
            'action_logits': action_logits,
            'value': value,
            'node_embeddings': node_embeddings
        }
    
    def get_action_and_value(self, node_features, edge_index, mask=None, action=None):
        """
        PPO-compatible interface for getting actions and values.
        
        Args:
            node_features: [batch_size, num_nodes, node_feature_dim] or [num_nodes, node_feature_dim]
            edge_index: single edge index tensor
            mask: validity mask for actions
            action: if provided, compute log prob of this action
        
        Returns:
            action, log_prob, entropy, value
        """
        output = self.forward(node_features, edge_index, mask=None)
        action_logits = output['action_logits']
        value = output['value']
        
        # Simple softmax without mask complications
        log_probs_raw = F.log_softmax(action_logits, dim=-1)
        probs = torch.exp(log_probs_raw)
        
        # Ensure valid probabilities
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / (probs.sum() + 1e-8)
        
        dist = torch.distributions.Categorical(probs=probs)
        
        if action is None:
            action = dist.sample()
            # Handle masking via rejection if needed
            if mask is not None:
                attempts = 0
                while not mask[action] and attempts < 10:
                    action = dist.sample()
                    attempts += 1
        
        log_prob = log_probs_raw[action]
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


def prepare_graph_for_gnn(adjacency_list):
    """
    Convert adjacency list from graph_generator to GNN input format.
    
    Args:
        adjacency_list: dict with node_id -> {duration, machine, job_id, operation_id, neighbors}
    
    Returns:
        node_features: [num_nodes, feature_dim] tensor
        edge_index: [2, num_edges] tensor
        node_id_map: dict mapping original node_id strings to integer indices
    """
    # Create node mapping
    node_ids = sorted(adjacency_list.keys())
    node_id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # Extract node features
    node_features = []
    for node_id in node_ids:
        node_data = adjacency_list[node_id]
        node_features.append([
            node_data['duration'],
            node_data['machine'],
            node_data['job_id'],
            node_data['operation_id']
        ])
    
    node_features = torch.tensor(node_features, dtype=torch.float32)
    
    # Extract edges
    src_list = []
    dst_list = []
    
    for node_id in node_ids:
        src_idx = node_id_map[node_id]
        for neighbor_id in adjacency_list[node_id]['neighbors']:
            dst_idx = node_id_map[neighbor_id]
            src_list.append(src_idx)
            dst_list.append(dst_idx)
    
    if len(src_list) > 0:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return node_features, edge_index, node_id_map


def main():
    """
    Demo: Create a job shop instance, generate graph, and run through GNN.
    """
    print("=" * 60)
    print("Job Shop Scheduling GNN with GAT Layers")
    print("=" * 60)
    
    # Generate a job shop instance
    instances = generate_general_instances(num_instances=1, seed=420)
    instance = instances[0]
    
    print(f"\nInstance: {instance}")
    print(f"  Jobs: {instance.num_jobs}")
    print(f"  Machines: {instance.num_machines}")
    print(f"  Operations: {instance.num_operations}")
    
    # Generate simple disjunctive graph
    adjacency_list = generate_graph(instance)
    print(f"\nGraph nodes: {len(adjacency_list)}")
    
    # Count edges
    total_edges = sum(len(node_data['neighbors']) for node_data in adjacency_list.values())
    print(f"Graph edges: {total_edges}")
    
    # Prepare for GNN
    node_features, edge_index, node_id_map = prepare_graph_for_gnn(adjacency_list)
    print(f"\nNode features shape: {node_features.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    
    # Create GNN policy
    policy = JobShopGNNPolicy(
        node_feature_dim=4,
        hidden_dim=32,
        num_heads=2,
        num_layers=3
    )
    
    print(f"\nGNN Policy Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        output = policy(node_features, edge_index)
        
        print(f"\nGNN Output:")
        print(f"  Action logits shape: {output['action_logits'].shape}")
        print(f"  Value: {output['value'].item():.4f}")
        print(f"  Node embeddings shape: {output['node_embeddings'].shape}")
        
        # Sample an action
        action_probs = F.softmax(output['action_logits'], dim=-1)
        action = torch.argmax(action_probs)
        print(f"\nSelected action (operation): {action.item()}")
        print(f"  Action probability: {action_probs[action].item():.4f}")
        
        # Show top 5 actions
        top_probs, top_actions = torch.topk(action_probs, min(5, len(action_probs)))
        print(f"\nTop 5 operations:")
        for i, (prob, act) in enumerate(zip(top_probs, top_actions)):
            node_id = [k for k, v in node_id_map.items() if v == act.item()][0]
            print(f"  {i+1}. {node_id} (idx {act.item()}): {prob.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
