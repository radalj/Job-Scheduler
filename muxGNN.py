import torch
import torch.nn as nn
import torch.nn.functional as F

from GNN import GATLayer


class MuxSimpleGNN(nn.Module):
    """
    Expected relation keys in edge_indices:
      - "precedence"
      - "machine"
    """

    def __init__(self, node_feature_dim=4, hidden_dim=16, num_heads=1, num_layers=2, dropout=0.1):
        super(MuxSimpleGNN, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.relation_names = ("precedence", "machine")

        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        self.relation_layers = nn.ModuleList()
        self.relation_gates = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            per_relation = nn.ModuleDict({
                rel: GATLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=False,
                )
                for rel in self.relation_names
            })
            self.relation_layers.append(per_relation)
            self.relation_gates.append(nn.Linear(hidden_dim, len(self.relation_names)))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.final_node_dim = hidden_dim
        self.final_graph_dim = hidden_dim
        self.graph_pooling = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_features, edge_indices):
        x = F.relu(self.input_proj(node_features))
        device = x.device

        for layer_idx in range(self.num_layers):
            relation_outputs = []
            for rel in self.relation_names:
                rel_edge_index = edge_indices.get(rel)
                if rel_edge_index is None:
                    rel_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                relation_outputs.append(self.relation_layers[layer_idx][rel](x, rel_edge_index))

            stacked = torch.stack(relation_outputs, dim=1)  # [N, R, H]
            gate_logits = self.relation_gates[layer_idx](x)
            gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)  # [N, R, 1]

            fused = (stacked * gate_weights).sum(dim=1)  # [N, H]
            x = self.layer_norms[layer_idx](x + fused)
            x = F.relu(x)

        node_embeddings = x
        graph_mean = torch.mean(node_embeddings, dim=0)
        graph_embedding = self.graph_pooling(graph_mean)

        return {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding
        }


class MuxJobShopGNNPolicy(nn.Module):
    """
    Multiplex policy for PPO using relation-specific graph channels.
    """

    def __init__(self, node_feature_dim=4, hidden_dim=16, num_heads=1, num_layers=2):
        super(MuxJobShopGNNPolicy, self).__init__()

        self.gnn = MuxSimpleGNN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )

        self.policy_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, edge_index=None, mask=None, edge_indices=None):
        if edge_indices is None:
            shared = edge_index if edge_index is not None else torch.zeros((2, 0), dtype=torch.long, device=node_features.device)
            edge_indices = {
                'precedence': shared,
                'machine': shared,
            }

        gnn_output = self.gnn(node_features, edge_indices)
        node_embeddings = gnn_output['node_embeddings']
        graph_embedding = gnn_output['graph_embedding']

        action_logits = self.policy_head(node_embeddings).squeeze(-1)
        value = self.value_head(graph_embedding).squeeze(-1)

        return {
            'action_logits': action_logits,
            'value': value,
            'node_embeddings': node_embeddings
        }

    def get_action_and_value(self, node_features, edge_index=None, mask=None, action=None, edge_indices=None):
        output = self.forward(node_features, edge_index=edge_index, mask=mask, edge_indices=edge_indices)
        action_logits = output['action_logits']
        value = output['value']

        log_probs_raw = F.log_softmax(action_logits, dim=-1)
        probs = torch.exp(log_probs_raw)
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / (probs.sum() + 1e-8)

        dist = torch.distributions.Categorical(probs=probs)

        if action is None:
            action = dist.sample()
            if mask is not None:
                attempts = 0
                while not mask[action] and attempts < 10:
                    action = dist.sample()
                    attempts += 1

        log_prob = log_probs_raw[action]
        entropy = dist.entropy()
        return action, log_prob, entropy, value
