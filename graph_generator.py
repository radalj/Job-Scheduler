from operation import Operation
from jobshop import JobShopInstance
import random
from generator import generate_general_instances


def generate_graph(instance: JobShopInstance):
    """
    Generate an adjacency list representation of a job shop scheduling graph.
    
    Args:
        instance: JobShopInstance containing jobs and operations
        
    Returns:
        dict: Adjacency list where each node maps to its neighbors and attributes
    """
    # Initialize adjacency list
    adjacency_list = {}
    
    # Create nodes for each operation
    for job_id, job in enumerate(instance.jobs):
        for op_id, operation in enumerate(job):
            node_id = f"job{job_id}_op{op_id}"
            adjacency_list[node_id] = {
                "duration": operation.duration,
                "machine": operation.machine_id,
                "job_id": job_id,
                "operation_id": op_id,
                "neighbors": []  # Adjacent nodes (successors)
            }
    
    # Add precedence constraints (job sequence edges)
    for job_id, job in enumerate(instance.jobs):
        for op_id in range(len(job) - 1):
            current_node = f"job{job_id}_op{op_id}"
            next_node = f"job{job_id}_op{op_id + 1}"
            adjacency_list[current_node]["neighbors"].append(next_node)
    
    # Add machine constraints (operations on same machine)
    # Group operations by machine
    # Group operations by machine using a dict to avoid index errors
    machine_operations = {}
    for job_id, job in enumerate(instance.jobs):
        for op_id, operation in enumerate(job):
            machine_id = operation.machine_id
            if machine_id not in machine_operations:
                machine_operations[machine_id] = []
            machine_operations[machine_id].append(f"job{job_id}_op{op_id}")
    
    # Add edges between operations on the same machine
    # Note: This creates a complete graph between operations on the same machine
    # In practice, you might want to add scheduling-specific constraints
    for machine_id, ops in machine_operations.items():
        for i, op1 in enumerate(ops):
            for j, op2 in enumerate(ops):
                if i != j:  # Don't add self-loops
                    if op2 not in adjacency_list[op1]["neighbors"]:
                        adjacency_list[op1]["neighbors"].append(op2)
    
    return adjacency_list


def generate_multiplex_graph(instance: JobShopInstance):
    """
    Build a multiplex graph with (n + 1) layers:
      Layer 0      -> precedence edges
      Layer 1..n   -> machine layers (one per machine)

    Returns:
        dict with:
            - num_nodes
            - node_features
            - layers: list of (src_list, dst_list)
    """

    num_nodes = instance.num_operations
    num_machines = instance.num_machines

    # ----------------------------
    # Node indexing
    # ----------------------------
    # Map (job_id, op_id) -> global node id
    node_map = {}
    current_id = 0
    for job_id, job in enumerate(instance.jobs):
        for op_id, op in enumerate(job):
            node_map[(job_id, op_id)] = current_id
            current_id += 1

    # ----------------------------
    # Node features (example)
    # ----------------------------
    node_features = []
    for job_id, job in enumerate(instance.jobs):
        for op_id, op in enumerate(job):
            node_features.append([
                op.duration,
                op.machine_id,
                job_id,
                op_id
            ])

    # ----------------------------
    # Layer 0: Precedence edges
    # ----------------------------
    prec_src = []
    prec_dst = []

    for job_id, job in enumerate(instance.jobs):
        for op_id in range(len(job) - 1):
            u = node_map[(job_id, op_id)]
            v = node_map[(job_id, op_id + 1)]
            prec_src.append(u)
            prec_dst.append(v)

    layers = [(prec_src, prec_dst)]

    # ----------------------------
    # Layers 1..n: Machine layers
    # ----------------------------
    # Create empty edge lists per machine
    machine_layers = [[] for _ in range(num_machines)]

    for job_id, job in enumerate(instance.jobs):
        for op_id, op in enumerate(job):
            m = op.machine_id
            node_id = node_map[(job_id, op_id)]
            machine_layers[m].append(node_id)

    # Build clique edges per machine
    for m in range(num_machines):
        ops = machine_layers[m]
        src = []
        dst = []

        for i in range(len(ops)):
            for j in range(len(ops)):
                if i == j:
                    continue
                src.append(ops[i])
                dst.append(ops[j])

        layers.append((src, dst))

    return {
        "num_nodes": num_nodes,
        "x": node_features,
        "layers": layers  # total = n+1 layers
    }


def pretty_print_mux_graph(instance, mux_graph, max_edges_per_layer=12):
    layers = mux_graph["layers"]

    rev = {}
    nid = 0
    for job_id, job in enumerate(instance.jobs):
        for op_id, _ in enumerate(job):
            rev[nid] = (job_id, op_id)
            nid += 1

    print("\n================ MULTIPLEX GRAPH ================")
    print(f"Instance: {getattr(instance, 'name', 'Unnamed')}")
    print(f"jobs={instance.num_jobs}, machines={instance.num_machines}, ops={instance.num_operations}")
    print(f"layers={len(layers)} (expected {instance.num_machines + 1})")
    print("-------------------------------------------------")

    for layer_idx, (src, dst) in enumerate(layers):
        if layer_idx == 0:
            title = "Layer 0: precedence (DAG)"
        else:
            title = f"Layer {layer_idx}: machine {layer_idx - 1}"

        print(f"\n{title} | edges={len(src)}")

        shown = min(max_edges_per_layer, len(src))
        for i in range(shown):
            u, v = src[i], dst[i]
            uj, uo = rev[u]
            vj, vo = rev[v]
            print(f"  job{uj}_op{uo}  ->  job{vj}_op{vo}")

        if len(src) > shown:
            print("  ...")

    print("=================================================\n")


if __name__ == "__main__":
    instance = generate_general_instances()[0]

    # Call multiplex graph generator ✅
    mux_graph = generate_multiplex_graph(instance)

    # Print operations (optional)
    print("\n--- Operations (node list) ---")
    for job in instance.jobs:
        for op in job:
            print(op)

    # Print mux graph in readable format ✅
    pretty_print_mux_graph(instance, mux_graph, max_edges_per_layer=10)
