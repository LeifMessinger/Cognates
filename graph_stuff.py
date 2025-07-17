import pandas as pd
import torch
import numpy as np
from itertools import combinations, chain
from collections import defaultdict
import re

def ipa_to_tensor(ipa_string, ipa_to_ids, max_length=50):
    """Convert IPA string to tensor of IDs."""
    if pd.isna(ipa_string):
        return torch.zeros(max_length, dtype=torch.long)
    
    ids = [ipa_to_ids.get(char, 0) for char in ipa_string]
    
    # Pad or truncate to max_length
    if len(ids) > max_length:
        ids = ids[:max_length]
    else:
        ids.extend([0] * (max_length - len(ids)))
    
    return torch.tensor(ids, dtype=torch.long)

def create_adjacency_matrix(words_data, model, ipa_to_ids, device='cpu'):
    """Create adjacency matrix for a group of words with the same meaning."""
    n_words = len(words_data)
    adj_matrix = torch.zeros((n_words, n_words))
    
    model.eval()
    with torch.no_grad():
        for i in range(n_words):
            for j in range(i+1, n_words):  # Only compute upper triangle
                word1 = words_data.iloc[i]['Phonological Form']
                word2 = words_data.iloc[j]['Phonological Form']
                
                # Convert to tensors
                tensor1 = ipa_to_tensor(word1, ipa_to_ids).unsqueeze(0)
                tensor2 = ipa_to_tensor(word2, ipa_to_ids).unsqueeze(0)
                
                # Create word pair tensor
                word_pair = torch.stack([tensor1, tensor2], dim=1).to(device)
                
                # Create masks (True for padding tokens)
                mask1 = (tensor1 == 0).to(device)
                mask2 = (tensor2 == 0).to(device)
                word_pair_masks = torch.stack([mask1, mask2], dim=1).to(device)
                
                # Get similarity score
                similarity = model(word_pair, word_pair_masks).item()
                
                # Fill both triangles of the matrix
                adj_matrix[i, j] = similarity
                adj_matrix[j, i] = similarity
    
    return adj_matrix

def conductance(cluster, adj):
    """Calculate conductance score for a cluster."""
    if len(cluster) == 0:
        return -1
    
    cluster_set = set(cluster)
    outside = 0
    inside = 0
    
    for i in cluster:
        for j in range(len(adj)):
            if i == j:  # Skip self-loops
                continue
                
            if j in cluster_set:
                inside += adj[i][j] / 2  # Divide by 2 to avoid double counting
            else:
                outside += adj[i][j]
    
    if outside == 0 or inside == 0:
        return -1
    
    return 1 - outside / ((2 * inside) + outside)

def density(cluster, adj):
    """Calculate density score for a cluster."""
    if len(cluster) == 0:
        return 0
    
    total_weight = 0
    for i in cluster:
        for j in cluster:
            if i != j:  # Exclude self-loops
                total_weight += adj[i][j]
    
    # Normalize by maximum possible connections
    max_connections = len(cluster) * (len(cluster) - 1)
    return total_weight / max_connections if max_connections > 0 else 0

def score_cluster(cluster, adj, conductance_weight=0.5, density_weight=0.5):
    """Score a cluster based on conductance and density."""
    if len(cluster) <= 1:
        return -1
    
    cond_score = conductance(cluster, adj)
    dens_score = density(cluster, adj)
    
    if cond_score == -1:
        return -1
    
    return conductance_weight * cond_score + density_weight * dens_score

def generate_all_clusters(available_nodes, max_size):
    """Generate all possible clusters up to max_size."""
    clusters = []
    for size in range(2, min(max_size + 1, len(available_nodes) + 1)):
        for cluster in combinations(available_nodes, size):
            clusters.append(list(cluster))
    return clusters

def find_best_clustering(adj_matrix, conductance_weight=0.5, density_weight=0.5):
    """Find the best clustering using greedy approach."""
    n_nodes = adj_matrix.shape[0]
    available_nodes = list(range(n_nodes))
    clusters = []
    
    while len(available_nodes) > 1:
        max_cluster_size = len(available_nodes) // 2
        if max_cluster_size < 2:
            break
        
        # Generate all possible clusters
        possible_clusters = generate_all_clusters(available_nodes, max_cluster_size)
        
        best_cluster = None
        best_score = -1
        
        # Find best cluster
        for cluster in possible_clusters:
            score = score_cluster(cluster, adj_matrix, conductance_weight, density_weight)
            if score > best_score:
                best_score = score
                best_cluster = cluster
        
        if best_cluster is None or best_score <= 0:
            break
        
        # Add best cluster and remove its nodes from available
        clusters.append(best_cluster)
        available_nodes = [node for node in available_nodes if node not in best_cluster]
    
    # Add remaining nodes as singleton clusters
    for node in available_nodes:
        clusters.append([node])
    
    return clusters

def evaluate_clustering_accuracy(predicted_clusters, true_cognate_classes):
    """Evaluate clustering accuracy against ground truth cognate classes."""
    # Create mapping from node index to predicted cluster
    node_to_pred_cluster = {}
    for cluster_id, cluster in enumerate(predicted_clusters):
        for node in cluster:
            node_to_pred_cluster[node] = cluster_id
    
    # Create mapping from node index to true cognate class
    node_to_true_class = {}
    for i, cognate_class in enumerate(true_cognate_classes):
        node_to_true_class[i] = cognate_class
    
    # Calculate accuracy metrics
    total_pairs = 0
    correct_pairs = 0
    
    n_nodes = len(true_cognate_classes)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            total_pairs += 1
            
            # Check if they should be in same cluster (same cognate class)
            should_be_together = (node_to_true_class[i] == node_to_true_class[j])
            
            # Check if they are in same predicted cluster
            are_together = (node_to_pred_cluster[i] == node_to_pred_cluster[j])
            
            if should_be_together == are_together:
                correct_pairs += 1
    
    accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
    return accuracy

def extract_cognate_class(cc_string):
    """Extract cognate class from cc field (e.g., 'few:I' -> 'I')."""
    if pd.isna(cc_string):
        return None
    return cc_string.split(':')[-1]

def create_groupings(df, ipa_to_ids):
    """
    Filter dataframe and create groupings by meaning.
    
    Args:
        df: DataFrame with columns including 'Meaning', 'Phonological Form', 'cc'
        ipa_to_ids: Dictionary mapping IPA characters to integer IDs
    
    Returns:
        Dict mapping meaning -> (word_array, cognate_class_label_array)
        where word_array is list of tensors and cognate_class_label_array is list of cognate classes
    """
    # Filter to only relevant columns
    filtered_df = df[['Meaning', 'Phonological Form', 'cc']].copy()
    
    # Remove rows with missing data
    filtered_df = filtered_df.dropna()
    
    groupings = {}
    
    # Group by meaning
    meaning_groups = filtered_df.groupby('Meaning')
    
    for meaning, group in meaning_groups:
        if len(group) < 2:
            continue  # Skip meanings with fewer than 2 words
        
        # Convert IPA words to tensors
        word_array = []
        for ipa_word in group['Phonological Form']:
            tensor = ipa_to_tensor(ipa_word, ipa_to_ids)
            word_array.append(tensor)
        
        # Extract cognate class labels
        cognate_class_label_array = [extract_cognate_class(cc) for cc in group['cc']]
        
        groupings[meaning] = (word_array, cognate_class_label_array)
    
    return groupings

def cluster_grouping(word_array, model, device='cpu'):
    """
    Cluster a group of words using the model to create adjacency matrix.
    
    Args:
        word_array: List of tensors representing IPA words
        model: Trained transformer model
        device: Device to run model on
    
    Returns:
        List of clusters, where each cluster is a list of indices
    """
    n_words = len(word_array)
    
    # Create adjacency matrix
    adj_matrix = torch.zeros((n_words, n_words))
    
    model.eval()
    with torch.no_grad():
        for i in range(n_words):
            for j in range(i+1, n_words):
                # Create word pair tensor
                word_pair = torch.stack([word_array[i], word_array[j]], dim=0).unsqueeze(0).to(device)
                
                # Create masks (True for padding tokens)
                mask1 = (word_array[i] == 0).unsqueeze(0).to(device)
                mask2 = (word_array[j] == 0).unsqueeze(0).to(device)
                word_pair_masks = torch.stack([mask1, mask2], dim=1).to(device)
                
                # Get similarity score
                similarity = model(word_pair, word_pair_masks).item()
                
                # Fill both triangles of the matrix
                adj_matrix[i, j] = similarity
                adj_matrix[j, i] = similarity
    
    # Find best clustering
    predicted_clusters = find_best_clustering(adj_matrix)
    
    return predicted_clusters

def evaluate_clusters(predicted_clusters, cognate_class_label_array, meaning):
    """
    Evaluate clustering accuracy against ground truth cognate classes.
    
    Args:
        predicted_clusters: List of clusters from cluster_grouping()
        cognate_class_label_array: List of true cognate class labels
        meaning: The meaning being evaluated (for printing)
    
    Returns:
        Dict with evaluation metrics
    """
    accuracy = evaluate_clustering_accuracy(predicted_clusters, cognate_class_label_array)
    
    results = {
        'meaning': meaning,
        'predicted_clusters': predicted_clusters,
        'true_classes': cognate_class_label_array,
        'accuracy': accuracy,
        'num_words': len(cognate_class_label_array),
        'num_predicted_clusters': len(predicted_clusters),
        'num_true_classes': len(set(cognate_class_label_array))
    }
    
    return results

def cluster_and_evaluate_all_meanings(df, model, ipa_to_ids, device='cpu'):
    """Process all word meanings and evaluate clustering using the modular approach."""
    # Step 1: Preprocessing
    groupings = create_groupings(df, ipa_to_ids)
    
    results = {}
    
    # Step 2 & 3: Clustering and Evaluation
    for meaning, (word_array, cognate_class_label_array) in groupings.items():
        print(f"\nProcessing meaning: {meaning}")
        print(f"Number of words: {len(word_array)}")
        
        # Step 2: Clustering
        predicted_clusters = cluster_grouping(word_array, model, device)
        
        # Step 3: Evaluation
        evaluation_results = evaluate_clusters(predicted_clusters, cognate_class_label_array, meaning)
        
        results[meaning] = evaluation_results
        
        print(f"Predicted clusters: {predicted_clusters}")
        print(f"True classes: {cognate_class_label_array}")
        print(f"Accuracy: {evaluation_results['accuracy']:.3f}")
    
    return results

# Example usage:
def main():
    # Load data
    df = pd.read_csv('data/ielexData.csv')
    
    # Create IPA to ID mapping
    import joblib
    ipa_embedder = joblib.load("data/embeddings/34.joblib")
    ipa_to_ids = ipa_embedder.char_to_idx
    
    # Step 1: Preprocessing
    groupings = create_groupings(df, ipa_to_ids)
    print(f"Created {len(groupings)} meaning groups")
    
    # Example of using the modular approach:
    model = torch.load('TransformerCognateModel_34.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Process a single meaning group
    # meaning = 'few'
    # if meaning in groupings:
    #     word_array, cognate_labels = groupings[meaning]
    #     clusters = cluster_grouping(word_array, model, device)
    #     results = evaluate_clusters(clusters, cognate_labels, meaning)
    #     print(f"Results for '{meaning}': {results}")
    
    # Or process all meanings
    all_results = cluster_and_evaluate_all_meanings(df, model, ipa_to_ids, device)
    accuracies = [r['accuracy'] for r in all_results.values()]
    print(f"\nOverall average accuracy: {np.mean(accuracies):.3f}")
    
if __name__ == "__main__":
    main()