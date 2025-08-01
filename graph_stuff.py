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

def evaluate_clustering_f1(predicted_clusters, true_cognate_classes):
    """Evaluate clustering F1 score against ground truth cognate classes."""
    # Create mapping from node index to predicted cluster
    node_to_pred_cluster = {}
    for cluster_id, cluster in enumerate(predicted_clusters):
        for node in cluster:
            node_to_pred_cluster[node] = cluster_id
    
    # Create mapping from node index to true cognate class
    node_to_true_class = {}
    for i, cognate_class in enumerate(true_cognate_classes):
        node_to_true_class[i] = cognate_class
    
    # Create lists for true labels and predicted labels
    true_labels = []
    pred_labels = []
    
    n_nodes = len(true_cognate_classes)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            # Check if they should be in same cluster (same cognate class)
            should_be_together = (node_to_true_class[i] == node_to_true_class[j])
            true_labels.append(should_be_together)
            
            # Check if they are in same predicted cluster
            are_together = (node_to_pred_cluster[i] == node_to_pred_cluster[j])
            pred_labels.append(are_together)
    
    # Calculate precision, recall, and F1 score
    from sklearn.metrics import precision_score, recall_score, f1_score
    # precision = precision_score(true_labels, pred_labels)
    # recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    return f1

def evaluate_clustering_nmi(predicted_clusters, true_cognate_classes):
    """Evaluate clustering NMI against ground truth cognate classes."""
    # Create mapping from node index to predicted cluster
    node_to_pred_cluster = {}
    for cluster_id, cluster in enumerate(predicted_clusters):
        for node in cluster:
            node_to_pred_cluster[node] = cluster_id
    
    # Create list for predicted labels
    pred_labels = [node_to_pred_cluster[i] for i in range(len(true_cognate_classes))]
    
    # Calculate NMI
    from sklearn.metrics import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(true_cognate_classes, pred_labels)
    
    return nmi

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
    filtered_df = df[['Language', 'Meaning', 'Phonological Form', 'cc']].copy()
    
    # Remove rows with missing data
    filtered_df = filtered_df.dropna()
    
    groupings = {}
    
    # Group by meaning
    meaning_groups = filtered_df.groupby('Meaning')
    
    for meaning, group in meaning_groups:
        if len(group) < 2:
            continue  # Skip meanings with fewer than 2 words
        
        # Convert IPA words to tensors
        languages = group['Language'].tolist()
        phonological_words = group['Phonological Form'].tolist()
        labels = [language + ': ' + word for language, word in zip(languages, phonological_words)]
        word_array = []
        for ipa_word in group['Phonological Form']:
            tensor = ipa_to_tensor(ipa_word, ipa_to_ids)
            word_array.append(tensor)
        
        # Extract cognate class labels
        cognate_class_label_array = [extract_cognate_class(cc) for cc in group['cc']]
        
        groupings[meaning] = (word_array, cognate_class_label_array, labels, meaning)
    
    return groupings

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
    f1 = evaluate_clustering_f1(predicted_clusters, cognate_class_label_array)
    nmi = evaluate_clustering_nmi(predicted_clusters, cognate_class_label_array)
    
    results = {
        'meaning': meaning,
        'predicted_clusters': predicted_clusters,
        'true_classes': cognate_class_label_array,
        'accuracy': accuracy,
        'f1': f1,
        'nmi': nmi,
        'num_words': len(cognate_class_label_array),
        'num_predicted_clusters': len(predicted_clusters),
        'num_true_classes': len(set(cognate_class_label_array))
    }
    
    return results