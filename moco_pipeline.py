import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm.notebook import tqdm

# Method 1: Vectorized with pad_sequence (Most Pythonic)
def preprocess_vectorized(all_pairs, ipa_embedder, device):
    """
    More efficient preprocessing using PyTorch's pad_sequence
    """
    # Convert all words to tensors first
    all_word_tensors = []
    pair_indices = []
    
    for pair_idx, (word1, word2) in enumerate(all_pairs):
        # Convert characters to indices
        word1_tensor = torch.tensor([ipa_embedder.char_to_idx[char] for char in word1], 
                                   dtype=torch.int, device=device)
        word2_tensor = torch.tensor([ipa_embedder.char_to_idx[char] for char in word2], 
                                   dtype=torch.int, device=device)
        
        all_word_tensors.extend([word1_tensor, word2_tensor])
        pair_indices.extend([pair_idx, pair_idx])
    
    # Pad all sequences at once
    padded = pad_sequence(all_word_tensors, batch_first=True, padding_value=0)
    
    # Reshape to [num_pairs, 2, max_len]
    batches = padded.view(len(all_pairs), 2, -1)
    
    # Create attention masks (True where padded)
    lengths = torch.tensor([len(tensor) for tensor in all_word_tensors], device=device)
    lengths = lengths.view(len(all_pairs), 2)
    
    max_len = batches.shape[2]
    batches_masks = torch.zeros((len(all_pairs), 2, max_len), dtype=torch.bool, device=device)
    
    for i in range(len(all_pairs)):
        for j in range(2):
            if lengths[i, j] < max_len:
                batches_masks[i, j, lengths[i, j]:] = True
    
    return batches, batches_masks, max_len

# Method 2: NumPy vectorization (Fastest for large datasets)
def preprocess_numpy(all_pairs, ipa_embedder, device):
    """
    NumPy-based preprocessing - fastest for large datasets
    """
    # Find max length
    max_length = max(max(len(word) for word in pair) for pair in all_pairs)
    
    # Pre-allocate numpy arrays
    batches_np = np.zeros((len(all_pairs), 2, max_length), dtype=np.int32)
    masks_np = np.zeros((len(all_pairs), 2, max_length), dtype=bool)
    
    # Vectorized character lookup
    char_to_idx = ipa_embedder.char_to_idx
    
    for pair_idx, (word1, word2) in enumerate(all_pairs):
        for word_idx, word in enumerate([word1, word2]):
            word_len = len(word)
            # Vectorized character to index conversion
            indices = [char_to_idx[char] for char in word]
            batches_np[pair_idx, word_idx, :word_len] = indices
            # Set mask for padding
            masks_np[pair_idx, word_idx, word_len:] = True
    
    # Convert to torch tensors
    batches = torch.from_numpy(batches_np).to(device)
    batches_masks = torch.from_numpy(masks_np).to(device)
    
    return batches, batches_masks, max_length

# Method 3: Lazy loading with Dataset class (Memory efficient)
class CognateDataset(torch.utils.data.Dataset):
    """
    Dataset class that preprocesses on-the-fly
    Avoids storing all preprocessed data in memory
    """
    def __init__(self, all_pairs, all_labels, ipa_embedder, max_length=None):
        self.all_pairs = all_pairs
        self.all_labels = all_labels
        self.ipa_embedder = ipa_embedder
        self.char_to_idx = ipa_embedder.char_to_idx
        
        # Calculate max length if not provided
        if max_length is None:
            self.max_length = max(max(len(word) for word in pair) for pair in all_pairs)
        else:
            self.max_length = max_length
    
    def __len__(self):
        return len(self.all_pairs)
    
    def __getitem__(self, idx):
        word1, word2 = self.all_pairs[idx]
        label = self.all_labels[idx]
        
        # Convert words to tensors
        word1_indices = [self.char_to_idx[char] for char in word1]
        word2_indices = [self.char_to_idx[char] for char in word2]
        
        # Pad to max length
        word1_padded = word1_indices + [0] * (self.max_length - len(word1_indices))
        word2_padded = word2_indices + [0] * (self.max_length - len(word2_indices))
        
        # Create masks
        mask1 = [False] * len(word1_indices) + [True] * (self.max_length - len(word1_indices))
        mask2 = [False] * len(word2_indices) + [True] * (self.max_length - len(word2_indices))
        
        # Stack into pair format
        word_pair = torch.tensor([word1_padded, word2_padded], dtype=torch.int)
        masks = torch.tensor([mask1, mask2], dtype=torch.bool)
        
        return word_pair, masks, torch.tensor(label, dtype=torch.float)

# Method 4: Batch processing with collate function (Most flexible)
def collate_fn(batch, ipa_embedder):
    """
    Custom collate function for DataLoader
    Processes batches dynamically with optimal padding per batch
    """
    pairs, labels = zip(*batch)
    
    # Flatten all words in the batch
    all_words = []
    pair_info = []  # (batch_idx, word_idx_in_pair)
    
    for batch_idx, (word1, word2) in enumerate(pairs):
        all_words.extend([word1, word2])
        pair_info.extend([(batch_idx, 0), (batch_idx, 1)])
    
    # Convert to tensors
    word_tensors = []
    for word in all_words:
        indices = [ipa_embedder.char_to_idx[char] for char in word]
        word_tensors.append(torch.tensor(indices, dtype=torch.int))
    
    # Pad to batch max (not global max)
    padded_words = pad_sequence(word_tensors, batch_first=True, padding_value=0)
    
    # Reshape back to pairs
    batch_size = len(pairs)
    max_len = padded_words.shape[1]
    
    word_pairs = padded_words.view(batch_size, 2, max_len)
    
    # Create masks
    masks = torch.zeros(batch_size, 2, max_len, dtype=torch.bool)
    word_idx = 0
    for batch_idx in range(batch_size):
        for pair_word_idx in range(2):
            word_len = len(word_tensors[word_idx])
            masks[batch_idx, pair_word_idx, word_len:] = True
            word_idx += 1
    
    labels_tensor = torch.stack([torch.tensor(label, dtype=torch.float) for label in labels])
    
    return word_pairs, masks, labels_tensor

# Method 5: Memory-mapped approach for very large datasets
def preprocess_memory_mapped(all_pairs, ipa_embedder, device, mmap_file='cognate_data.npy'):
    """
    Memory-mapped preprocessing for datasets that don't fit in RAM
    """
    import numpy as np
    
    max_length = max(max(len(word) for word in pair) for pair in all_pairs)
    
    # Create memory-mapped array
    shape = (len(all_pairs), 2, max_length)
    batches_mmap = np.memmap(mmap_file, dtype=np.int32, mode='w+', shape=shape)
    
    # Process in chunks to avoid memory issues
    chunk_size = 1000
    for chunk_start in tqdm(range(0, len(all_pairs), chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, len(all_pairs))
        
        for i in range(chunk_start, chunk_end):
            pair = all_pairs[i]
            for word_idx, word in enumerate(pair):
                for char_idx, char in enumerate(word):
                    batches_mmap[i, word_idx, char_idx] = ipa_embedder.char_to_idx[char]
    
    # Flush to disk
    del batches_mmap
    
    # Return memory-mapped tensor wrapper
    class MemoryMappedTensor:
        def __init__(self, filename, shape, dtype=np.int32):
            self.mmap = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
        
        def __getitem__(self, idx):
            return torch.from_numpy(self.mmap[idx].copy()).to(device)
        
        def __len__(self):
            return self.mmap.shape[0]
    
    return MemoryMappedTensor(mmap_file, shape), max_length

# Usage examples and benchmarking
def benchmark_methods(all_pairs, ipa_embedder, device):
    """
    Benchmark different preprocessing methods
    """
    import time
    
    methods = {
        'vectorized': preprocess_vectorized,
        'numpy': preprocess_numpy,
    }
    
    results = {}
    
    for name, method in methods.items():
        print(f"Testing {name}...")
        start_time = time.time()
        
        try:
            batches, masks, max_len = method(all_pairs, ipa_embedder, device)
            end_time = time.time()
            
            results[name] = {
                'time': end_time - start_time,
                'memory': batches.element_size() * batches.nelement(),
                'shape': batches.shape
            }
            print(f"  Time: {results[name]['time']:.2f}s")
            print(f"  Memory: {results[name]['memory'] / 1024**2:.1f} MB")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[name] = {'error': str(e)}
    
    return results

# Example usage:
"""
# For small to medium datasets:
batches, masks, max_len = preprocess_numpy(all_pairs, ipa_embedder, device)

# For large datasets:
dataset = CognateDataset(all_pairs, all_labels, ipa_embedder)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=lambda x: collate_fn(x, ipa_embedder))

# For very large datasets:
mmap_tensor, max_len = preprocess_memory_mapped(all_pairs, ipa_embedder, device)
"""