import pandas as pd
import torch
import torch.nn as nn

class SetEmbedding:
    def __init__(self, set_of_things, embedding_dim):
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(len(set_of_things), embedding_dim, padding_idx=0, _freeze=True)

        self.ids = {thing: torch.tensor([idx]) for idx, thing in enumerate(set_of_things)}

    def __call__(self, character):
        # This is a dummy embedding; replace with your actual embedding function
        return self.embedding(self.ids[character])

class FrozenIPAEmbedding:
	def __init__(self, char_to_idx, weights, embedding_dim=6, device=torch.get_default_device()):
		self.device = device
		self.embedding_dim = embedding_dim
		self.char_to_idx = char_to_idx
		self.embeddings = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)
	
	def __call__(self, character):
		idx = self.char_to_idx.get(character, 0)  # Default to 0 (PAD) for unseen chars
		return self.embeddings(torch.tensor([idx], device=self.device))

class IPAEmbedding:
	def __init__(self, all_characters, embedding_dim=6, device=torch.get_default_device()):
		self.embedding_dim = embedding_dim
		self.device = device
		
		# Load GLED data for training character embeddings
		gled_df = pd.read_csv("data/gled.tsv", delimiter="\t")
		words = gled_df['FORM'].dropna().tolist()
		
		# Create character contexts for word2vec-style training
		contexts = []
		for word in words:
			if not isinstance(word, str):
				continue
			chars = list(word)
			if len(chars) <= 1:
				continue
			# For each character, create context pairs with surrounding characters
			for i in range(len(chars)):
				# Use a window of 2 characters on each side
				context_start = max(0, i-2)
				context_end = min(len(chars), i+3)
				target = chars[i]
				context = chars[context_start:i] + chars[i+1:context_end]
				contexts.append((target, context))
		
		# Create vocabulary of unique characters
		self.char_to_idx = {char: idx+1 for idx, char in enumerate(all_characters)}
		self.char_to_idx['<PAD>'] = 0
		self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
		vocab_size = len(self.char_to_idx)
		
		# Initialize embedding matrices
		import torch.nn as nn
		self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
		
		# Train character embeddings using negative sampling approach
		self._train_embeddings(contexts, epochs=1)
		
	def _train_embeddings(self, contexts, epochs=1, batch_size=128, lr=0.01):
		import random
		import torch.optim as optim
		import torch.nn.functional as F
		
		optimizer = optim.Adam(self.embeddings.parameters(), lr=lr)
		
		from tqdm import tqdm
		for epoch in range(epochs):
			random.shuffle(contexts)
			total_loss = 0
			
			for i in tqdm(range(0, len(contexts), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
				batch = contexts[i:i+batch_size]
				
				# Prepare target and context tensors
				target_chars = [pair[0] for pair in batch]
				context_chars = [pair[1] for pair in batch]
				
				# Convert to indices
				target_indices = torch.tensor([self.char_to_idx.get(char, 0) for char in target_chars], device=self.device)
				
				# Process each context
				batch_loss = 0
				for idx, (target_idx, context) in enumerate(zip(target_indices, context_chars)):
					context_indices = torch.tensor([self.char_to_idx.get(char, 0) for char in context], device=self.device)
					if len(context_indices) == 0:
						continue
					
					# Get embeddings
					target_emb = self.embeddings(target_idx)
					context_embs = self.embeddings(context_indices)
					
					# Calculate similarity
					similarity = F.cosine_similarity(target_emb.unsqueeze(0), context_embs, dim=1)
					
					# Use binary cross entropy with positive examples
					pos_loss = -torch.log(torch.sigmoid(similarity)).mean()
					
					# Negative sampling
					neg_indices = torch.randint(1, len(self.char_to_idx), (len(context)*3,), device=self.device)
					neg_embs = self.embeddings(neg_indices)
					neg_similarity = F.cosine_similarity(target_emb.unsqueeze(0), neg_embs, dim=1)
					neg_loss = -torch.log(1 - torch.sigmoid(neg_similarity)).mean()
					
					# Combined loss
					loss = pos_loss + neg_loss
					batch_loss += loss
				
				if batch_loss > 0:
					optimizer.zero_grad()
					batch_loss.backward()
					optimizer.step()
					total_loss += batch_loss.item()
			
			if epoch % 2 == 0:
				print(f"Epoch {epoch}, Loss: {total_loss/max(1, len(contexts)//batch_size):.4f}")
		
		print("Character embedding training complete")
	
	def __call__(self, character):
		idx = self.char_to_idx.get(character, 0)  # Default to 0 (PAD) for unseen chars
		return self.embeddings(torch.tensor([idx], device=self.device))
	
	def frozen(self):
		return FrozenIPAEmbedding(self.char_to_idx, self.embeddings.weight.detach(), embedding_dim=self.embedding_dim)