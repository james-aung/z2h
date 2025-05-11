import torch
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
   block_size: int = 8
   batch_size: int = 32
   learning_rate: float = 1e-3
   num_training_steps: int = 10000
   train_test_ratio: float = 0.9
   eval_interval: int = 100
   embed_size: int = 32

class BaseTokenizer(ABC):
   @abstractmethod
   def encode(self, s: str) -> List[int]:
      pass

   @abstractmethod
   def decode(self, l: List[int]) -> str:
      pass
   
   @property
   @abstractmethod
   def vocab_size(self) -> int:
      pass
   
class CharacterTokenizer(BaseTokenizer):
   def __init__(self, text):
      self.vocab = sorted(list(set(text)))
      self.stoi = { ch:i for i,ch in enumerate(self.vocab) }
      self.itos = { i:ch for i,ch in enumerate(self.vocab) }

   def encode(self, s: str) -> List[int]:
      return [self.stoi[c] for c in s]

   def decode(self, l: List[int]) -> str:
      return ''.join([self.itos[i] for i in l])

   @property
   def vocab_size(self):
      return len(self.vocab)

class Dataset:
   def __init__(self, path, train_test_ratio=0.9):
      text = open(path, 'r', encoding='utf-8').read()
      self.tokenizer = CharacterTokenizer(text)
      self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
      self.train = self.data[:int(train_test_ratio*len(self.data))]
      self.test = self.data[int(train_test_ratio*len(self.data)):]

   
   def get_batch(self, split, block_size, batch_size):
      data = self.train if split == 'train' else self.test
      random_indices = torch.randint(len(data) - block_size, (batch_size,))
      x = torch.stack([data[i:i+block_size] for i in random_indices])
      y = torch.stack([data[i+1:i+block_size+1] for i in random_indices])
      return x, y
   
class FeedForward(nn.Module):
   def __init__(self, embed_size):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(embed_size, embed_size),
         nn.ReLU(),
      )

   def forward(self, x):
      return self.net(x)

class Head(nn.Module):
   def __init__(self, head_size, embed_size, block_size):
      super().__init__()
      self.key = nn.Linear(embed_size, head_size, bias=False)
      self.query = nn.Linear(embed_size, head_size, bias=False)
      self.value = nn.Linear(embed_size, head_size, bias=False)
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

   def forward(self, x):
      B, T, C = x.shape
      k = self.key(x)
      q = self.query(x)
      weights = q @ k.transpose(-2, -1) * (C ** -0.5)
      weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
      weights = F.softmax(weights, dim=-1)
      out = weights @ self.value(x)
      return out

class MultiHeadAttention(nn.Module):
   def __init__(self, num_heads, head_size, embed_size, block_size):
      super().__init__()
      self.heads = nn.ModuleList([Head(head_size, embed_size, block_size) for _ in range(num_heads)])

   def forward(self, x):
      return torch.cat([head(x) for head in self.heads], dim=-1)

class Block(nn.Module):
   def __init__(self, embed_size, block_size, num_heads):
      super().__init__()
      head_size = embed_size // num_heads
      self.sa = MultiHeadAttention(num_heads, head_size, embed_size, block_size)
      self.ffwd = FeedForward(embed_size)

   def forward(self, x):
      x = self.sa(x)
      x = self.ffwd(x)
      return x

class BigramLanguageModel(nn.Module):
   def __init__(self, vocab_size, block_size, embed_size):
      super().__init__()
      self.block_size = block_size
      self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
      self.position_embedding_table = nn.Embedding(block_size, embed_size)
      self.blocks = nn.Sequential(
         Block(num_heads=4, block_size=block_size, embed_size=embed_size),
         Block(num_heads=4, block_size=block_size, embed_size=embed_size),
         Block(num_heads=4, block_size=block_size, embed_size=embed_size),
      )
      self.lm_head = nn.Linear(embed_size, vocab_size)

   def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
      B, T = idx.shape
      token_embeddings = self.token_embedding_table(idx)
      position_embeddings = self.position_embedding_table(torch.arange(T))
      x = token_embeddings + position_embeddings
      for block in self.blocks:
         x = block(x)
      logits = self.lm_head(x)
      if targets is None:
         loss = None
      else:
         B, T, C = logits.shape
         logits = logits.view(B*T, C)
         targets = targets.view(B*T)
         loss = nn.functional.cross_entropy(logits, targets)
      return logits, loss

   def generate(self, idx, max_new_tokens):
      for _ in range(max_new_tokens):
         idx_cond = idx[:, -self.block_size:]
         logits, loss = self(idx_cond)
         logits = logits[:, -1, :] # note: using all-but-last dimension
         probs = F.softmax(logits, dim=-1)
         idx_next = torch.multinomial(probs, num_samples=1)
         idx = torch.cat((idx, idx_next), dim=-1)
      return idx

class ModelTrainer:
   def __init__(self, model: nn.Module, dataset: Dataset, config: ModelConfig):
      self.model = model
      self.dataset = dataset
      self.config = config
      self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
      self.losses = []

   def train_step(self) -> float:
      xb, yb = self.dataset.get_batch('train', self.config.block_size, self.config.batch_size)
      logits, loss = self.model(xb, yb)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      return loss.item()

   def train(self):
      for step in range(self.config.num_training_steps):
         loss = self.train_step()
         self.losses.append(loss)
         
         # Evaluate and print losses every eval_interval steps
         if step % self.config.eval_interval == 0:
            losses = self.estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")

   @torch.no_grad()
   def estimate_loss(self):
      out = {}
      self.model.eval()
      for split in ['train', 'test']:
         losses = torch.zeros(self.config.eval_interval)
         for k in range(self.config.eval_interval):
            X, Y = self.dataset.get_batch(split, self.config.block_size, self.config.batch_size)
            logits, loss = self.model(X, Y)
            losses[k] = loss.item()
         out[split] = losses.mean()
      self.model.train()
      return out


class TextGenerator:
    def __init__(self, model: BigramLanguageModel, tokenizer: BaseTokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate(self, prompt: str = '\n', max_new_tokens: int = 1000) -> str:
        starting_tokens = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long)
        starting_tokens = starting_tokens.unsqueeze(0)
        generated_tokens = self.model.generate(starting_tokens, max_new_tokens)
        return self.tokenizer.decode(generated_tokens[0].tolist())

def main():
    config = ModelConfig()
    dataset = Dataset('gpt/data/input.txt', config.train_test_ratio)
    model = BigramLanguageModel(dataset.tokenizer.vocab_size, config.block_size, config.embed_size)
    trainer = ModelTrainer(model, dataset, config)
    generator = TextGenerator(model, dataset.tokenizer)

    print("\nInitial generation:")
    print(generator.generate(prompt="\n", max_new_tokens=1000))
    
    trainer.train()
    
    print("\nFinal generation:")
    print(generator.generate(prompt="\n", max_new_tokens=1000))

if __name__ == "__main__":
    main()