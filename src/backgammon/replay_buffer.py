import torch
import random
import math
from typing import List, Tuple, Any
from src.backgammon.config import Config

# -------------------------------------------------
# FastSumTree (Optimized for Strict Capacity)
# -------------------------------------------------
class FastSumTree:
    """
    A binary tree where each node is the sum of its children.
    Leaf nodes store the priorities.
    """
    def __init__(self, max_size: int):
        self.max_size = max_size
        
        # Tree capacity must be a power of 2 for the binary search logic
        self.tree_capacity = 1
        while self.tree_capacity < max_size:
            self.tree_capacity *= 2

        # Nodes: 2 * tree_capacity - 1
        # Indices 0 to (tree_capacity - 2) are internal nodes
        # Indices (tree_capacity - 1) onwards are leaf nodes
        self.tree = [0.0] * (2 * self.tree_capacity - 1)
        self.data = [None] * self.max_size
        
        self.write_ptr = 0
        self.current_count = 0

    def update(self, tree_idx: int, priority: float):
        """Updates a priority and propagates the change up the tree."""
        priority = max(float(priority), 1e-6)
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate change to root
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, data: Any, priority: float):
        """Adds data with a priority, overwriting the oldest if full."""
        # Find the leaf index corresponding to our circular pointer
        tree_idx = self.write_ptr + self.tree_capacity - 1
        
        self.data[self.write_ptr] = data
        self.update(tree_idx, priority)
        
        # FIFO Logic: Wrap pointer around at exactly max_size
        self.write_ptr = (self.write_ptr + 1) % self.max_size
        self.current_count = min(self.current_count + 1, self.max_size)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """Prefix sum search to find a leaf node for a value s."""
        idx = 0
        while idx < self.tree_capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
                
        # Map tree index back to data index
        data_idx = idx - (self.tree_capacity - 1)
        
        # Safety: Ensure data_idx is within current valid bounds
        # This can happen if s lands in the 'dead' space of a power-of-2 tree
        if data_idx >= self.max_size or self.data[data_idx] is None:
            data_idx = (self.write_ptr - 1) % self.max_size
            idx = data_idx + self.tree_capacity - 1

        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total_priority(self):
        return max(self.tree[0], 1e-6)

# -------------------------------------------------
# PrioritizedReplayBuffer
# -------------------------------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, device: str = "cpu"):
        self.max_size = capacity
        self.tree = FastSumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        self.device = device

    def add(self, data: Any, priority: float = None):
        """Adds a single transition."""
        if priority is None:
            priority = self.max_priority
        
        p = (abs(priority) + 1e-5) ** self.alpha
        self.tree.add(data, p)

    def extend(self, data_list: List[Any]):
        """Adds a list of transitions (used after a full match)."""
        for data in data_list:
            self.add(data)

    def sample(self, batch_size: int) -> Tuple[List[Any], torch.Tensor, torch.Tensor]:
        if self.tree.current_count < batch_size:
            return None, None, None

        batch, tree_idxs, priorities = [], [], []
        segment = self.tree.total_priority / batch_size
        
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            tree_idxs.append(idx)
            priorities.append(p)

        # Importance Sampling Weights: (N * P(i))^-beta / max_weight
        probs = [p / self.tree.total_priority for p in priorities]
        weights = [(self.tree.current_count * pr) ** -self.beta for pr in probs]
        
        max_w = max(weights) if weights else 1.0
        weights_t = torch.tensor([w / max_w for w in weights], dtype=torch.float32, device=self.device)
        indices_t = torch.tensor(tree_idxs, dtype=torch.long, device=self.device)

        return batch, indices_t, weights_t

    def update_priorities(self, indices: torch.Tensor, td_errors: Any):
        """Updates priorities in the tree based on new TD errors from training."""
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().tolist()
        
        indices_list = indices.detach().cpu().tolist()
        
        for idx, error in zip(indices_list, td_errors):
            p = (abs(error) + 1e-5) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.current_count

# -------------------------------------------------
# SimpleReplayBuffer (Non-Prioritized Fallback)
# -------------------------------------------------
class SimpleReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def add(self, data: Any):
        self.data[self.write_idx] = data
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def extend(self, data_list: List[Any]):
        for data in data_list:
            self.add(data)

    def sample(self, batch_size: int):
        if self.size < batch_size: return None, None, None
        indices = [random.randint(0, self.size - 1) for _ in range(batch_size)]
        batch = [self.data[i] for i in indices]
        return batch, torch.tensor(indices), torch.ones(batch_size)

    def __len__(self):
        return self.size

def get_replay_buffer(capacity: int, prioritized: bool = True, **kwargs):
    if prioritized:
        return PrioritizedReplayBuffer(capacity, **kwargs)
    return SimpleReplayBuffer(capacity)