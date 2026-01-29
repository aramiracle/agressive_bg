import torch
import random
import math
from typing import List, Tuple, Any
from src.backgammon.config import Config

# -------------------------------------------------
# FastSumTree (PURE PYTHON - Optimized for speed)
# -------------------------------------------------
class FastSumTree:
    def __init__(self, capacity: int):
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2

        # Standard list is faster than torch.tensor for scalar updates
        self.tree = [0.0] * (2 * self.capacity - 1)
        self.data = [None] * self.capacity
        self.write_idx = 0
        self.count = 0

    def update(self, idx: int, priority: float):
        priority = max(priority, 1e-6) # Safety floor
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def add(self, data: Any, priority: float):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def get(self, s: float) -> Tuple[int, float, Any]:
        idx = 0
        leaf_start = self.capacity - 1
        
        while idx < leaf_start:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree): break
            
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
                
        data_idx = idx - self.capacity + 1
        data_idx = max(0, min(data_idx, self.count - 1))
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total_priority(self):
        return max(self.tree[0], 1e-6)

# -------------------------------------------------
# PrioritizedReplayBuffer (Fixed & Complete)
# -------------------------------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, device: str = "cpu"):
        self.tree = FastSumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        self.device = device

    def add(self, data: Any, priority: float = None):
        if priority is None:
            priority = self.max_priority
        
        p = (abs(priority) + 1e-5) ** self.alpha
        self.tree.add(data, p)

    def extend(self, data_list: List[Any]):
        """Added back to fix the AttributeError"""
        for data in data_list:
            self.add(data)

    def sample(self, batch_size: int) -> Tuple[List[Any], torch.Tensor, torch.Tensor]:
        if self.tree.count < batch_size:
            return None, None, None

        batch, tree_idxs, priorities = [], [], []
        segment = self.tree.total_priority / batch_size
        
        # Priority-based sampling
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            tree_idxs.append(idx)
            priorities.append(p)

        # Calculate importance sampling weights
        # Prob = priority / total_priority
        probs = [p / self.tree.total_priority for p in priorities]
        weights = [(self.tree.count * pr) ** -self.beta for pr in probs]
        
        # Max weight normalization for stability
        max_w = max(weights) if weights else 1.0
        weights_t = torch.tensor([w / max_w for w in weights], dtype=torch.float32, device=self.device)
        indices_t = torch.tensor(tree_idxs, dtype=torch.long, device=self.device)

        return batch, indices_t, weights_t

    def update_priorities(self, indices: torch.Tensor, td_errors: Any):
        """Added back to update MCTS error/loss after training steps"""
        # Convert to list for fast iteration
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().tolist()
        
        indices_list = indices.detach().cpu().tolist()
        
        for idx, error in zip(indices_list, td_errors):
            p = (abs(error) + 1e-5) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.count

# -------------------------------------------------
# SimpleReplayBuffer
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