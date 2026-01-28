# =============================
# TORCH VERSION: NaN-SAFE
# =============================

import torch
from typing import List, Tuple, Any
from src.backgammon.config import Config


# -------------------------------------------------
# FastSumTree (TORCH)
# -------------------------------------------------
class FastSumTree:
    def __init__(self, capacity: int, device: str = "cpu"):
        self.device = device
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2

        self.tree = torch.zeros(2 * self.capacity - 1, dtype=torch.float32, device=self.device)
        self.data = [None] * self.capacity
        self.write_idx = 0
        self.count = 0

    def add(self, data: Any, priority: float):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def update(self, idx: int, priority: float):
        priority = float(priority)
        if not torch.isfinite(torch.tensor(priority)):
            priority = 0.0
        priority = max(priority, Config.MIN_PRIOR)

        change = priority - self.tree[idx].item()
        self.tree[idx] = priority

        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def update_batch(self, indices: torch.Tensor, priorities: torch.Tensor):
        priorities = torch.nan_to_num(priorities, nan=0.0, posinf=1.0, neginf=1.0)
        priorities = torch.clamp(priorities, Config.MIN_PRIOR, Config.MAX_PRIOR)

        changes = priorities - self.tree[indices]
        self.tree[indices] = priorities

        while True:
            indices = (indices - 1) // 2
            self.tree.index_add_(0, indices, changes)
            if indices[0] == 0:
                break

    def get(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.zeros(len(s), dtype=torch.int64, device=self.device)
        leaf_start = self.capacity - 1

        while idx[0] < leaf_start:
            left = 2 * idx + 1
            right = left + 1
            left_vals = self.tree[left]

            go_right = s > left_vals
            s[go_right] -= left_vals[go_right]

            idx[go_right] = right[go_right]
            idx[~go_right] = left[~go_right]

        data_idxs = idx - self.capacity + 1
        data_idxs = torch.clamp(data_idxs, 0, self.count - 1)
        return idx, self.tree[idx], data_idxs

    @property
    def total_priority(self) -> float:
        total = float(self.tree[0].item())
        if not torch.isfinite(torch.tensor(total)) or total <= 0.0:
            return Config.MIN_PRIOR
        return total


# -------------------------------------------------
# PrioritizedReplayBuffer (TORCH)
# -------------------------------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001, device: str = "cpu"):
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.tree = FastSumTree(capacity, device=device)
        self.max_priority = 1.0
        self.device = device

    def add(self, data: Any, priority: float = None):
        if priority is None:
            priority = self.max_priority

        priority = float(priority)
        if not torch.isfinite(torch.tensor(priority)):
            priority = self.max_priority

        priority = max(priority, 1e-6) ** self.alpha
        self.tree.add(data, priority)

    def extend(self, data_list: List[Any]):
        for data in data_list:
            self.add(data)

    def sample(self, batch_size: int) -> Tuple[List[Any], torch.Tensor, torch.Tensor]:
        if self.tree.count < batch_size:
            return [], torch.tensor([]), torch.tensor([])

        total_p = self.tree.total_priority
        segment = total_p / batch_size

        s = torch.rand(batch_size, device=self.device) * segment
        s += torch.arange(batch_size, device=self.device) * segment

        tree_idxs, priorities, data_idxs = self.tree.get(s)

        probs = priorities / total_p

        self.beta = min(1.0, self.beta + self.beta_increment)

        weights = (self.tree.count * probs) ** -self.beta
        weights /= max(weights.max().item(), Config.MIN_PRIOR)

        if not torch.isfinite(weights).all():
            return [], torch.tensor([]), torch.tensor([])

        batch = [self.tree.data[i] for i in data_idxs.tolist()]
        return batch, tree_idxs, weights

    def update_priorities(self, indices: List[int], td_errors: List[float]):
        indices = torch.tensor(indices, device=self.device)
        td_errors = torch.tensor(td_errors, device=self.device)
        td_errors = torch.nan_to_num(td_errors, nan=0.0, posinf=1.0, neginf=1.0)

        priorities = (td_errors.abs() + Config.KL_EPSILON) ** self.alpha
        priorities = torch.clamp(priorities, Config.MIN_PRIOR, Config.MAX_PRIOR)

        self.max_priority = max(self.max_priority, priorities.max().item())
        self.tree.update_batch(indices, priorities)

    def __len__(self) -> int:
        return self.tree.count


# -------------------------------------------------
# SimpleReplayBuffer (TORCH, minimal)
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
        if self.size < batch_size:
            return [], torch.tensor([]), torch.tensor([])

        indices = torch.randint(0, self.size, (batch_size,))
        batch = [self.data[i] for i in indices.tolist()]
        weights = torch.ones(batch_size, dtype=torch.float32)
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        pass

    def __len__(self):
        return self.size


# -------------------------------------------------
# Factory (UNCHANGED)
# -------------------------------------------------
def get_replay_buffer(capacity: int, prioritized: bool = True, **kwargs):
    if prioritized:
        return PrioritizedReplayBuffer(capacity, **kwargs)
    return SimpleReplayBuffer(capacity)
