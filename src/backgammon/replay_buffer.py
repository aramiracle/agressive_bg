import random
import numpy as np
from collections import deque
from src.backgammon.config import Config


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with Sum Tree."""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # prioritization exponent
        self.beta = beta    # importance sampling exponent
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0
    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def _total(self):
        return self.tree[0]
    
    def add(self, data, priority=None):
        if priority is None:
            priority = self.max_priority
        
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self._update(idx, priority ** self.alpha)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def _update(self, idx, priority):
        # Ensure priority is never exactly 0 to keep the tree searchable
        priority = max(priority, Config.KL_EPSILON)
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def sample(self, batch_size):
        if self.size < batch_size:
            return [], [], []

        batch, indices, priorities = [], [], []
        # Total is the sum of all priorities for valid data only
        total = self._total()
        segment = total / batch_size
        
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx = self._retrieve(0, s)
            data_idx = idx - self.capacity + 1
            
            # If we hit an uninitialized slot (should not happen with size check)
            # we force it to the most recent valid entry
            if self.data[data_idx] is None:
                data_idx = (self.write_idx - 1) % self.capacity
                idx = data_idx + self.capacity - 1

            batch.append(self.data[data_idx])
            indices.append(idx)
            priorities.append(self.tree[idx])

        # Importance sampling weights calculation
        probs = np.array(priorities) / total
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + Config.KL_EPSILON) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self._update(idx, priority)
    
    def extend(self, data_list):
        for data in data_list:
            self.add(data)
    
    def __len__(self):
        return self.size


class SimpleReplayBuffer:
    """Simple uniform replay buffer using deque."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, data):
        self.buffer.append(data)
    
    def extend(self, data_list):
        self.buffer.extend(data_list)
    
    def sample(self, batch_size):
        # Direct sampling from deque without conversion
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        batch = [self.buffer[i] for i in indices]
        weights = [1.0] * len(batch)
        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        pass  # No-op for uniform buffer
    
    def __len__(self):
        return len(self.buffer)


def get_replay_buffer(capacity, prioritized=True, **kwargs):
    """Factory function to get appropriate replay buffer."""
    if prioritized:
        return PrioritizedReplayBuffer(capacity, **kwargs)
    return SimpleReplayBuffer(capacity)