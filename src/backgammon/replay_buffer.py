import numpy as np
from typing import List, Tuple, Any, Optional

# Mocking Config
try:
    from src.backgammon.config import Config
except ImportError:
    class Config:
        KL_EPSILON = 1e-6

class FastSumTree:
    """
    A vectorized SumTree implementation using Numpy.
    Capacity is padded to the next power of 2 for efficient tree traversal.
    """
    def __init__(self, capacity: int):
        # Calculate next power of 2 for efficient tree properties
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
            
        self.tree = np.zeros(2 * self.capacity - 1)
        self.data = np.array([None] * self.capacity, dtype=object)
        self.write_idx = 0
        self.count = 0

    def add(self, data: Any, priority: float):
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def update(self, idx: int, priority: float):
        """Update single priority."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def update_batch(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Vectorized update for a batch of indices.
        Much faster than calling update() in a loop.
        """
        # Calculate changes
        changes = priorities - self.tree[indices]
        self.tree[indices] = priorities
        
        # Propagate changes up the tree
        self._propagate_batch(indices, changes)

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        while parent != 0:
            parent = (parent - 1) // 2
            self.tree[parent] += change

    def _propagate_batch(self, indices: np.ndarray, changes: np.ndarray):
        """Vectorized propagation."""
        # We need to be careful about duplicate parents in a batch update.
        # np.add.at is used to handle unbuffered in-place operations safely
        # when indices are repeated.
        
        # Current indices
        current_idxs = indices
        
        while True:
            # Calculate parents
            current_idxs = (current_idxs - 1) // 2
            
            # Identify unique parents to avoid double counting if doing simple addition
            # But np.add.at handles duplicates correctly
            np.add.at(self.tree, current_idxs, changes)
            
            # Break if we reached the root (index 0)
            # We check if all indices are 0 (technically they converge to 0 at different speeds
            # but usually batch sampling happens at same depth).
            # Optimization: The root is 0. If min(current_idxs) == 0, we might be done,
            # but some might still be deeper.
            # Safest check:
            if current_idxs[0] == 0: 
                break

    def get(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized tree retrieval. 
        s: array of values to retrieve
        Returns: (tree_indices, priorities, data_indices)
        """
        idx = np.zeros(len(s), dtype=np.int64)
        
        # Traverse down the tree
        # Since capacity is power of 2, depth is log2(capacity)
        # We can hardcode the loop size or use while
        
        # To make it truly fast, we loop through layers
        # Left child is 2*i + 1
        
        # We iterate until we reach leaf nodes. 
        # Leaf nodes start at self.capacity - 1
        leaf_start = self.capacity - 1
        
        # While indices are strictly internal nodes
        while idx[0] < leaf_start:
            left = 2 * idx + 1
            right = left + 1
            
            # Get values of left children
            left_vals = self.tree[left]
            
            # Where s > left_child, go right
            go_right = s > left_vals
            
            # If going right, subtract left value from s
            s[go_right] -= left_vals[go_right]
            
            # Update indices
            idx[go_right] = right[go_right]
            idx[~go_right] = left[~go_right]

        # Map tree index to data index
        data_idxs = idx - self.capacity + 1
        
        # Handling numerical instability (rounding errors)
        # Ensure we don't go out of bounds of existing data
        # (Though with power of 2 padding, tree usually has zeros at end)
        data_idxs = np.clip(data_idxs, 0, self.count - 1)
        
        return idx, self.tree[idx], data_idxs

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    Optimized Prioritized Experience Replay.
    Uses FastSumTree for O(log N) sampling and updates.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.tree = FastSumTree(capacity)
        self.max_priority = 1.0

    def add(self, data: Any, priority: float = None):
        if priority is None:
            priority = self.max_priority
        else:
            priority = priority ** self.alpha
            
        self.tree.add(data, priority)

    def extend(self, data_list: List[Any]):
        """Add multiple experiences efficiently."""
        for data in data_list:
            self.add(data)

    def sample(self, batch_size: int) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        if self.tree.count < batch_size:
            return [], np.array([]), np.array([])
        
        # Stratified sampling logic (vectorized)
        total_p = self.tree.total_priority
        segment = total_p / batch_size
        
        # Generate 's' values: One random number per segment
        # np.arange creates the segment starts, random adds offset
        s = np.random.uniform(0, segment, size=batch_size) + np.arange(batch_size) * segment
        
        # Vectorized tree search
        tree_idxs, priorities, data_idxs = self.tree.get(s)
        
        # Retrieve data
        # Note: self.tree.data is a numpy array of objects, allowing array indexing
        batch = self.tree.data[data_idxs]
        
        # Calculate IS Weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Probability = priority / total_priority
        # Avoid division by zero with small epsilon if needed, but tree nodes shouldn't be 0
        sampling_probabilities = priorities / total_p
        
        # Compute weights: (N * P)^(-beta)
        weights = np.power(self.tree.count * sampling_probabilities, -self.beta)
        weights /= weights.max() # Normalize
        
        return batch, tree_idxs, weights

    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on TD errors using vectorized operations."""
        # Convert inputs to numpy arrays if they aren't already
        indices = np.array(indices)
        td_errors = np.array(td_errors)
        
        # Calculate new priorities
        priorities = (np.abs(td_errors) + Config.KL_EPSILON) ** self.alpha
        
        # Update max priority for new elements
        self.max_priority = max(self.max_priority, np.max(priorities))
        
        # Batch update the tree
        self.tree.update_batch(indices, priorities)

    def __len__(self):
        return self.tree.count


class SimpleReplayBuffer:
    """
    Vectorized Uniform Replay Buffer.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        # Pre-allocate numpy array for object references
        self.data = np.array([None] * capacity, dtype=object)
        self.write_idx = 0
        self.size = 0
    
    def add(self, data: Any):
        self.data[self.write_idx] = data
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def extend(self, data_list: List[Any]):
        for data in data_list:
            self.add(data)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, List[int], np.ndarray]:
        if self.size < batch_size:
            return [], [], []
            
        # Fast sampling using numpy
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = self.data[indices]
        
        # Weights are all 1.0
        weights = np.ones(batch_size, dtype=np.float32)
        
        return batch, indices, weights
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        pass
    
    def __len__(self):
        return self.size

def get_replay_buffer(capacity: int, prioritized: bool = True, **kwargs):
    if prioritized:
        return PrioritizedReplayBuffer(capacity, **kwargs)
    return SimpleReplayBuffer(capacity)