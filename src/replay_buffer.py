import torch
import random
from typing import List, Tuple, Any
from src.config import Config

# -------------------------------------------------
# FastSumTree (Pure PyTorch)
# -------------------------------------------------
class FastSumTree:
    def __init__(self, max_size: int, device: str = "cpu"):
        self.max_size = max_size
        self.device = torch.device(device)
        
        # Calculate capacity as power of 2
        self.tree_capacity = 1
        while self.tree_capacity < max_size:
            self.tree_capacity *= 2

        # THE TREE IS NOW A TENSOR
        # We use a float tensor for the tree sums
        self.tree = torch.zeros(2 * self.tree_capacity - 1, dtype=torch.float32, device=self.device)
        
        # We keep data in a standard list because storage requirements for "Data" 
        # vary wildly (objects, dicts, tuples), making tensors hard to manage for storage.
        # However, the *indices* will be managed entirely in tensors.
        self.data = [None] * self.max_size
        
        self.write_ptr = 0
        self.current_count = 0
        
        # Pre-compute tree depth for the sampling loop
        # bit_length() is fast, used to determine loop depth
        self.depth = (self.tree_capacity).bit_length() - 1

    def add_batch(self, data_list: List[Any], priorities: torch.Tensor):
        """
        Adds a batch of transitions. 
        Priorities must be a Tensor (on device preferably).
        """
        count = len(data_list)
        if count == 0:
            return

        # 1. Store Data (Python List - Unavoidable overhead for complex objects)
        end_ptr = self.write_ptr + count
        if end_ptr <= self.max_size:
            self.data[self.write_ptr:end_ptr] = data_list
            write_idxs = torch.arange(self.write_ptr, end_ptr, device=self.device)
        else:
            overflow = end_ptr - self.max_size
            self.data[self.write_ptr:] = data_list[:self.max_size - self.write_ptr]
            self.data[:overflow] = data_list[self.max_size - self.write_ptr:]
            
            # Create wrapping indices
            idx1 = torch.arange(self.write_ptr, self.max_size, device=self.device)
            idx2 = torch.arange(0, overflow, device=self.device)
            write_idxs = torch.cat([idx1, idx2])

        # 2. Update Tree (Pure Tensor Operation)
        # Map write pointer to tree leaf index
        tree_idxs = write_idxs + (self.tree_capacity - 1)
        self.update_batch(tree_idxs, priorities)

        self.write_ptr = end_ptr % self.max_size
        self.current_count = min(self.current_count + count, self.max_size)

    def update_batch(self, tree_idxs: torch.Tensor, priorities: torch.Tensor):
        """
        Updates priorities for specific tree indices and propagates up.
        """
        if not isinstance(priorities, torch.Tensor):
            priorities = torch.tensor(priorities, device=self.device, dtype=torch.float32)
        
        # Ensure minimum priority
        priorities = torch.clamp(priorities, min=Config.MIN_PRIOR)
        
        # 1. Update Leaves
        self.tree.index_put_((tree_idxs,), priorities)

        # 2. Propagate Up (The "Heavy" Lifting)
        # Instead of recursive addition, we recalculate parents layer by layer.
        # This is extremely fast on GPU as it's just vectorized addition.
        
        # We start at the leaves we just modified
        idx = tree_idxs
        
        # Loop strictly for the depth of the tree
        for _ in range(self.depth + 1):
            if idx.numel() == 0: 
                break
                
            # Move to parent: (i - 1) // 2
            idx = (idx - 1) // 2
            
            # Standardize indices (unique) to avoid doing math on the same parent twice
            # Note: unique() can be slow, but essential for correctness in batch updates
            idx = torch.unique(idx)
            
            # Root check (index -1 after the floor division if idx was 0)
            if idx[0] == -1:
                break

            # Calculate children indices
            left = 2 * idx + 1
            right = left + 1
            
            # Sum Logic: Parent = Left + Right
            # We use index_select or direct slicing. Direct slicing with tensor indices is fast.
            # Safety check for right child bounds (though usually guaranteed by structure)
            vals = self.tree[left] + self.tree[right]
            
            self.tree.index_put_((idx,), vals)

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Perform vectorized tree traversal entirely on the Tensor.
        """
        total_p = self.total_priority
        segment = total_p / batch_size
        
        # Generate search values [0, segment, 2*segment, ...] + jitter
        # entirely on device
        r = torch.rand(batch_size, device=self.device)
        s = (torch.arange(batch_size, device=self.device, dtype=torch.float32) + r) * segment

        # Start at root (index 0)
        idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # Traverse the tree
        # This loop runs 'depth' times. Operations inside are fully vectorized.
        for _ in range(self.depth):
            left = 2 * idx + 1
            right = left + 1
            
            # Get values of left children
            # We must mask out indices that might go out of bounds (though typically shouldn't)
            left_vals = self.tree[left]
            
            # Decide direction
            # If s <= left_val: Go Left.
            # If s > left_val:  Subtract left_val, Go Right.
            
            go_right = s > left_vals
            
            # If we go right, we subtract the left value from our search 's'
            # We use torch.where to conditionally subtract
            s = torch.where(go_right, s - left_vals, s)
            
            # Update index: Left is 'left', Right is 'left + 1'
            idx = left + go_right.long()

        # Map back to data indices
        data_idxs = idx - (self.tree_capacity - 1)
        
        # Safety clamp to ensure we don't crash on float precision errors at boundaries
        data_idxs = torch.clamp(data_idxs, 0, self.max_size - 1)
        
        # Get priorities directly from tree (no need to copy to CPU)
        priorities = self.tree[idx]
        
        # We return data_idxs as a list only at the very end to fetch objects
        return idx, priorities, data_idxs.cpu().tolist()

    @property
    def total_priority(self):
        # Return a float, but carefully handle the tensor scalar
        return max(self.tree[0].item(), Config.MIN_PRIOR)

# -------------------------------------------------
# PrioritizedReplayBuffer (Pure PyTorch)
# -------------------------------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, device: str = "cpu"):
        self.alpha = alpha
        self.beta = beta
        self.device = device
        # Initialize tree on the specific device
        self.tree = FastSumTree(capacity, device=device)
        self.max_priority = 1.0

    def add(self, data: Any, priority: float = None):
        """Adds a single transition."""
        if priority is None:
            priority = self.max_priority
        
        # Math on CPU for single scalar is fine, but we prepare for batch add
        p = (abs(priority) + Config.MIN_PRIOR) ** self.alpha
        
        # Wrap in list/tensor
        self.tree.add_batch([data], torch.tensor([p], device=self.device))

    def extend(self, data_list: List[Any]):
        """Adds a list of transitions."""
        count = len(data_list)
        if count == 0: return
        
        # Create priority tensor on device immediately
        p_val = (self.max_priority ** self.alpha) if self.max_priority > 0 else 1.0
        priorities = torch.full((count,), p_val, device=self.device, dtype=torch.float32)
        
        self.tree.add_batch(data_list, priorities)

    def sample(self, batch_size: int) -> Tuple[List[Any], torch.Tensor, torch.Tensor]:
        if self.tree.current_count < batch_size:
            return None, None, None

        # 1. GPU Tree Search
        # Returns indices and priorities already on-device
        tree_idxs, priorities, data_idxs = self.tree.get_batch(batch_size)

        # 2. Retrieve Data Objects
        # This is the only CPU interaction: fetching the complex objects
        batch = [self.tree.data[i] for i in data_idxs]

        # 3. Compute Weights (Pure Tensor Math on Device)
        # No CPU <-> GPU copy needed here
        probs = priorities / self.tree.total_priority
        weights = (self.tree.current_count * probs) ** -self.beta
        
        # Normalize
        weights = weights / weights.max()
        
        # tree_idxs and weights are ALREADY tensors on the correct device.
        return batch, tree_idxs, weights

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """
        Updates priorities. 
        Expects `td_errors` to ideally already be a Tensor on the correct device 
        (which it usually is coming from the loss function).
        """
        # Ensure input is tensor
        if not isinstance(td_errors, torch.Tensor):
            td_errors = torch.tensor(td_errors, device=self.device)
        
        # Detach to ensure we don't break gradients, but keep on device
        td_errors = td_errors.detach()
        
        # Calculate priorities (Vectorized GPU op)
        new_priorities = (torch.abs(td_errors) + Config.MIN_PRIOR) ** self.alpha
        
        # Update max priority (Item access causes a sync, but it's just one float)
        self.max_priority = max(self.max_priority, new_priorities.max().item())
        
        # Batch update tree
        self.tree.update_batch(indices, new_priorities)

    def __len__(self):
        return self.tree.current_count

# SimpleReplayBuffer remains unchanged...
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

def get_replay_buffer(capacity: int, prioritized: bool = True, device: str = "cpu", **kwargs):
    if prioritized:
        return PrioritizedReplayBuffer(capacity, device=device, **kwargs)
    return SimpleReplayBuffer(capacity)