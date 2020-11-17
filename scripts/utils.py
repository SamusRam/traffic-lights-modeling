from dataclasses import dataclass, field
from torch.autograd import Variable

# helping class for beam search
@dataclass(order=True)
class HeapEntry:
    neg_log_prob: float
    lane_idx: Variable = field(compare=False)
    beam_order_idx: int = field(compare=False)

