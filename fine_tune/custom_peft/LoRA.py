from torch import nn

class ClassicalLoRAAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 4, alpha: float = 8.0, dropout: float = 0.05):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)

    def forward(self, hidden):
        update = self.up(self.dropout(self.down(hidden)))
        return hidden + self.scale * update