
class ExcitationBlock(nn.Module):
    """Excitation Block with Gated Attention."""
    def __init__(self, in_features, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features // reduction)
        self.gate = nn.Sequential(
            nn.Linear(in_features // reduction, in_features),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm1d(in_features)

    def forward(self, x):
        identity = x
        x = self.fc1(x)
        x = F.relu(x)
        gate = self.gate(x)
        x = identity * gate
        x = self.bn(x)
        return x
