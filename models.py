import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange


dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
#  FEATURE EMBEDDING BASE CLASS
# ─────────────────────────────────────────────
class FeatureEmbedding(nn.Module):
    """
    Base class for input feature embeddings.
    All embeddings must implement forward() and expose output_dim.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim  # FNN uses this to set first layer size automatically

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")


# ─────────────────────────────────────────────
#  FOURIER EMBEDDING
# ─────────────────────────────────────────────
class FourierEmbedding(FeatureEmbedding):
    """
    Fourier feature embedding to combat spectral bias.
    
    Transforms input x into [sin(Bx), cos(Bx)] where B is a matrix of frequencies.

    Frequency collapse is avoided by:
      1. Evenly spaced init  → guarantees coverage of target frequency range
      2. Learnable B         → frequencies migrate toward what the data needs
      3. L1 warm start       → penalty only activates after l1_start_epoch,
                               giving real frequencies time to establish themselves
    
    Args:
        input_dim      : input dimensions (1 for 1D problems)
        embed_dim      : number of frequency components (output is 2*embed_dim)
        scale          : max frequency in linspace init — set near your function's frequency
        learnable      : if True, B is a trainable parameter; if False, it is frozen
        l1_strength    : L1 penalty weight on B (only if learnable=True)
        l1_start_epoch : epoch after which L1 penalty activates (warm start)
    """
    def __init__(self, input_dim, embed_dim, scale=10.0,
                 learnable=True, l1_strength=1e-4, l1_start_epoch=1000):
        super().__init__(input_dim, output_dim=2 * embed_dim + input_dim)  # +1 for raw x        self.embed_dim = embed_dim
        self.learnable = learnable
        self.l1_strength = l1_strength
        self.l1_start_epoch = l1_start_epoch

        # Evenly spaced init — guarantees the target frequency range is covered
        # from the start, avoiding the random-init shadowing problem
        B = torch.linspace(0.1, scale, embed_dim).unsqueeze(0).expand(input_dim, -1).clone()

        if learnable:
            self.B = nn.Parameter(B)        # frequencies migrate during training
        else:
            self.register_buffer('B', B)    # frozen

    def forward(self, x):
        x_proj = x @ self.B.to(x.dtype)
        features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return torch.cat([x, features], dim=-1)  # ← append raw x

    def l1_penalty(self, epoch):
        """Returns L1 penalty on B, zero before warm start epoch."""
        if not self.learnable or epoch < self.l1_start_epoch:
            return 0.0
        return self.l1_strength * torch.sum(torch.abs(self.B))


# ─────────────────────────────────────────────
#  FNN
# ─────────────────────────────────────────────
class FNN(nn.Module):
    def __init__(self, layers, activation=nn.ReLU, embedding=None):
        """
        layers    : list of ints e.g. [1, 256, 256, 1].
                    If embedding is provided, first element is ignored and
                    replaced automatically with embedding.output_dim.
        embedding : a FeatureEmbedding instance or None
        """
        super().__init__()
        self.embedding = embedding

        # If an embedding is provided, override the first layer input size
        if embedding is not None:
            layers = [embedding.output_dim] + layers[1:]

        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = activation()

    def forward(self, x):
        if self.embedding is not None:
            x = self.embedding(x)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def loss_fn(self, pred, target):
        return torch.mean((pred - target) ** 2)

    def train_model(self, x_train, y_train, epochs=1000, lr=1e-3,
                    track_loss=False, x_val=None, y_val=None, lr_change=None):
        optimizer = optim.Adam([
            {'params': self.embedding.parameters(), 'lr': lr * 0.1},
            {'params': self.layers.parameters(), 'lr': lr}
        ], lr=lr)
        if track_loss:
            loss_history = []
            val_loss = []

        for epoch in trange(epochs, desc="Training Epoch: "):
            if lr_change is not None and epoch == lr_change[0]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_change[1]

            self.train()
            optimizer.zero_grad()
            pred = self.forward(x_train)
            loss = self.loss_fn(pred, y_train)

            # L1 penalty on embedding frequencies (warm start handled inside)
            if self.embedding is not None:
                loss = loss + self.embedding.l1_penalty(epoch)

            if track_loss:
                pred_val = self.forward(x_val)
                val_loss.append(self.loss_fn(pred_val, y_val).item())
                loss_history.append(loss.item())

            loss.backward()
            optimizer.step()

        return loss_history, val_loss if track_loss else None

    def mesh_location(self):
        weights, biases = [], []
        for layer in self.layers:
            weights.append(layer.weight.data.cpu().numpy())
            biases.append(layer.bias.data.cpu().numpy())
        return weights, biases


# ─────────────────────────────────────────────
#  PINN
# ─────────────────────────────────────────────
class PINN(FNN):
    def __init__(self, layers, activation=nn.Tanh, embedding=None):
        super().__init__(layers, activation, embedding)

    def physics_loss(self, points, target_physics):
        x = points.clone().detach().requires_grad_(True)
        phy_behav = target_physics(x)
        y = self.forward(x)
        du_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        du_dxx = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]
        pde_loss = torch.mean((du_dxx - phy_behav) ** 2)

        x_left  = torch.tensor([[-4.0]], dtype=torch.float32, device=points.device)
        x_right = torch.tensor([[4.0]],  dtype=torch.float32, device=points.device)
        bc_loss = (self.forward(x_left) - 2) ** 2 + (self.forward(x_right) - 2) ** 2

        return pde_loss +  2*bc_loss.mean()


    def train_model(self, x_train, y_train, epochs=1000, lr=1e-3,
                    track_loss=False, x_val=None, y_val=None, lr_change=None,
                    target_physics=None, lambda_phy=0.1, lambda_data=0.9):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        if track_loss:
            loss_history = []
            val_loss = []

        for epoch in trange(epochs, desc="Training Epoch: "):
            if lr_change is not None and epoch == lr_change[0]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_change[1]

            self.train()
            optimizer.zero_grad()
            pred = self.forward(x_train)
            loss = lambda_data * self.loss_fn(pred, y_train) + \
                   lambda_phy * self.physics_loss(x_train, target_physics)

            # L1 penalty on embedding frequencies (warm start handled inside)
            if self.embedding is not None:
                loss = loss + self.embedding.l1_penalty(epoch)

            if track_loss:
                pred_val = self.forward(x_val)
                val_loss.append(self.loss_fn(pred_val, y_val).item())
                loss_history.append(loss.item())

            loss.backward()
            optimizer.step()

        return loss_history, val_loss if track_loss else None