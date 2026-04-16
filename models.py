import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

# float64 is better for FNOs. Change to 32 for faster training with other models
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
TODO: think about making a function on FNN like self.safety_checks to do before training and eval
to avoid matrix explosions and other common issues.
"""

class FNN(nn.Module):
    def __init__(self, layers, activation=nn.ReLU):
        """
        layers : list of ints, e.g. [2, 50, 50, 1] for a 2-input 2-hidden 1-output net.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1], dtype=dtype, device=device))
        self.activation = activation()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x.to(device))
    
    def loss_fn(self, pred, target):
        return torch.mean((pred - target) ** 2)
    
    def train_model(self, x_train, y_train, epochs=1000, lr=1e-3, track_loss=False, x_val=None, y_val=None, lr_change=None):
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        if x_val is not None:
            x_val = x_val.to(device)
        if y_val is not None:
            y_val = y_val.to(device)

        optimizer = optim.Adam(self.parameters(), lr=lr)  # Using Adam optimizer. parameters is hereded from nn.Module
        if track_loss:
            loss_history = []
            val_loss = []

        for epoch in trange(epochs, desc="Training Epoch: "):
            # Check if is time to change learning rate
            if lr_change is not None and epoch == lr_change[0]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_change[1]

            self.train()  # Set the model to training mode
            optimizer.zero_grad()
            pred = self.forward(x_train)
            loss = self.loss_fn(pred, y_train, )
            if track_loss:
                pred_val = self.forward(x_val)
                val_loss.append(self.loss_fn(pred_val, y_val).item())
                loss_history.append(loss.item())
            loss.backward()  # Compute gradients
            optimizer.step()  # Update the parameters based on the computed gradients

        return loss_history, val_loss if track_loss else None
    
    def mesh_location(self):
        # I want to make a visual rapresentation of where the shallow NN is putting the mesh nodes after training. Therefore I need to extract weights and biases
        weights = []
        biases = []
        for layer in self.layers:
            weights.append(layer.weight.data.cpu().numpy())
            biases.append(layer.bias.data.cpu().numpy())
        return weights, biases
    
class PINN(FNN):
    def __init__(self, layers, activation=nn.Tanh):
        super().__init__(layers, activation)

    def physics_loss(self, points, target_physics):
        x = points.clone().detach().requires_grad_(True)
        phy_behav = target_physics(x)
        y = self.forward(x)
        du_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        du_dxx = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]
        return torch.mean((du_dxx - phy_behav) ** 2)

    def train_model(self, x_train, y_train, epochs=1000, lr=1e-3, target_physics=None, lambda_phy=0.1, lambda_data=0.9, 
                    loss_tracking=False, validation_data=None, lr_scheduler=None):
        
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        optimizer = optim.Adam(self.parameters(), lr=lr)  # Using Adam optimizer. parameters is hereded from nn.Module
        if loss_tracking:
            train_losses = []
            if validation_data is not None:
               x_val, y_val = validation_data
               x_val = x_val.to(device)
               y_val = y_val.to(device)
               val_losses = []
        epochs = trange(epochs, desc="Training Epoch: ")
        for epoch in epochs:
            # Check if is time to change learning rate
            if lr_scheduler is not None and epoch == lr_scheduler['step']:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_scheduler['lr']

            self.train()  # Set the model to training mode
            optimizer.zero_grad()
            pred = self.forward(x_train)
            loss = lambda_data * self.loss_fn(pred, y_train) + lambda_phy * self.physics_loss(x_train, target_physics)
            loss.backward()  # Compute gradients
            optimizer.step()  # Update the parameters based on the computed gradients
            if validation_data is not None:
                with torch.no_grad():
                    y_pred_val = self.forward(x_val)
                    val_loss = self.loss_fn(y_pred_val, y_val)
                if loss_tracking:
                    val_losses.append(val_loss.item())
                    train_losses.append(loss.item())
            elif loss_tracking:
                train_losses.append(loss.item())
            epochs.set_postfix({'train_loss': loss.item(), 'val_loss': val_loss.item() if (validation_data is not None) else 'N/A'})

        if loss_tracking:
            return train_losses, val_losses if validation_data is not None else train_losses
    
class ModelDiscovery(FNN):
    def __init__(self, layers, activation=nn.Tanh):
        """
        In Model discovery f_theta is a function of both x and t, therefore make sure that the input
        dimensions considers also the time variable. Thus input should be of the form [t, x] and not just x.

        Bear in Mind that usually xt[:,0] is the timevariable, the spatial variable is xt[:,1]
        """
        super().__init__(layers, activation)
        # self.A = nn.Parameter(torch.tensor(1, dtype=dtype, device=device, requires_grad=True))
        # self.B = nn.Parameter(torch.tensor(1, dtype=dtype, device=device, requires_grad=True))
        self.C = nn.Parameter(torch.tensor(1, dtype=dtype, device=device, requires_grad=True))
        self.D = nn.Parameter(torch.tensor(1, dtype=dtype, device=device, requires_grad=True))

    def model_loss(self, xt_train, y_train, lam_pde, lam_mse):
        # ---- Compute Gradients ----
        xt = xt_train.clone().detach().requires_grad_(True)
        y = self.forward(xt)
        grads = torch.autograd.grad(y.sum(), xt, create_graph=True)[0]
        dy_dt = grads[:, 0]
        dy_dx = grads[:, 1]
        dy_dxx = torch.autograd.grad(dy_dx.sum(), xt, create_graph=True)[0][:,1]
        
        # ---- PDE and MSE Loss ----
        pde_loss = torch.mean((dy_dt - self.C * dy_dx - self.D * dy_dxx) ** 2)
        mse_loss = self.loss_fn(y, y_train)

        return lam_pde * pde_loss + lam_mse * mse_loss

    def train_model(self, xt_train, y_train, epochs=1000, lr=1e-3, batch_size=256, lam_pde=0.5, lam_mse=0.5,
                    validation_data=None, loss_tracking=False, lr_scheduler=None):
        
        xt_train = xt_train.to(device)
        y_train = y_train.to(device)

        # ---- Safety Check ----
        y_train = y_train.unsqueeze(1) if len(y_train.shape) == 1 else y_train
        assert y_train.shape[1] == 1, "y_train should have only 1 column"

        if validation_data is not None:
            xt_val, y_val = validation_data
            xt_val = xt_val.to(device)
            y_val = y_val.to(device)
            y_val = y_val.unsqueeze(1) if len(y_val.shape) == 1 else y_val
            assert y_val.shape[1] == 1, "y_val should have only 1 column"

        if loss_tracking:
            train_losses = []
            if validation_data is not None:
               val_losses = []

        optimizer = optim.Adam(self.parameters(), lr=lr)  # Using Adam optimizer. parameters is hereded from nn.Module
        # ---- Create batch training ----
        # Consider DataLoader if we want to shuffle the data every epoch
        batch = torch.randperm(xt_train.size(0))[:batch_size]
        xt_train, y_train = xt_train[batch], y_train[batch]

        # ---- Training Loop with Progress Bar ----
        epochs = trange(epochs, desc="Training Epoch: ")
        for epoch in epochs:
            # Check if is time to change learning rate
            if lr_scheduler is not None and epoch == lr_scheduler['step']:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_scheduler['lr']

            self.train()  # Set the model to training mode
            optimizer.zero_grad()
            loss = self.model_loss(xt_train, y_train, lam_pde, lam_mse)
            loss.backward()
            optimizer.step()

            if validation_data is not None:
                with torch.no_grad():
                    y_pred_val = self.forward(xt_val)
                    val_loss = self.loss_fn(y_pred_val, y_val)
                if loss_tracking:
                    val_losses.append(val_loss.item())
                    train_losses.append(loss.item())
            elif loss_tracking:
                train_losses.append(loss.item())

            # Create a postfix to show val and train loss in the progress bar
            epochs.set_postfix({'train_loss': loss.item(), 'val_loss': val_loss.item() if (validation_data is not None) else 'N/A'}) 
        
        if loss_tracking:
            return train_losses, val_losses if validation_data is not None else train_losses

class FNO(nn.Module):
    def __init__(self, modes, latent_width, in_channels=1, out_channels=1, n_layers=4):
        super().__init__()

        self.modes = modes
        self.latent_width = latent_width
        self.in_channels = in_channels # just to generalize, not only 1 input
        self.out_channels = out_channels # just to generalize, not only 1 output
        self.n_layers = n_layers

        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128

        # Lifting: in_channels -> latent_width
        self.enc = nn.Linear(self.in_channels, self.latent_width, dtype=dtype, device=device)

        # Stacked Fourier layers: each has spectral weights + pointwise skip
        self.spectral_weights = nn.ParameterList([
            nn.Parameter(
                (
                    torch.randn(self.latent_width, self.latent_width, self.modes, device=device)
                    + 1j * torch.randn(self.latent_width, self.latent_width, self.modes, device=device)
                ).to(cdtype)
            )
            for _ in range(self.n_layers)
        ])
        self.pointwise = nn.ModuleList([
            nn.Linear(self.latent_width, self.latent_width, dtype=dtype, device=device)
            for _ in range(self.n_layers)
        ])

        # Projection: latent_width -> out_channels
        self.dec = nn.Linear(self.latent_width, self.out_channels, dtype=dtype, device=device)

    def fix_dim(self, t):
            if t.dim() == 2:       # (samples, channels) — single sample
                t = t.unsqueeze(0)                 # -> (1, samples, channels)
            elif t.dim() == 3:     # already (batch, samples, channels)
                pass
            else:
                raise ValueError(
                    f"Expected tensor with 2 or 3 dims, got {t.dim()}"
                )
            return t

    def sanity_check(self, x, y):
        x = self.fix_dim(x).to(device)
        y = self.fix_dim(y).to(device)
        assert x.shape[0] == y.shape[0], "Batch size of x and y must match"
        assert x.shape[1] == y.shape[1], "Number of samples in x and y must match"
        assert x.shape[2] == self.in_channels, f"Input x must have {self.in_channels} channels, got {x.shape[2]}"
        assert y.shape[2] == self.out_channels, f"Output y must have {self.out_channels} channels, got {y.shape[2]}"
        assert self.modes <= x.shape[1] // 2 + 1, "Number of modes must be less than or equal to half the number of samples plus one (due to rfft)"
        return x, y

    def forward(self, x):
        """
        x:       [batch (b), samples (s), in_channels]
        output:  [b, s, out_channels]

        Architecture per Fourier layer:
            x_out = GELU( spectral_conv(x) + pointwise(x) )
        The spectral_conv keeps only the lowest `modes` frequencies.
        The pointwise linear is the local skip connection.
        Last layer has no activation (feeds into the decoder).
        """
        n_samples = x.shape[1]

        # Lifting
        x_enc = self.enc(x)  # [b, s, latent_width]

        # Stacked Fourier layers
        for i in range(self.n_layers):
            # Spectral branch
            x_ft = torch.fft.rfft(x_enc, dim=1)
            x_ft_low = torch.einsum(
                "bmn, ndm -> bmd",
                x_ft[:, :self.modes, :],
                self.spectral_weights[i],
            )
            x_ft_out = torch.zeros_like(x_ft)
            x_ft_out[:, :self.modes, :] = x_ft_low
            x_spectral = torch.fft.irfft(x_ft_out, n=n_samples, dim=1)

            # Pointwise skip branch
            x_local = self.pointwise[i](x)

            # Combine + activate (no activation on last layer)
            x = x_spectral + x_local
            if i < self.n_layers - 1:
                x = torch.nn.functional.gelu(x)

        # Projection
        return self.dec(x)

    def loss(self, gt, pred):
        return torch.mean(torch.mean((gt - pred) ** 2, dim=(1,2)))

    def train_model(self, x_train, y_train, epochs=1000, lr=1e-3, validation_data=None, loss_tracking=False, lr_scheduler=None):
        """
        Trains the model on the provided training data.
        Accepts training inputs with shape (a, b, c) where:
        - a is the batch size
        - b is the number of samples (e.g., arc points)
        - c is the number of input/output channels
        """
        
        x_train, y_train = self.sanity_check(x_train, y_train)
        if validation_data is not None:
            x_val, y_val = validation_data
            x_val, y_val = self.sanity_check(x_val, y_val)
        if loss_tracking:
            train_losses = []
            val_losses = []

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        epochs = trange(epochs, desc="Training Epoch: ")
        for epoch in epochs:
            if lr_scheduler is not None and epoch == lr_scheduler['step']:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_scheduler['lr']
            pred = self.forward(x_train)
            loss = self.loss(y_train, pred)
            if loss_tracking:
                train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if validation_data is not None:
                with torch.no_grad():
                    val_pred = self.forward(x_val)
                    val_loss = self.loss(y_val, val_pred)
                if loss_tracking:
                    val_losses.append(val_loss.item())

            epochs.set_postfix({'train_loss': loss.item(), 'val_loss': val_loss.item() if (validation_data is not None) else 'N/A'})
        if loss_tracking and validation_data is not None:
            return train_losses, val_losses
        elif loss_tracking:
            return train_losses

    def predict(self, x):
        """
        Predict the output for a given input x.
        Accepts inputs with shape (b, c) or (a, b, c) where:
        - b is the number of samples
        - c is the number of input channels
        """
        x = self.fix_dim(x).to(device)
        with torch.no_grad():
            prediction = self.forward(x)
            return prediction.squeeze(0) if prediction.shape[0] == 1 else prediction