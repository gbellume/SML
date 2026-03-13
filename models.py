import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange


dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FNN(nn.Module):
    def __init__(self, layers, activation=nn.ReLU):
        """
        layers : list of ints, e.g. [2, 50, 50, 1] for a 2-input 2-hidden 1-output net.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = activation()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def loss_fn(self, pred, target):
        return torch.mean((pred - target) ** 2)
    
    def train_model(self, x_train, y_train, epochs=1000, lr=1e-3, track_loss=False, x_val=None, y_val=None, lr_change=None):
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

    def train_model(self, x_train, y_train, epochs=1000, lr=1e-3, track_loss=False, x_val=None, y_val=None, lr_change=None, target_physics=None, lambda_phy=0.1, lambda_data=0.9):
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
            loss = lambda_data * self.loss_fn(pred, y_train) + lambda_phy * self.physics_loss(x_train, target_physics)
            if track_loss:
                pred_val = self.forward(x_val)
                val_loss.append(self.loss_fn(pred_val, y_val).item())
                loss_history.append(loss.item())
            loss.backward()  # Compute gradients
            optimizer.step()  # Update the parameters based on the computed gradients

        return loss_history, val_loss if track_loss else None