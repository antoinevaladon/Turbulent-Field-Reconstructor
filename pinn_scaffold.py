import random
import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# 1. GROUND TRUTH FIELD
# ---------------------------------------------------------------------------
def true_field(x, y):
    """Exact harmonic function used as ground truth. Real part of e^(x+iy), so it satisfies Laplace's equation."""
    return torch.exp(x) * torch.cos(y)


# ---------------------------------------------------------------------------
# 2. SPARSE DATA POINTS (simulated "sensor" measurements)
# ---------------------------------------------------------------------------
def make_data_points(n_points, domain, seed=0):
    """Samples n_points random locations and records their true u values. Simulates sparse real-world measurements."""
    (xmin, xmax), (ymin, ymax) = domain
    random.seed(seed)  # reproducible sampling

    x_list = []
    y_list = []
    u_list = []

    for _ in range(n_points):
        x_component = random.uniform(xmin, xmax)
        y_component = random.uniform(ymin, ymax)
        u_component = true_field(
            torch.tensor(x_component), torch.tensor(y_component)
        )  # the "measured" value at this point

        x_list.append(x_component)
        y_list.append(y_component)
        u_list.append(u_component)

    x_data = torch.tensor(x_list, dtype=torch.float32).reshape(-1, 1)  # (N,1) shape for Linear layers
    y_data = torch.tensor(y_list, dtype=torch.float32).reshape(-1, 1)
    u_data = torch.tensor(u_list, dtype=torch.float32).reshape(-1, 1)

    return x_data, y_data, u_data


# ---------------------------------------------------------------------------
# 3. COLLOCATION POINTS (physics-only, no known u)
# ---------------------------------------------------------------------------
def make_collocation_points(n_points, domain, seed=1):
    """Samples n_points unlabeled locations across the domain for enforcing the physics residual. No known u values here."""
    (xmin, xmax), (ymin, ymax) = domain
    torch.manual_seed(seed)

    x_f = torch.rand(n_points, 1) * (xmax - xmin) + xmin  # scale [0,1) into domain
    y_f = torch.rand(n_points, 1) * (ymax - ymin) + ymin

    x_f.requires_grad_(True)  # set after creation so these stay leaf tensors
    y_f.requires_grad_(True)

    return x_f, y_f


# ---------------------------------------------------------------------------
# 4. THE NETWORK
# ---------------------------------------------------------------------------
class PINN(nn.Module):
    """Small MLP mapping (x, y) -> scalar u. Uses tanh activations so the output stays twice-differentiable."""

    def __init__(self, hidden_dim=32, n_hidden_layers=3):
        super().__init__()

        layers = []
        layers.append(nn.Linear(2, hidden_dim))  # input: x, y
        layers.append(nn.Tanh())

        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))  # output: u, no activation

        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)  # (N,1)+(N,1) -> (N,2)
        u = self.net(inputs)
        return u


# ---------------------------------------------------------------------------
# 5. PHYSICS RESIDUAL (the actual "PINN" part)
# ---------------------------------------------------------------------------
def laplace_residual(model, x_f, y_f):
    """Computes u_xx + u_yy at each collocation point via autograd. Should be pushed toward zero during training."""
    u = model(x_f, y_f)

    u_x = torch.autograd.grad(
        u, x_f, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]  # du/dx
    u_xx = torch.autograd.grad(
        u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]  # d^2u/dx^2

    u_y = torch.autograd.grad(
        u, y_f, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]  # du/dy
    u_yy = torch.autograd.grad(
        u_y, y_f, grad_outputs=torch.ones_like(u_y), create_graph=True
    )[0]  # d^2u/dy^2

    residual = u_xx + u_yy
    return residual


def divergence_residual(model, x_f, y_f, z_f):
    """Computes div(B) = dBx/dx + dBy/dy + dBz/dz at each collocation point. Enforces Gauss's law for magnetism (not yet wired into training)."""
    B = model(x_f, y_f, z_f)  # (N,3) -> Bx, By, Bz

    Bx = B[:, 0:1]  # slice, not index, to keep (N,1) shape
    By = B[:, 1:2]
    Bz = B[:, 2:3]

    dBx_dx = torch.autograd.grad(
        Bx, x_f, grad_outputs=torch.ones_like(Bx), create_graph=True
    )[0]
    dBy_dy = torch.autograd.grad(
        By, y_f, grad_outputs=torch.ones_like(By), create_graph=True
    )[0]
    dBz_dz = torch.autograd.grad(
        Bz, z_f, grad_outputs=torch.ones_like(Bz), create_graph=True
    )[0]

    divergence = dBx_dx + dBy_dy + dBz_dz
    return divergence


def gauss_loss(model, x_f, y_f, z_f):
    """Mean squared divergence, the vector analog of physics_loss. Not yet added into total_loss / train()."""
    div = divergence_residual(model, x_f, y_f, z_f)
    return torch.mean(div ** 2)


# ---------------------------------------------------------------------------
# 6. LOSSES
# ---------------------------------------------------------------------------
def data_loss(model, x_data, y_data, u_data):
    """Mean squared error between predicted and known u at the sparse data points."""
    u_pred = model(x_data, y_data)
    return torch.mean((u_pred - u_data) ** 2)


def physics_loss(model, x_f, y_f):
    """Mean squared Laplace residual across collocation points."""
    residual = laplace_residual(model, x_f, y_f)
    return torch.mean(residual ** 2)


# ---------------------------------------------------------------------------
# 7. TRAINING LOOP
# ---------------------------------------------------------------------------
def train(model, x_data, y_data, u_data, x_f, y_f,
          n_epochs=5000, lr=1e-3, lam=1.0):
    """Standard Adam loop minimizing data_loss + lam * physics_loss. Logs both loss terms separately to check neither is being ignored."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"data_loss": [], "physics_loss": [], "total_loss": []}

    for epoch in range(n_epochs):
        optimizer.zero_grad()  # clear old gradients before this step

        d_loss = data_loss(model, x_data, y_data, u_data)
        p_loss = physics_loss(model, x_f, y_f)
        total = d_loss + lam * p_loss

        total.backward()  # compute gradient of total w.r.t. every weight
        optimizer.step()  # nudge weights

        history["data_loss"].append(d_loss.item())
        history["physics_loss"].append(p_loss.item())
        history["total_loss"].append(total.item())

        if epoch % 500 == 0 or epoch == n_epochs - 1:
            print(
                f"epoch {epoch:5d} | "
                f"data_loss {d_loss.item():.6f} | "
                f"physics_loss {p_loss.item():.6f} | "
                f"total {total.item():.6f}"
            )

    return model, history


# ---------------------------------------------------------------------------
# 8. EVALUATION
# ---------------------------------------------------------------------------
def evaluate_on_grid(model, domain, resolution=100):
    """Predicts u on a fine grid across the domain and compares to true_field. Shows how well the reconstruction did away from the sparse points."""
    (xmin, xmax), (ymin, ymax) = domain

    xs = torch.linspace(xmin, xmax, resolution)
    ys = torch.linspace(ymin, ymax, resolution)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")

    x_flat = grid_x.reshape(-1, 1)
    y_flat = grid_y.reshape(-1, 1)

    with torch.no_grad():  # no training happening, skip gradient tracking
        u_pred = model(x_flat, y_flat)
        u_true = true_field(x_flat, y_flat)

    u_pred_grid = u_pred.reshape(resolution, resolution)
    u_true_grid = u_true.reshape(resolution, resolution)

    abs_error = torch.abs(u_pred_grid - u_true_grid)
    print(f"max abs error: {abs_error.max().item():.6f}")
    print(f"mean abs error: {abs_error.mean().item():.6f}")

    return grid_x, grid_y, u_pred_grid, u_true_grid


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    domain = ((-1.0, 1.0), (-1.0, 1.0))

    x_data, y_data, u_data = make_data_points(n_points=20, domain=domain, seed=0)
    x_f, y_f = make_collocation_points(n_points=500, domain=domain, seed=1)

    model = PINN(hidden_dim=32, n_hidden_layers=3)

    model, history = train(
        model, x_data, y_data, u_data, x_f, y_f,
        n_epochs=5000, lr=1e-3, lam=1.0
    )

    grid_x, grid_y, u_pred_grid, u_true_grid = evaluate_on_grid(
        model, domain, resolution=100
    )

    # save everything a separate script needs to plot, without re-running training
    import os
    SHARED_DIR = "pinn_shared"  # both scripts read/write here
    os.makedirs(SHARED_DIR, exist_ok=True)

    torch.save({
        "grid_x": grid_x,
        "grid_y": grid_y,
        "u_pred_grid": u_pred_grid,
        "u_true_grid": u_true_grid,
        "x_data": x_data,
        "y_data": y_data,
        "u_data": u_data,
        "history": history,
    }, os.path.join(SHARED_DIR, "pinn_results.pt"))

    torch.save(model.state_dict(), os.path.join(SHARED_DIR, "pinn_weights.pt"))  # trained weights, for live re-querying
