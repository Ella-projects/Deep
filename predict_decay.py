import torch
import torch.nn as nn

# Normalization constants set by the notebook after training.
# Run the Q4 training cell; it will regenerate this file with the correct values.
MEANS = [None]  # 7 floats (StandardScaler means over 7 features)
STDS  = [None]  # 7 floats (StandardScaler scales over 7 features)


class OrbitalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 128),  nn.BatchNorm1d(128),  nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def predict(parameters):
    """
    parameters: B x 6 float tensor
        [initial_altitude_km, satellite_mass_kg, cross_sectional_area_m2,
         orbital_eccentricity, solar_activity_index, drag_coefficient]
    Returns: B x 1 tensor of predicted decay times in days.
    """
    device = torch.device("cuda" if parameters.is_cuda else "cpu")

    # Compute area-to-mass ratio and append as 7th feature
    area_to_mass = parameters[:, 2:3] / parameters[:, 1:2]
    x7 = torch.cat([parameters.float(), area_to_mass], dim=1)

    means = torch.tensor(MEANS, dtype=torch.float32).to(device)
    stds  = torch.tensor(STDS,  dtype=torch.float32).to(device)
    x7 = (x7 - means) / stds

    model = OrbitalMLP().to(device)
    model.load_state_dict(torch.load("weights.pkl", map_location=device))
    model.eval()

    with torch.no_grad():
        log_pred = model(x7)
        pred = torch.expm1(log_pred)  # inverse of log1p applied during training

    return pred
