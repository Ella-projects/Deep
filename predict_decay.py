import torch
import torch.nn as nn
import numpy as np

# Normalization constants — set by the notebook after training (10 values each).
MEANS = [None]  # 10 floats
STDS  = [None]  # 10 floats


class OrbitalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 128),  nn.BatchNorm1d(128),  nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256),  nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 256), nn.BatchNorm1d(256),  nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128),  nn.ReLU(),
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
    p = parameters.float()

    # Engineered features (must match training exactly)
    altitude         = p[:, 0:1]
    mass             = p[:, 1:2]
    area             = p[:, 2:3]
    solar            = p[:, 4:5]
    area_to_mass     = area / mass
    log_altitude     = torch.log(altitude)
    log_area_to_mass = torch.log(area_to_mass)
    log_solar        = torch.log(solar)

    x10 = torch.cat([p, area_to_mass, log_altitude, log_area_to_mass, log_solar], dim=1)

    means = torch.tensor(MEANS, dtype=torch.float32).to(device)
    stds  = torch.tensor(STDS,  dtype=torch.float32).to(device)
    x10 = (x10 - means) / stds

    model = OrbitalMLP().to(device)
    model.load_state_dict(torch.load("weights.pkl", map_location=device))
    model.eval()

    with torch.no_grad():
        log_pred = model(x10)
        pred = torch.expm1(log_pred)

    return pred
