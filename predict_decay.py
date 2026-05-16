import torch
import torch.nn as nn
import numpy as np
import os

_DIR = os.path.dirname(os.path.abspath(__file__))


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

    area_to_mass     = p[:, 2:3] / p[:, 1:2]
    log_altitude     = torch.log(p[:, 0:1])
    log_area_to_mass = torch.log(area_to_mass)
    log_solar        = torch.log(p[:, 4:5])

    x10 = torch.cat([p, area_to_mass, log_altitude, log_area_to_mass, log_solar], dim=1)

    checkpoint = torch.load(os.path.join(_DIR, "weights.pkl"), map_location=device)
    means = torch.tensor(checkpoint["means"], dtype=torch.float32).to(device)
    stds  = torch.tensor(checkpoint["stds"],  dtype=torch.float32).to(device)
    x10 = (x10 - means) / stds

    preds = []
    for state_f16 in checkpoint["models"]:
        m = OrbitalMLP().to(device)
        m.load_state_dict({k: v.float() for k, v in state_f16.items()})
        m.eval()
        with torch.no_grad():
            preds.append(torch.expm1(m(x10)))

    return torch.stack(preds).mean(0)
