from __future__ import division
from __future__ import unicode_literals
import torch
import datetime

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error as MAE

from dimenet.model.dimenet import DimeNet


# validation inference loop
def val_inference(model, loader):
    y_preds = []
    y_actual = []

    for data in tqdm(loader, leave=False):
        data = data.to(device)

        # get prediction
        with torch.no_grad():
            pred = model(data.z, data.pos, data.batch)

        y_preds.append(pred.view(-1))
        y_actual.append(data.y)

    y_preds = torch.cat(y_preds, dim=0).detach().cpu()
    y_actual = torch.cat(y_actual, dim=0).detach().cpu()

    mae = MAE(y_actual, y_preds)
    r2 = r2_score(y_actual, y_preds)

    print()
    print(f'Validating... '
          f'MAE: {mae:.3f}, '
          f'R2: {r2:.3f} on val_dataset, '
          f'completed at {datetime.datetime.now().strftime("%H:%M")}')

    return y_actual, y_preds, mae, r2


# test inference loop
def test_inference(model, loader):
    y_test, y_test_pred, mae, r2 = val_inference(model, loader)
    print(f'\nTest Results:\tMAE: {mae:.3f}, R2: {r2:.3f}')

    return y_test.numpy(), y_test_pred.numpy(), mae, r2


# %% Training
criterion = nn.L1Loss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets = ['matbench_jdft2d', 'matbench_phonons', 'matbench_dielectric',
            'matbench_log_kvrh', 'matbench_log_gvrh', 'matbench_perovskites',
            'matbench_mp_gap', 'matbench_mp_is_metal', 'matbench_mp_e_form']

dataset = 'matbench_log_gvrh'

full_data = np.load(f'data/{dataset}.npy', allow_pickle=True).tolist()

new_data = []
for cif in full_data:
    pos = cif['R']
    z = cif['Z']
    y = cif['Y']
    new_data.append(Data(
        z=torch.tensor(z, dtype=torch.long).reshape(-1),
        y=torch.tensor(y, dtype=torch.float),
        pos=torch.tensor(pos, dtype=torch.float)
        ))

# 60/20/20 split for the training, validation, and test dataset
random_state = np.random.RandomState(seed=42)
length = len(new_data)
a = round(0.6 * length)
b = round(0.2 * length)
perm = list(random_state.permutation(np.arange(len(new_data))))
train_idx = perm[:a]
val_idx = perm[a:a+b]
test_idx = perm[a+b:]

mat_train = [new_data[i] for i in train_idx]
mat_val = [new_data[i] for i in val_idx]
mat_test = [new_data[i] for i in test_idx]

materials_data = mat_train

model = DimeNet(hidden_channels=128, out_channels=1, num_blocks=4,
                num_bilinear=8, out_emb_size=256, int_emb_size=64,
                basis_emb_size=8, num_spherical=7, num_radial=6,
                cutoff=6.0, envelope_exponent=5, num_before_skip=1,
                num_after_skip=2, num_output_layers=3)

model = model.to(device)

epochs = 200
batch_size = 4
loader = DataLoader(materials_data, batch_size=batch_size)
val_loader = DataLoader(mat_val, batch_size=batch_size)

optimizer = optim.AdamW(model.parameters(), weight_decay=0.01, amsgrad=True)
lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer,
                                           base_lr=1e-4,
                                           max_lr=1e-3,
                                           cycle_momentum=False,
                                           step_size_up=len(loader))

print(f'Started training at {datetime.datetime.now().strftime("%H:%M")}')
print("See you in 10 years (ノ͡° ͜ʖ ͡°)ノ︵┻┻\n")

r2_train = []
mae_train = []

r2_val = []
mae_val = []

# Training Loop
for epoch in range(1, epochs + 1):

    y_preds = []
    y_actual = []

    for data in tqdm(loader, leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data.z, data.pos, data.batch)

        y_preds.append(pred.view(-1))
        y_actual.append(data.y)

        loss = criterion(pred.view(-1), data.y)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    y_preds = torch.cat(y_preds, dim=0).detach().cpu()
    y_actual = torch.cat(y_actual, dim=0).detach().cpu()

    r2 = r2_score(y_actual, y_preds)
    mae = MAE(y_actual, y_preds)

    r2_train.append(r2)
    mae_train.append(mae)
    print()
    print(f'Epoch: {epoch:02d}, '
          f'MAE: {mae:.3f}, '
          f'R2: {r2:.3f}, '
          f'completed at {datetime.datetime.now().strftime("%H:%M")}')

    print(lr_scheduler._last_lr)

    # validation inference
    y_val, y_val_pred, mae_v, r2_v = val_inference(model, val_loader)
    mae_val.append(mae_v)
    r2_val.append(r2_v)

    date = datetime.date.today().strftime('%m-%d')

r2_train = pd.DataFrame(r2_train)
mae_train = pd.DataFrame(mae_train)

r2_val = pd.DataFrame(r2_val)
mae_val = pd.DataFrame(mae_val)

# Test Performance
test_loader = DataLoader(mat_test, batch_size=batch_size)
actual, predicted, mae_test, r2_test = test_inference(model, test_loader)
