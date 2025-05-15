import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
def reparametrize_to_frame_time(df, column, clock_time='TimeFromDotsOn', frame_time='FrameTime'):
    new_col = f"{column}-Frame"
    result = []

    for idx in df.index:
        clock = np.array(df.at[idx, clock_time])
        target = np.array(df.at[idx, frame_time])
        data = np.array(df.at[idx, column])

        f = interp1d(clock, data, kind='cubic', bounds_error=False, fill_value='extrapolate', axis=-1)
        interp = f(target)
        result.append(interp)

    df[new_col] = result

    
import numpy as np

def fix_dotsPos_shapes(df):
    # Iterate over all trials
    for trial_idx in range(len(df['dotsPos'])):
        trial = df['dotsPos'][trial_idx]
        
        # Iterate over each frame in the trial
        for frame_idx in range(len(trial)):
            frame = trial[frame_idx]
            
            # Convert numpy arrays to lists of lists
            if isinstance(frame, np.ndarray):
                # If the frame has one dot, convert it to shape (2,1)
                if frame.ndim == 1 and frame.shape[0] == 2:
                    # Reshape the frame to (2,1) for a single dot
                    trial[frame_idx] = frame.reshape(2, 1).tolist()
                
                # If there are zero dots, ensure the shape is (2,0)
                elif frame.size == 0:
                    # Create an empty list of lists with shape (2,0)
                    trial[frame_idx] = np.empty((2, 0)).tolist()
                
                # If there are multiple dots, convert the array to a list of lists
                elif frame.ndim == 2:
                    trial[frame_idx] = frame.tolist()
        
        # Replace the trial back in the dataframe in-place
        df.at[trial_idx, 'dotsPos'] = trial


def rescale_and_rotate_dots(df, scale_factor=1/29.2, rotate=True):
    for trial_idx in range(len(df)):
        trial = df['dotsPos'][trial_idx]
        targ1 = np.array(df.at[trial_idx, 'targ1Pos'])
        targ2 = np.array(df.at[trial_idx, 'targ2Pos'])

        if rotate:
            vec = targ1 - targ2
            theta = -np.arctan2(vec[1], vec[0])  # negative for clockwise rotation

            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
        else:
            R = np.eye(2)

        # Transform each frame
        for frame_idx in range(len(trial)):
            frame = trial[frame_idx]
            if isinstance(frame, list) and len(frame) == 2:
                dots = np.array(frame) * scale_factor  # scale
                rotated = R @ dots  # rotate
                trial[frame_idx] = rotated.tolist()

        df.at[trial_idx, 'dotsPos'] = trial


def create_movie(df, bins=51):
    # Determine max absolute x and y across all trials and frames
    max_val = 0
    for trial in df['dotsPos']:
        for frame in trial:
            if frame and len(frame[0]) > 0:
                x = np.array(frame[0])
                y = np.array(frame[1])
                max_val = max(max_val, np.abs(x).max(), np.abs(y).max())

    # Define bin edges
    edges = np.linspace(-max_val, max_val, bins + 1)

    # Process each trial
    movies = []
    for trial in df['dotsPos']:
        movie = []
        for frame in trial:
            if frame and len(frame[0]) > 0:
                x = np.array(frame[0])
                y = np.array(frame[1])
                H, _, _ = np.histogram2d(y, x, bins=[edges, edges])
            else:
                H = np.zeros((bins, bins), dtype=int)
            movie.append(H.astype(int))
        movies.append(np.stack(movie))
    
    df['Movie'] = movies



def crop_movie_to_valid_times(df):
    for idx in df.index:
        full_times = df.at[idx, 'FrameTime']
        valid_times = df.at[idx, 'TimeFromDotsOn']

        # Get valid range
        t_min, t_max = valid_times[0], valid_times[-1]

        # Find indices within this range
        mask = (full_times >= t_min) & (full_times <= t_max)
        indices = np.where(mask)[0]

        # Crop Movie and FrameTime
        df.at[idx, 'Movie'] = df.at[idx, 'Movie'][indices]
        df.at[idx, 'FrameTime'] = full_times[indices]



class PaddedTrialDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, masks):
        self.inputs = inputs
        self.targets = targets
        self.masks = masks

    def __getitem__(self, index):
        # Convert (T, H, W) -> (1, T, H, W) by permuting and unsqueezing if needed.
        # If already (T, H, W), permute to (1, T, H, W)
        x = self.inputs[index]
        if x.dim() == 3:
            # (T, H, W) -> (1, T, H, W)
            x = x.unsqueeze(0)
        # For clarity, permute even if shape is already (1, T, H, W)
        x = x.permute(0, 1, 2, 3)  # (1, T, H, W) stays the same, but allows for preprocessing if needed.
        return x, self.targets[index], self.masks[index]

    def __len__(self):
        return len(self.inputs)

def get_dataloader(df, output, batch_size=32, trial_mask=None, behavioral_axis=None, stratified=None):
    if trial_mask is not None:
        df = df[trial_mask].reset_index(drop=True)

    if stratified is not None and behavioral_axis is not None:
        train_idx = []
        val_idx = []
        for group, group_df in df.groupby(stratified, observed=True):
            group_df = group_df.copy()
            quantiles = pd.qcut(group_df[behavioral_axis], q=10, duplicates='drop')
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            g_train_idx, g_val_idx = next(splitter.split(group_df, quantiles))
            train_idx.extend(group_df.index[g_train_idx])
            val_idx.extend(group_df.index[g_val_idx])
    elif stratified is not None:
        stratify_labels = df[stratified].values
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(df, stratify_labels))
    elif behavioral_axis is not None:
        quantiles = pd.qcut(df[behavioral_axis], q=10, duplicates='drop')
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(df, quantiles))
    else:
        N = len(df)
        idx = np.random.permutation(N)
        split = int(N * 0.8)
        train_idx, val_idx = idx[:split], idx[split:]

    def build_dataset(indices):
        movies = [torch.tensor(df.at[i, 'Movie'], dtype=torch.float32) for i in indices]
        targets = [torch.tensor(df.at[i, output], dtype=torch.float32) for i in indices]

        max_len = max(movie.shape[0] for movie in movies)

        pad_inputs = []
        pad_targets = []
        pad_masks = []

        for x, y in zip(movies, targets):
            pad_len = max_len - x.shape[0]
            x_padded = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_len))
            # x_padded is (T, H, W)
            pad_inputs.append(x_padded)
            if y.ndim == 1:
                y = y.unsqueeze(0)  # reshape to (1, T)

            pad_targets.append(torch.nn.functional.pad(y, (0, pad_len)))
            mask = torch.ones_like(y, dtype=torch.bool)
            mask = torch.nn.functional.pad(mask, (0, pad_len))
            pad_masks.append(mask)

        # Assert consistent dimensionality across targets
        assert all(y.shape[0] == pad_targets[0].shape[0] for y in pad_targets), "Inconsistent target dimensions across trials"

        # Stack the padded inputs (list of (T, H, W)) into a tensor of shape (N, T, H, W)
        X = torch.stack(pad_inputs)
        Y = torch.stack(pad_targets)
        M = torch.stack(pad_masks)

        return PaddedTrialDataset(X, Y, M)

    train_dataset = build_dataset(train_idx)
    val_dataset = build_dataset(val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



from tqdm import tqdm

# Laplacian regularization for convolutional weights
import torch.nn.functional as F
def laplacian_regularization(conv_weights):
    """
    Computes Laplacian regularization for 1D, 2D, and 3D convolutional weights.
    """
    laplacian_loss = 0.0

    if conv_weights.dim() == 3:  # 1D
        stencil = torch.tensor([-1, 2, -1], dtype=conv_weights.dtype, device=conv_weights.device).view(1, 1, 3)
        for weight in conv_weights:
            for w in weight:
                lap = F.conv1d(w[None, None], stencil, padding=1)
                laplacian_loss += (lap**2).sum() / 6

    elif conv_weights.dim() == 4:  # 2D
        stencil = torch.tensor([[0.25, 0.5, 0.25], [0.5, -3, 0.5], [0.25, 0.5, 0.25]],
                               dtype=conv_weights.dtype, device=conv_weights.device).view(1, 1, 3, 3)
        for weight in conv_weights:
            for w in weight:
                lap = F.conv2d(w[None, None], stencil, padding=1)
                laplacian_loss += (lap**2).sum() / 10

    elif conv_weights.dim() == 5:  # 3D
        stencil = 1/26 * torch.tensor(
            [[[2, 3, 2], [3, 6, 3], [2, 3, 2]],
             [[3, 6, 3], [6, -88, 6], [3, 6, 3]],
             [[2, 3, 2], [3, 6, 3], [2, 3, 2]]],
            dtype=conv_weights.dtype, device=conv_weights.device).view(1, 1, 3, 3, 3)
        for weight in conv_weights:
            for w in weight:
                lap = F.conv3d(w[None, None], stencil, padding=1)
                laplacian_loss += (lap**2).sum() / 12

    return laplacian_loss

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda', lr=1e-3, weight_decay=1e-4, patience=5, reg_weight=None):
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss(reduction='none')

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

        for X, Y, M in train_bar:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            optimizer.zero_grad()

            output = model(X)
            loss = loss_fn(output, Y)
            masked_loss = (loss * M).sum() / M.sum()
            # Add Laplacian regularization if enabled
            if reg_weight is not None:
                laplacian_loss = 0.0
                for name, param in model.named_parameters():
                    if any(k in name for k in ["conv1d", "conv2d", "conv3d"]) and param.requires_grad:
                        laplacian_loss += laplacian_regularization(param)
                masked_loss = masked_loss + reg_weight * laplacian_loss
            masked_loss.backward()
            optimizer.step()
            total_train_loss += masked_loss.item()

        model.eval()
        total_val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)

        with torch.no_grad():
            for X, Y, M in val_bar:
                X, Y, M = X.to(device), Y.to(device), M.to(device)
                output = model(X)
                loss = loss_fn(output, Y)
                masked_loss = (loss * M).sum() / M.sum()
                total_val_loss += masked_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Restore best model weights
    model.load_state_dict(best_model_state)




#--- Motion Networks ---#




import torch
import torch.nn as nn
import torch.nn.functional as F
class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
        super(CausalConv3d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(0, padding[1], padding[2]),  # pad only spatially here
            dilation=dilation,
        )

    def forward(self, x):
        # Apply left-padding in time dimension manually
        pad_time = (self.kernel_size[0] - 1) * self.dilation[0]
        x = F.pad(x, (0, 0, 0, 0, pad_time, 0))  # only pad the front of the time axis
        return self.conv(x)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,  # No automatic padding
        )

    def forward(self, x):
        # Calculate the amount of left-padding needed for causality
        pad = (self.kernel_size - 1) * self.dilation
        x = nn.functional.pad(x, (pad, 0))  # Pad only the left side
        return self.conv(x)


class MotionEncodingNetwork(nn.Module):
    def __init__(self, latent_dim=10, temporal_layer_type="gru", base_channels=32):
        super(MotionEncodingNetwork, self).__init__()

        # First temporal and spatial convolutional block
        self.conv1d_time = CausalConv1d(in_channels=1, out_channels=base_channels, kernel_size=11)
        self.conv2d_spatial_1 = nn.Conv2d(in_channels=base_channels, out_channels=base_channels * 2, kernel_size=7, stride=2)
        self.layer_norm_1 = nn.LayerNorm(base_channels * 2)

        # Second spatial convolutional block
        self.conv2d_spatial_2 = nn.Conv2d(in_channels=base_channels * 2, out_channels=base_channels * 4, kernel_size=5, stride=2)
        self.layer_norm_2 = nn.LayerNorm(base_channels * 4)

        # Third spatial convolutional block
        self.conv2d_spatial_3 = nn.Conv2d(in_channels=base_channels * 4, out_channels=base_channels * 4, kernel_size=3, stride=2)
        self.layer_norm_3 = nn.LayerNorm(base_channels * 4)

        # Average Pooling to collapse spatial dimensions
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (channels, 1, 1)
        
        # Batch norm after spatial pooling
        self.batch_norm_after_pool = nn.BatchNorm1d(base_channels * 4)

        # Temporal Encoding Layer
        if temporal_layer_type == "gru":
            self.temporal_layer = nn.GRU(input_size=base_channels * 4, hidden_size=latent_dim, num_layers=1, dropout=0.3, batch_first=True)
            self.dropout = nn.Dropout(0.3)
        elif temporal_layer_type == "lstm":
            self.temporal_layer = nn.LSTM(input_size=base_channels * 4, hidden_size=latent_dim, num_layers=1, dropout=0.3, batch_first=True)
            self.dropout = nn.Dropout(0.3)
        elif temporal_layer_type == "transformer":
            self.temporal_layer = nn.TransformerEncoderLayer(d_model=base_channels * 4, nhead=8, batch_first=True)
            self.fc_temporal = nn.Linear(base_channels * 4, latent_dim)
        elif temporal_layer_type is None:
            self.temporal_layer = None
            self.fc_temporal = nn.Linear(base_channels * 4, latent_dim)

    def forward(self, x):
    
        # Input x: (B, 1, T, H, W)
        B, C, T, H, W = x.shape

        # First temporal convolution
        x = x.permute(0, 3, 4, 1, 2).reshape(-1, 1, T)  # (B, H, W, 1, T) -> (B*H*W, 1, T)
        x = self.conv1d_time(x)  # Shape: (B*H*W, base_channels, T)
        base_channels = self.conv1d_time.conv.out_channels

        # Reshape for spatial convolution
        x = x.reshape(B, H, W, base_channels, T).permute(0, 4, 3, 1, 2).contiguous()  # (B, T, base_channels, H, W)
        x = x.reshape(B * T, base_channels, H, W)  # Flatten batch and time

        # First spatial convolution
        x = F.relu(self.conv2d_spatial_1(x))
        x = x.permute(0, 2, 3, 1).contiguous()  # Shape: (batch * time, height, width, channels)
        x = self.layer_norm_1(x).permute(0, 3, 1, 2)  # Back to (batch * time, channels, height, width)

        # Second spatial convolution
        x = F.relu(self.conv2d_spatial_2(x))
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.layer_norm_2(x).permute(0, 3, 1, 2)

        # Third spatial convolution
        x = F.relu(self.conv2d_spatial_3(x))
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.layer_norm_3(x).permute(0, 3, 1, 2)

        # Average Pooling to collapse spatial dimensions
        x = self.avg_pool(x)  # Shape: (batch * time, base_channels*4, 1, 1)
        x = x.contiguous().view(B, T, -1)  # Explicit view for MPS compatibility

        # # Apply BatchNorm after pooling
        # x = x.permute(0, 2, 1).contiguous()  # Shape: (batch, channels, time)
        # x = self.batch_norm_after_pool(x)  # BatchNorm1d normalizes over channels
        # x = x.permute(0, 2, 1)  # Shape: (batch, time, channels)

        # Temporal encoding
        if isinstance(self.temporal_layer, (nn.GRU, nn.LSTM)):
            x, _ = self.temporal_layer(x)  # Shape: (batch, time, latent_dim)
            x = self.dropout(x)
        elif isinstance(self.temporal_layer, nn.TransformerEncoderLayer):
            x = self.temporal_layer(x)  # Shape: (batch, time, base_channels*4)
            x = self.fc_temporal(x)  # Shape: (batch, time, latent_dim)
        elif self.temporal_layer is None:  # Linear mapping from channels to latent_dim
            x = self.fc_temporal(x)  # Shape: (batch, time, latent_dim)

        x = x.permute(0, 2, 1)  # Shape: (batch, latent_dim, time)

        return x  # Final shape: (batch, latent_dim, time)

 

import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionEncodingNetwork3D(nn.Module):
    def __init__(self, latent_dim=10, hidden_dim=64, temporal_layer_type="gru"):
        super(MotionEncodingNetwork3D, self).__init__()

        self.hidden_dim = hidden_dim

        # First 3D convolutional block using CausalConv3d
        self.conv3d_1 = CausalConv3d(
            in_channels=1,
            out_channels=int(hidden_dim / 4),
            kernel_size=(11, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 0, 0)
        )
        self.layer_norm_1 = nn.LayerNorm(int(hidden_dim / 4))  # Normalize over channels only

        # Second 3D convolutional block using CausalConv3d
        self.conv3d_2 = CausalConv3d(
            in_channels=int(hidden_dim / 4),
            out_channels=int(hidden_dim / 2),
            kernel_size=(1, 5, 5),
            stride=(1, 2, 2),
            padding=(0, 0, 0)
        )
        self.layer_norm_2 = nn.LayerNorm(int(hidden_dim / 2))  # Normalize over channels only

        # Third 3D convolutional block using CausalConv3d
        self.conv3d_3 = CausalConv3d(
            in_channels=int(hidden_dim / 2),
            out_channels=hidden_dim,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 0, 0)
        )
        self.layer_norm_3 = nn.LayerNorm(hidden_dim)  # Normalize over channels only

        # Average Pooling to collapse spatial dimensions

        # Temporal Encoding Layer
        if temporal_layer_type == "gru":
            self.temporal_layer = nn.GRU(
                input_size=hidden_dim, hidden_size=latent_dim, num_layers=1, dropout=0.3, batch_first=True
            )
            self.dropout = nn.Dropout(0.3)
        elif temporal_layer_type == "lstm":
            self.temporal_layer = nn.LSTM(
                input_size=hidden_dim, hidden_size=latent_dim, num_layers=1, dropout=0.3, batch_first=True
            )
            self.dropout = nn.Dropout(0.3)
        elif temporal_layer_type == "rnn":
            self.temporal_layer = nn.RNN(
                input_size=hidden_dim,
                hidden_size=latent_dim,
                num_layers=1,
                dropout=0.3,
                nonlinearity="relu",
                batch_first=True,
            )
            self.dropout = nn.Dropout(0.3)
        elif temporal_layer_type == "transformer":
            self.temporal_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
            self.fc_temporal = nn.Linear(hidden_dim, latent_dim)
        elif temporal_layer_type is None:
            self.temporal_layer = None
            self.fc_temporal = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        batch_size, channels, time, height, width = x.shape

        # First 3D convolution
        x = F.relu(self.conv3d_1(x))  # Shape: (batch, hidden_dim, time, height, width)
        x = x.permute(0, 2, 3, 4, 1)  # Shape: (batch, time, height, width, hidden_dim)
        x = self.layer_norm_1(x).permute(0, 4, 1, 2, 3)  # Normalize over hidden_dim (channels)

        # Second 3D convolution
        x = F.relu(self.conv3d_2(x))  # Shape: (batch, hidden_dim, time, height, width)
        x = x.permute(0, 2, 3, 4, 1)  # Shape: (batch, time, height, width, hidden_dim)
        x = self.layer_norm_2(x).permute(0, 4, 1, 2, 3)  # Normalize over hidden_dim (channels)

        # Third 3D convolution
        x = F.relu(self.conv3d_3(x))  # Shape: (batch, hidden_dim, time, height, width)
        x = x.permute(0, 2, 3, 4, 1)  # Shape: (batch, time, height, width, hidden_dim)
        x = self.layer_norm_3(x).permute(0, 4, 1, 2, 3)  # Normalize over hidden_dim (channels)

        # Average Pooling to collapse spatial dimensions (manual mean over H, W)
        x = x.mean(dim=[3, 4])  # Shape: (batch, hidden_dim, time)

        # Temporal encoding
        x = x.permute(0, 2, 1)  # Shape: (batch, time, hidden_dim)
        if isinstance(self.temporal_layer, (nn.GRU, nn.LSTM, nn.RNN)):
            x, _ = self.temporal_layer(x)  # Shape: (batch, time, latent_dim)
            x = self.dropout(x)
        elif isinstance(self.temporal_layer, nn.TransformerEncoderLayer):
            x = self.temporal_layer(x)  # Shape: (batch, time, hidden_dim)
            x = self.fc_temporal(x)  # Shape: (batch, time, latent_dim)
        elif self.temporal_layer is None:  # Linear mapping from hidden_dim to latent_dim
            x = self.fc_temporal(x)  # Shape: (batch, time, latent_dim)

        x = x.permute(0, 2, 1)  # Shape: (batch, latent_dim, time)

        return x  # Final shape: (batch, latent_dim, time)




class TemporalEncodingNetwork(nn.Module):
    def __init__(self, height, width, latent_dim=10, temporal_layer_type="gru"):
        """
        Temporal encoding network with image size as an input.

        Parameters:
        - height: Height of the input image.
        - width: Width of the input image.
        - latent_dim: Number of latent features in the temporal layer.
        - temporal_layer_type: Type of temporal encoding layer ("gru", "lstm", "transformer", or "none").
        """
        super(TemporalEncodingNetwork, self).__init__()
        
        self.height = height
        self.width = width
        input_dim = height * width  # Flattened input dimensions

        # Temporal Encoding Layer
        if temporal_layer_type == "gru":
            self.temporal_layer = nn.GRU(input_size=input_dim, hidden_size=latent_dim, num_layers=5, dropout=0.3, batch_first=True)
            self.dropout = nn.Dropout(0.3)
        elif temporal_layer_type == "lstm":
            self.temporal_layer = nn.LSTM(input_size=input_dim, hidden_size=latent_dim, num_layers=3, dropout=0.3, batch_first=True)
            self.dropout = nn.Dropout(0.3)
        elif temporal_layer_type == "transformer":
            self.temporal_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=10, batch_first=True)
            self.fc_temporal = nn.Linear(input_dim, latent_dim)
        elif temporal_layer_type == "none":
            self.temporal_layer = None
            self.fc_temporal = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        """
        Forward pass for the temporal encoding network.

        Parameters:
        - x: Input tensor of shape (batch, channels, time, height, width).

        Returns:
        - Output tensor of shape (batch, latent_dim, time).
        """
        batch_size, channels, time, height, width = x.shape
        assert height == self.height and width == self.width, "Input image dimensions must match the initialized size."

        # Reshape the movie: combine pixels into features
        x = x.view(batch_size, time, -1)  # Shape: (batch, time, height * width * channels)

        # Temporal encoding
        if isinstance(self.temporal_layer, (nn.GRU, nn.LSTM)):
            x, _ = self.temporal_layer(x)  # Shape: (batch, time, latent_dim)
            x = self.dropout(x)
        elif isinstance(self.temporal_layer, nn.TransformerEncoderLayer):
            x = self.temporal_layer(x)  # Shape: (batch, time, input_dim)
            x = self.fc_temporal(x)  # Shape: (batch, time, latent_dim)
        elif self.temporal_layer is None:  # Linear mapping from input_dim to latent_dim
            x = self.fc_temporal(x)  # Shape: (batch, time, latent_dim)

        x = x.permute(0, 2, 1)  # Shape: (batch, latent_dim, time)

        return x  # Final shape: (batch, latent_dim, time)







import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

def evaluate_fit_quality(model, train_loader, val_loader, df, choice_col='choice'):
    model.eval()
    device = next(model.parameters()).device

    def collect_predictions(loader, indices):
        preds, targets, masks, rts, choices = [], [], [], [], []
        with torch.no_grad():
            for (X, Y, M) in loader:
                X, Y, M = X.to(device), Y.to(device), M.to(device)
                out = model(X)
                preds.append(out.cpu())
                targets.append(Y.cpu())
                masks.append(M.cpu())
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        masks = torch.cat(masks)
        for idx in indices:
            rts.append(df.at[idx, 'RT'])
            choices.append(df.at[idx, choice_col])
        return preds, targets, masks, np.array(rts), np.array(choices)

    # Get train/val indices from DataLoader datasets
    train_indices = train_loader.dataset.indices if hasattr(train_loader.dataset, 'indices') else range(len(train_loader.dataset))
    val_indices = val_loader.dataset.indices if hasattr(val_loader.dataset, 'indices') else range(len(val_loader.dataset))

    train_preds, train_tgts, train_masks, train_rts, train_choices = collect_predictions(train_loader, train_indices)
    val_preds, val_tgts, val_masks, val_rts, val_choices = collect_predictions(val_loader, val_indices)

    # Plot a few trial fits using plotly
    import random
    random.seed(42)
    train_idxs = random.sample(range(len(train_tgts)), 3)
    val_idxs = random.sample(range(len(val_tgts)), 3)
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"Train Trial {i}" for i in train_idxs] + [f"Val Trial {i}" for i in val_idxs])
    for j, i in enumerate(train_idxs):
        mask = train_masks[i, 0].bool()
        fig.add_trace(go.Scatter(y=train_tgts[i, 0, mask], mode='lines', name='Target', line=dict(color='blue'), showlegend=(j==0)), row=1, col=j+1)
        fig.add_trace(go.Scatter(y=train_preds[i, 0, mask], mode='lines', name='Predicted', line=dict(color='red'), showlegend=(j==0)), row=1, col=j+1)
    for j, i in enumerate(val_idxs):
        mask = val_masks[i, 0].bool()
        fig.add_trace(go.Scatter(y=val_tgts[i, 0, mask], mode='lines', name='Target', line=dict(color='blue'), showlegend=False), row=2, col=j+1)
        fig.add_trace(go.Scatter(y=val_preds[i, 0, mask], mode='lines', name='Predicted', line=dict(color='red'), showlegend=False), row=2, col=j+1)
    fig.update_layout(title="Sample Trial Fits", height=600, width=1000)
    fig.show()

    # --- Replace R² computation and plotting with PearsonR, nRMSE, and masked MSE loss ---


    def compute_metrics(preds, tgts, masks):
        corrs, nrmse, mse = [], [], []
        for p, t, m in zip(preds, tgts, masks):
            m = m.bool()
            t_m = t[m].numpy()
            p_m = p[m].numpy()
            if len(t_m) > 1:
                # Pearson correlation
                corr, _ = pearsonr(t_m, p_m)
                corrs.append(corr)

                # nRMSE
                rmse = np.sqrt(mean_squared_error(t_m, p_m))
                scale = t_m.max() - t_m.min() if t_m.max() > t_m.min() else 1.0
                nrmse.append(rmse / scale)

                # Raw loss
                mse.append(mean_squared_error(t_m, p_m))
            else:
                corrs.append(np.nan)
                nrmse.append(np.nan)
                mse.append(np.nan)
        return np.array(corrs), np.array(nrmse), np.array(mse)

    corr_train, nrmse_train, mse_train = compute_metrics(train_preds, train_tgts, train_masks)
    corr_val, nrmse_val, mse_val = compute_metrics(val_preds, val_tgts, val_masks)

    # Plot metrics vs RT using plotly
    df_metrics = pd.DataFrame({
        'RT': np.concatenate([train_rts, val_rts]),
        'PearsonR': np.concatenate([corr_train, corr_val]),
        'nRMSE': np.concatenate([nrmse_train, nrmse_val]),
        'Loss': np.concatenate([mse_train, mse_val]),
        'Choice': pd.Categorical(np.concatenate([train_choices, val_choices])),
        'Set': ['Train'] * len(train_rts) + ['Val'] * len(val_rts)
    })

    for metric in ['PearsonR', 'nRMSE', 'Loss']:
        fig = px.scatter(df_metrics, x='RT', y=metric, color='Choice', facet_col='Set',
                         title=f"{metric} vs RT Colored by Choice",
                         category_orders={'Set': ['Train', 'Val']})
        fig.update_layout(height=500, width=1000)
        fig.show()




# --- MotionDetectionNetwork3Dv2 ---

class MotionDetectionNetwork3Dv2(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, latent_dim=10):
        super(MotionDetectionNetwork3Dv2, self).__init__()

        # 3D convolution block
        self.conv3d = CausalConv3d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=(11, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 0, 0),
        )
        self.bn3d = nn.BatchNorm3d(hidden_dim)

        # 2D convolution blocks (applied per frame)
        self.conv2d_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=9, padding=0)
        self.bn2d_1 = nn.BatchNorm2d(hidden_dim)
        self.conv2d_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=0)
        self.bn2d_2 = nn.BatchNorm2d(hidden_dim)

        # Final linear projection (no temporal layer)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # Input x: (batch, 1, time, height, width)
        x = self.conv3d(x)  # (B, C, T, H, W)
        x = self.bn3d(x)
        x = F.relu(x)

        # Prepare for 2D conv per frame: (B*T, C, H, W)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).contiguous()

        x = F.relu(self.bn2d_1(self.conv2d_1(x)))
        x = F.relu(self.bn2d_2(self.conv2d_2(x)))

        # Global average pooling over spatial dimensions
        x = x.mean(dim=[2, 3])  # (B*T, C)

        # Linear projection
        x = self.fc(x)  # (B*T, latent_dim)

        # Reshape back to (B, latent_dim, T)
        x = x.view(B, T, -1).permute(0, 2, 1).contiguous()

        return x  # Shape: (B, latent_dim, T)