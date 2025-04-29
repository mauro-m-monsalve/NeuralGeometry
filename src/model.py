import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import trange
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import random
import itertools


class DecisionMakingModel:
    """
    DecisionMakingModel simulates neural population dynamics for perceptual decision-making in a retinotopic neural field.

    This model implements excitatory and inhibitory neural populations arranged on a 2D grid, with spatially structured recurrent connectivity and input patterns. 
    It supports multiple targets, external sensory evidence input, and spatiotemporally correlated noise. 
    The dynamics are governed by a set of differential equations discretized over time.

    Key Features:
    - Loads full model configuration from a YAML file.
    - Supports flexible device placement: CPU, CUDA, or MPS.
    - Implements wrap-around or bounded spatial grid behavior.
    - Provides methods to generate condition-dependent inputs, simulate neural dynamics, and extract decision-related activity.
    - Captures decision termination times based on activity crossing a threshold.
    - Provides utility methods for saving, plotting, and parallel simulation.

    Main Attributes:
    - cfg: Full model configuration dictionary.
    - device: Torch device where tensors are allocated.
    - conditions: Dictionary of experimental/task conditions (e.g., coherence levels).
    - targets: Information about decision targets (positions and names).
    - target_names: List of target names for identification and plotting.

    Typical Usage:
        model = DecisionMakingModel(params='path_to_config.yaml')
        model.initialize()
        model.run()
        df = model.terminate_decision()

    """
    def __init__(self, params='model_config.yaml'):
        with open(params, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.params_path = params
        self.device = self._select_device(self.cfg['simulation']['device'])
        self.conditions = self.cfg['input']['conditions']  # Dictionary of conditions
        self.targets = {
            'centers': [t['center'] for t in self.cfg['input']['targets']],
            'names': [t['name'] for t in self.cfg['input']['targets']],
            'list': self.cfg['input']['targets']
        }
        self.target_names = self.targets['names']

    def _select_device(self, requested_device):
        """
        Selects the computation device based on availability and user preference.

        Args:
            requested_device (str): Desired device string ('cuda', 'mps', 'auto', or fallback to 'cpu').

        Returns:
            torch.device: Selected computation device.
        """
        if requested_device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif requested_device == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        elif requested_device == 'auto':
            if torch.cuda.is_available():
                print('Auto-selected CUDA')
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                print('Auto-selected MPS')
                return torch.device('mps')
            else:
                print('No GPU available, using CPU.')
                return torch.device('cpu')
        else:
            print('Running on CPU')
            return torch.device('cpu')

    def initialize(self):
        """
        Initializes the model simulation space, kernel matrices, input maps, and other configuration-dependent attributes.
        Sets up spatial grid, time bins, connectivity kernels, and cropping masks for saving.
        """
        cfg = self.cfg
        self.mcfg = cfg['model']
        self.icfg = cfg['input']
        self.scfg = cfg['simulation']


        self.dt = self.scfg['dt']
        self.T = int(self.scfg['T'])
        self.n_steps = int(self.T / self.dt) + 1
        self.bin_steps = int(self.scfg['time_bin_size'] / self.dt)
        self.n_bins = self.n_steps // self.bin_steps
        self.Nx, self.Ny = self.scfg['grid_size']
        self.NTrials = self.scfg['NTrials']
        self.inv_taue_dt = torch.tensor(self.dt / self.mcfg['taue'], dtype=torch.float32, device=self.device)
        self.inv_taui_dt = torch.tensor(self.dt / self.mcfg['taui'], dtype=torch.float32, device=self.device)
        self.sqrt_dt_tau = torch.sqrt(torch.tensor(2 * self.dt / self.icfg['noise_tau'], dtype=torch.float32, device=self.device))

        extent_x, extent_y = self.scfg['extent_dva']
        self.lx = np.linspace(-extent_x / 2, extent_x / 2, self.Nx)
        self.ly = np.linspace(-extent_y / 2, extent_y / 2, self.Ny)
        self.xx, self.yy = np.meshgrid(self.lx, self.ly)

        if self.scfg.get('boundary_conditions', 'none') == 'wrap':
            dx = self.xx.reshape(1, -1) - self.xx.reshape(-1, 1)
            dy = self.yy.reshape(1, -1) - self.yy.reshape(-1, 1)
            grid_spacing_x = self.lx[1] - self.lx[0]
            grid_spacing_y = self.ly[1] - self.ly[0]
            grid_size_x = self.Nx * grid_spacing_x
            grid_size_y = self.Ny * grid_spacing_y
            dx = dx - np.round(dx / grid_size_x) * grid_size_x
            dy = dy - np.round(dy / grid_size_y) * grid_size_y
            self.dis2 = dx ** 2 + dy ** 2
        else:
            self.dis2 = (self.xx.reshape(1, -1) - self.xx.reshape(-1, 1)) ** 2 + \
                        (self.yy.reshape(1, -1) - self.yy.reshape(-1, 1)) ** 2

        self._build_kernels()

        self.Wee_T = self.Wee.T
        self.Wei_T = self.Wei.T
        self.Wie_T = self.Wie.T
        self.Wii_T = self.Wii.T
        del self.Wee, self.Wei, self.Wie, self.Wii

        self.target_inputs = []
        for t in self.targets['list']:
            cx, cy = t['center']
            self.target_inputs.append(
                np.exp(-((self.xx.reshape(-1) - cx)**2 + (self.yy.reshape(-1) - cy)**2) / self.icfg['input_sigma']**2)
            )
        stacked = np.stack(self.target_inputs).astype(np.float32)
        self.target_inputs = torch.tensor(stacked, device=self.device)

        snoise = self.icfg['noise_spatialcor_sigma']
        W = np.exp(-self.dis2 / snoise**2)
        W /= np.sqrt(np.sum(W**2, axis=1, keepdims=True))
        self.Wnoise = torch.tensor(W.astype(np.float32), device=self.device)

        sx, sy = self.scfg['extent_save']
        self.save_ix = np.where(np.abs(self.lx) <= sx / 2)[0]
        self.save_iy = np.where(np.abs(self.ly) <= sy / 2)[0]
        self.crop_x = slice(self.save_ix[0], self.save_ix[-1] + 1)
        self.crop_y = slice(self.save_iy[0], self.save_iy[-1] + 1)
        self.crop_Nx = self.crop_x.stop - self.crop_x.start
        self.crop_Ny = self.crop_y.stop - self.crop_y.start

        tin_radius = self.scfg['tin_radius']
        self.tin_masks = []
        for t in self.targets['list']:
            cx, cy = t['center']
            dist2 = (self.xx - cx) ** 2 + (self.yy - cy) ** 2
            mask = (dist2 <= tin_radius**2)
            cropped_mask = mask[self.crop_y, self.crop_x].flatten()
            self.tin_masks.append(cropped_mask)

    def _build_kernels(self):
        """
        Constructs recurrent connectivity kernels (Wee, Wei, Wie, Wii) using Gaussian-shaped profiles.
        These are used in the dynamic update of excitatory and inhibitory units.
        """
        m = self.mcfg
        d = self.dis2
        dev = self.device
        self.Wee = torch.tensor(m['wee'] * np.exp(-d / m['see']**2) / np.sum(np.exp(-d / m['see']**2), axis=1, keepdims=True), dtype=torch.float32, device=dev)
        self.Wei = torch.tensor(m['wei'] * np.exp(-d / m['sei']**2) / np.sum(np.exp(-d / m['sei']**2), axis=1, keepdims=True), dtype=torch.float32, device=dev)
        self.Wie = torch.tensor(m['wie'] * np.exp(-d / m['sie']**2) / np.sum(np.exp(-d / m['sie']**2), axis=1, keepdims=True), dtype=torch.float32, device=dev)
        self.Wii = torch.tensor(m['wii'] * np.exp(-d / m['sii']**2) / np.sum(np.exp(-d / m['sii']**2), axis=1, keepdims=True), dtype=torch.float32, device=dev)

    def to(self, device):
        """
        Moves all device-bound tensors to the specified device (e.g., 'cpu', 'cuda', 'mps').
        """
        if hasattr(self, 'target_inputs'):
            self.target_inputs = self.target_inputs.to(device)
        if hasattr(self, 'Wnoise'):
            self.Wnoise = self.Wnoise.to(device)
        if hasattr(self, 'Wee_T'):
            self.Wee_T = self.Wee_T.to(device)
            self.Wei_T = self.Wei_T.to(device)
            self.Wie_T = self.Wie_T.to(device)
            self.Wii_T = self.Wii_T.to(device)
        self.device = device  # update model's device attribute

    def cpu(self):
        """
        Moves all torch.Tensor attributes of the model to CPU for safe pickling or evaluation.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(torch.device('cpu')))
        self.device = torch.device('cpu')

    def get_input(self, condition, active_target, balanced_gain=True):
        """
        Generates the deterministic input pattern for the current condition and active target.

        Args:
            condition (tuple): Tuple of condition parameters (e.g., motion coherence, color coherence).
            active_target (dict): Dictionary containing the coordinates and name of the target to be prefered by the conditions.
            balanced_gain (bool): Whether to apply balanced gain to non-active targets.

        Returns:
            torch.Tensor: Input activity pattern of shape [Npix].
        """
        c = self.icfg['c']
        g = self.icfg['g']
        motion_coherence = condition[0]
        input_pattern = 0
        N = len(self.target_inputs)

        for i, t_input in enumerate(self.target_inputs):
            if self.targets['centers'][i] == active_target['center']:
                gain = 1 + g * motion_coherence
            else:
                gain = 1.0 if not balanced_gain else 1 - g * motion_coherence / (N - 1)
            input_pattern += gain * t_input
        input_pattern *= c
        return input_pattern.to(dtype=torch.float32, device=self.device)

    def get_noise(self, eta0=None):
        """
        Generates temporally and spatially correlated noise for all trials.

        Args:
            eta0 (torch.Tensor, optional): Initial condition for noise.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Correlated noise [NTrials, T, Npix] and final noise value for continuation.
        """
        Nx, Ny = self.Nx, self.Ny
        n_steps = self.bin_steps
        NTrials = self.NTrials
        tau = self.icfg['noise_tau']
        sigma = self.icfg['noise_sigma']
        dev = self.device

        eta = torch.randn(NTrials, n_steps, Nx * Ny, device=dev, dtype=torch.float32)
        uncorr_inoise = torch.zeros_like(eta)
        uncorr_inoise[:, 0, :] = eta0 if eta0 is not None else eta[:, 0, :]

        for t in range(1, n_steps):
            uncorr_inoise[:, t, :] = (
                uncorr_inoise[:, t - 1, :]
                + (self.dt / tau) * (-uncorr_inoise[:, t - 1, :])
                + self.sqrt_dt_tau * sigma * eta[:, t, :]
            )

        corr_noise = torch.matmul(uncorr_inoise, self.Wnoise)
        return corr_noise, uncorr_inoise[:, -1, :].detach()

    def simulate(self, input_data, use_progress=True):
        """
        Simulates the neural field model over time given the input data.

        Args:
            input_data (torch.Tensor): External input pattern for simulation.
            use_progress (bool): Whether to display a progress bar.

        Returns:
            dict: Simulation results with keys 're', 'input', and optionally 'ri' (if enabled).
        """
        relu = torch.nn.ReLU()
        save_ri = self.scfg.get('save_ri', False)

        re_prev = torch.full((self.NTrials, self.Nx * self.Ny), self.mcfg['re0'], device=self.device, dtype=torch.float32)
        ri_prev = torch.full((self.NTrials, self.Nx * self.Ny), self.mcfg['ri0'], device=self.device, dtype=torch.float32)
        re_bins, ri_bins, input_bins = [], [], []

        re = torch.empty(self.NTrials, self.bin_steps, self.Nx * self.Ny, device=self.device, dtype=torch.float32)
        ri = torch.empty_like(re)
        eta0 = None

        bar = trange(self.n_bins, desc="Simulating") if use_progress else range(self.n_bins)
        for _ in bar:
            inoise, eta0 = self.get_noise(eta0)
            total_input = input_data + inoise

            re[:, 0, :] = re_prev
            ri[:, 0, :] = ri_prev

            for t in range(1, self.bin_steps):
                re[:, t, :] = re[:, t-1, :] + self.inv_taue_dt * (
                    -re[:, t-1, :] + self.mcfg['ke'] * relu(
                        re[:, t-1, :] @ self.Wee_T - ri[:, t-1, :] @ self.Wei_T + total_input[:, t, :]
                    ) ** self.mcfg['ne']
                )
                ri[:, t, :] = ri[:, t-1, :] + self.inv_taui_dt * (
                    -ri[:, t-1, :] + self.mcfg['ki'] * relu(
                        re[:, t-1, :] @ self.Wie_T - ri[:, t-1, :] @ self.Wii_T + input_data
                    ) ** self.mcfg['ni']
                )

            re_prev = re[:, -1, :].detach()
            ri_prev = ri[:, -1, :].detach()

            re_crop = re.mean(1).view(self.NTrials, self.Ny, self.Nx)[:, self.crop_y, self.crop_x]
            in_crop = total_input.mean(1).view(self.NTrials, self.Ny, self.Nx)[:, self.crop_y, self.crop_x]
            re_bins.append(re_crop.reshape(self.NTrials, -1).cpu())
            input_bins.append(in_crop.reshape(self.NTrials, -1).cpu())

            if save_ri:
                ri_crop = ri.mean(1).view(self.NTrials, self.Ny, self.Nx)[:, self.crop_y, self.crop_x]
                ri_bins.append(ri_crop.reshape(self.NTrials, -1).cpu())
            #bar()

        result = {
            're': torch.stack(re_bins, dim=1).numpy(),
            'input': torch.stack(input_bins, dim=1).numpy()
        }
        if save_ri:
            result['ri'] = torch.stack(ri_bins, dim=1).numpy()

        return result

    
    def run(self):
        """
        Runs simulation over all defined conditions and target combinations serially.
        Stores results in self.data.
        """
        self.data = []
        conditions_list = list(itertools.product(*self.conditions.values()))
        for target in self.targets['list']:
            for condition in conditions_list:
                print(f"\n[run] Condition: {dict(zip(self.conditions.keys(), condition))}, Target: {target['name']}")
                input_data = self.get_input(condition, target)
                result = self.simulate(input_data)
                self.data.append({
                    'target': target,
                    **dict(zip(self.conditions.keys(), condition)),
                    **result
                })
    

    def parallel_run(self, n_jobs=-1):
        """
        Runs simulations in parallel using joblib for all conditions and targets.

        Args:
            n_jobs (int): Number of parallel jobs. Defaults to -1 (use all processors).

        Stores:
            self.data (list): List of dictionaries with results per condition-target pair.
        """
        from joblib import Parallel, delayed
    
        def run_condition(params_path, target_dict, condition_tuple):
            # Re-initialize model in subprocess with same config
            model = DecisionMakingModel(params=params_path)
            model.initialize()
            input_data = model.get_input(condition_tuple, target_dict)
            result = model.simulate(input_data)
            return {
                'target': target_dict,
                **dict(zip(model.conditions.keys(), condition_tuple)),
                **result
            }
    
        conditions_list = list(itertools.product(*self.conditions.values()))
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_condition)(self.params_path, target, condition)
            for target in self.targets['list']
            for condition in conditions_list
        )
        self.data = results


    def terminate_decision(self, remove_outliers=None):
        """
        Processes simulation results to detect decision times and choices for each trial.

        Args:
            remove_outliers (float, optional): If specified, removes trials with RT in the lowest and highest
                                               `remove_outliers` percentile.

        Returns:
            pd.DataFrame: Contains reaction times, choices, conditions, and neural activity history.
                          Also includes 'tin_masks' in DataFrame attributes.
        """
        tbin_ms = self.scfg['time_bin_size']
        threshold = self.scfg['decision_threshold']
        save_ri = self.scfg.get('save_ri', False)

        results = []
        incomplete_count = 0

        for condition in self.data:
            target = condition['target']
            RE = condition['re']       # [NTrials, T, Npix]
            IN = condition['input']    # [NTrials, T, Npix]
            RI = condition.get('ri', None) if save_ri else None

            for tr in range(RE.shape[0]):
                trial_re = RE[tr]         # [T, Npix]
                trial_input = IN[tr]      # [T, Npix]
                trial_ri = RI[tr] if RI is not None else None

                # Average Tin activity over time for each target
                tin_curves = [trial_re[:, mask].mean(1) for mask in self.tin_masks]
                tin_curves = np.stack(tin_curves, axis=0)  # [n_targets, T]

                # Compute decision variable: max diff between any two Tins
                diffs = np.max(tin_curves, axis=0) - np.min(tin_curves, axis=0)

                above_thresh = np.where(diffs >= threshold)[0]
                if len(above_thresh) == 0:
                    incomplete_count += 1
                    continue  # no decision

                decision_bin = above_thresh[0]
                RT = decision_bin * tbin_ms
                choice = np.argmax(tin_curves[:, decision_bin])

                entry = {
                    'RT': RT,
                    'choice': self.target_names[choice],
                'target': target['name'],
                    **{key: condition[key] for key in self.conditions.keys()},
                    're': trial_re[:decision_bin + 1].T,         # [Npix, T_decision]
                    'input': trial_input[:decision_bin + 1].T    # [Npix, T_decision]
                }

                if trial_ri is not None:
                    entry['ri'] = trial_ri[:decision_bin + 1].T

                results.append(entry)

        if incomplete_count > 0:
            print(f"[terminate_decision] {incomplete_count} trials did not reach decision threshold and were excluded.")

        df = pd.DataFrame(results)
        if remove_outliers is not None:
            lower = np.percentile(df['RT'], remove_outliers)
            upper = np.percentile(df['RT'], 100 - remove_outliers)
            df = df[(df['RT'] >= lower) & (df['RT'] <= upper)].reset_index(drop=True)
        df.attrs['tin_masks'] = {name: mask for name, mask in zip(self.target_names, self.tin_masks)}
        return df

    
    
    
    
    
    ###############################  PLOTTING METHODS  ##########################

    def plot_trial(self, trial=None, colors=None, save_fig=False, **condition_filters):
        """
        Plots the trial activity: mean Tin time courses and animated heatmap of the excitatory population.

        Args:
            trial (int, optional): Index of the trial to plot. If None, randomly selected.
            colors (list, optional): List of color strings for each target mask.
            save_fig (bool): Whether to save figures to disk.
            **condition_filters: Keyword arguments to specify condition filters (e.g., coherence=0.5).
        """
        # Filter data based on provided conditions
        entry = next(
            e for e in self.data
            if all(e.get(cond) == val for cond, val in condition_filters.items())
        )

        RE = entry['re']
        tbin_ms = self.scfg['time_bin_size']
        n_bins = RE.shape[1]
        NTrials = RE.shape[0]

        if trial is None:
            trial = random.randint(0, NTrials - 1)

        target_names = self.target_names  # Updated to use the new target_names location
        if colors is None:
            colors = ['xkcd:neon blue', 'xkcd:neon pink', 'xkcd:neon purple',
                      'xkcd:neon yellow', 'xkcd:neon red', 'xkcd:neon green'][:len(self.tin_masks)]
        hex_colors = [mcolors.to_hex(mcolors.to_rgb(c)) for c in colors]

        # Plot 1: Tin timecourse
        time = np.arange(n_bins) * tbin_ms
        re_grid = RE[trial]  # [n_bins, Npix]
        traces = []
        for i, (mask, name) in enumerate(zip(self.tin_masks, target_names)):
            traces.append(go.Scatter(
                x=time,
                y=re_grid[:, mask].mean(axis=1),
                mode='lines',
                name=name,
                line=dict(color=hex_colors[i])
            ))

        fig1 = go.Figure(data=traces)
        fig1.update_layout(
            title=f"Mean Tin Activity (Trial {trial}, " + ", ".join(f"{k}={v}" for k, v in condition_filters.items()) + ")",
            xaxis_title="Time (ms)",
            yaxis_title="Firing Rate (Hz)",
            template="plotly_white",
            height=350,
            width=500,
            margin=dict(t=60, b=40, l=30, r=30)
        )
        if save_fig:
            import os
            os.makedirs("figures", exist_ok=True)
            import re
            def sanitize(s): return re.sub(r'[^\w.-]', '_', str(s))
            condition_str = "_".join(f"{sanitize(k)}-{sanitize(v)}" for k, v in condition_filters.items())
            fig1.write_html(f"figures/Tin_Activity_{condition_str}.html")
        fig1.show()

        # Plot 2: Animated RE heatmap
        extent_x, extent_y = self.scfg['extent_save']
        xvals = np.linspace(-extent_x / 2, extent_x / 2, self.crop_Nx)
        yvals = np.linspace(-extent_y / 2, extent_y / 2, self.crop_Ny)

        initial_z = RE[trial, 0].reshape(self.crop_Ny, self.crop_Nx)
        heatmap = go.Heatmap(
            z=initial_z, x=xvals, y=yvals,
            colorscale='inferno', showscale=False, zsmooth=False,
            name="heatmap"
        )

        # Targets & circles
        markers, circles = [], []
        r = self.scfg['tin_radius']
        theta = np.linspace(0, 2*np.pi, 100)
        for i, (t, name) in enumerate(zip(self.icfg['targets'], target_names)):
            cx, cy = t['center']
            markers.append(go.Scatter(
                x=[cx], y=[cy], mode='markers',
                marker=dict(size=10, color='white', line=dict(color='gray', width=1)),
                showlegend=False
            ))
            circles.append(go.Scatter(
                x=cx + r * np.cos(theta), y=cy + r * np.sin(theta),
                mode='lines', line=dict(color=hex_colors[i], width=2),
                showlegend=False
            ))

        # Animation frames
        frames = []
        for b in range(n_bins):
            z_frame = RE[trial, b].reshape(self.crop_Ny, self.crop_Nx)
            annotations = [dict(
                x=extent_x / 2 + extent_x * 0.05,
                y=extent_y / 2,
                xanchor='left',
                text=f"Mean Tin activity",
                showarrow=False,
                font=dict(size=12)
            )]
            for i, (mask, name) in enumerate(zip(self.tin_masks, target_names)):
                mean_val = RE[trial, b, mask].mean()
                annotations.append(dict(
                    x=extent_x / 2 + extent_x * 0.05,
                    y=extent_y / 2 - (i+1) * extent_y * 0.08,
                    xanchor='left',
                    text=f"<b>{name}</b>: {mean_val:.1f} Hz",
                    showarrow=False,
                    font=dict(size=12, color=hex_colors[i])
                ))

            frames.append(go.Frame(
                data=[dict(type='heatmap', z=z_frame)],
                layout=dict(annotations=annotations),
                name=f"{b}"
            ))

        fig2 = go.Figure(
            data=[heatmap] + markers + circles,
            frames=frames,
            layout=go.Layout(
                title=f"Neural Population Activity (Trial {trial}, " + ", ".join(f"{k}={v}" for k, v in condition_filters.items()) + ")",
                xaxis=dict(visible=False, scaleanchor="y"),
                yaxis=dict(visible=False),
                annotations=frames[0].layout.annotations,
                template="plotly_white",
                height=500,
                width=700,
                margin=dict(t=60, b=40, l=30, r=140),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 50, "redraw": True},
                                          "fromcurrent": True, "transition": {"duration": 0}}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate", "transition": {"duration": 0}}])
                    ],
                    x=0.1, y=-0.12
                )],
                sliders=[dict(
                    steps=[dict(method='animate',
                                args=[[f"{b}"], dict(frame=dict(duration=0, redraw=True),
                                                     mode='immediate')],
                                label=f"{int(b * tbin_ms)} ms") for b in range(n_bins)],
                    x=0.1, y=-0.18, len=0.85,
                    currentvalue=dict(font=dict(size=12), prefix='   Time: ', visible=True)
                )]
            )
        )
        if save_fig:
            fig2.write_html(f"figures/Population_Activity_{condition_str}.html")
        fig2.show()



# Utility function: get_pixel_at
def get_pixel_at(loc=[0, 0], grid_shape=(31, 31), extent=(30, 30)):
    """
    Returns the index (flattened) of the pixel closest to the given location in a spatial grid.

    Args:
        loc (list or tuple): [x, y] coordinates to find the nearest pixel.
        grid_shape (tuple): (Nx, Ny) number of pixels in x and y directions.
        extent (tuple): (X, Y) extent of the grid in dva units from -X/2 to X/2 and -Y/2 to Y/2.

    Returns:
        int: Flattened index of the pixel closest to the specified location.
    """
    Nx, Ny = grid_shape
    X, Y = extent
    x_vals = np.linspace(-X / 2, X / 2, Nx)
    y_vals = np.linspace(-Y / 2, Y / 2, Ny)
    xx, yy = np.meshgrid(x_vals, y_vals)
    distances = (xx - loc[0])**2 + (yy - loc[1])**2
    return np.argmin(distances)




# ===============================================
# Local Linear Model Fitting for Evidence Modulation
# ===============================================


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from src.geometry import uniformize

def fit_local_linear_model(
    df,
    choice='Contra',
    device='auto',
    arc_window_size=0.1,
    arc_window_step=0.1,
    rt_window_size=0.1,
    rt_window_step=0.1,
    lambda_derivative=0.001,
    n_epochs=500,
    lr=1e-1,
    patience=20,
    min_delta=1e-4,
    verbose=True
):
    """
    Fit a local linear model using Input-Arc to predict Resolution and Uncertainty Projections across the arc-length time series.
    Performs time-resolved predictions over moving 2D windows (Arc × RT), with a smoothing penalty on the spatial derivatives of the input weights.

    Args:
        df (pd.DataFrame): DataFrame with 'choice', 'RT', 'input-Arc', 'ResolutionProjection', 'UncertaintyProjection', 'Arc-Length'.
        choice (str): Choice to filter (e.g., 'Contra').
        device (str or torch.device): 'auto', 'cuda', 'mps', or 'cpu'.
        arc_window_size (float): Arc-length window size (fraction of [0,1]).
        arc_window_step (float): Arc-length step size (fraction of [0,1]).
        rt_window_size (float): RT window size (fraction of [0,1]).
        rt_window_step (float): RT window step size (fraction of [0,1]).
        lambda_derivative (float): Weight for smoothness penalty across input pixels.
        n_epochs (int): Number of optimization epochs.
        lr (float): Learning rate.
        patience (int): Number of epochs with no improvement to trigger early stopping.
        min_delta (float): Minimum loss improvement to reset patience.
        verbose (bool): Whether to print window progress.

    Returns:
        pd.DataFrame: Each row is a window with keys:
            'arc_center', 'rt_center', 'weights_resolution', 'weights_uncertainty', 'bias_resolution', 'bias_uncertainty'.
    """

    # Device selection
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    # Data selection and preparation
    dfc = df[df['choice'] == choice].copy()
    if len(dfc) == 0:
        raise ValueError(f"No rows for choice={choice}")

    Input_Arc = np.stack([np.array(x) for x in dfc['input-Arc'].values])  # (trials, Npix, T)
    ResolutionProjection = np.stack([np.array(x) for x in dfc['ResolutionProjection'].values])  # (trials, T)
    UncertaintyProjection = np.stack([np.array(x) for x in dfc['UncertaintyProjection'].values])  # (trials, T)
    RT = uniformize(dfc['RT'].values)

    Input_Arc = torch.tensor(Input_Arc, dtype=torch.float32, device=device)
    ResolutionProjection = torch.tensor(ResolutionProjection, dtype=torch.float32, device=device)
    UncertaintyProjection = torch.tensor(UncertaintyProjection, dtype=torch.float32, device=device)
    RT = torch.tensor(RT, dtype=torch.float32, device=device)

    _, D2, T = Input_Arc.shape
    D = int(np.sqrt(D2))

    arc_values = torch.linspace(0, 1, T, device=device)

    # Define windows for arc and RT
    arc_starts = torch.arange(0, 1 - arc_window_size + 1e-8, arc_window_step, device=device)
    rt_starts = torch.arange(0, 1 - rt_window_size + 1e-8, rt_window_step, device=device)
    windows = [(a.item(), r.item()) for a in arc_starts for r in rt_starts]
    num_windows = len(windows)

    # Model parameters: weights and biases for resolution and uncertainty
    w_input_resolution = nn.Parameter(torch.randn(num_windows, D, D, device=device))
    w_input_uncertainty = nn.Parameter(torch.randn(num_windows, D, D, device=device))
    bias_resolution = nn.Parameter(1 + torch.randn(num_windows, device=device))
    bias_uncertainty = nn.Parameter(torch.randn(num_windows, device=device))

    optimizer = optim.Adam([w_input_resolution, w_input_uncertainty, bias_resolution, bias_uncertainty], lr=lr)
    mse_loss = nn.MSELoss()

    # Training loop
    loss_history = []
    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in tqdm(range(n_epochs), desc="Training", unit="epoch"):
        total_loss = 0

        for idx, (arc_start, rt_start) in enumerate(windows):
            arc_end = arc_start + arc_window_size
            rt_end = rt_start + rt_window_size

            arc_segment = (arc_values >= arc_start) & (arc_values < arc_end)  # (T,)
            rt_mask = (RT >= rt_start) & (RT < rt_end)  # (trials,)

            if arc_segment.sum() == 0 or rt_mask.sum() == 0:
                continue

            Input_segment = Input_Arc[rt_mask][:, :, arc_segment]  # (selected_trials, D2, selected_T)
            Target_resolution_segment = ResolutionProjection[rt_mask][:, arc_segment]  # (selected_trials, selected_T)
            Target_uncertainty_segment = UncertaintyProjection[rt_mask][:, arc_segment]  # (selected_trials, selected_T)

            # Normalize input
            Input_mean = torch.mean(Input_segment, dim=(0, 2), keepdim=True)
            Input_segment = Input_segment - Input_mean
            Input_norm = torch.norm(Input_segment, dim=1, keepdim=True) + 1e-8
            Input_segment = Input_segment / Input_norm

            pred_resolution = (w_input_resolution[idx].view(1, D2, 1) * Input_segment).sum(dim=1) + bias_resolution[idx]
            pred_uncertainty = (w_input_uncertainty[idx].view(1, D2, 1) * Input_segment).sum(dim=1) + bias_uncertainty[idx]

            loss_resolution = mse_loss(pred_resolution, Target_resolution_segment)
            loss_uncertainty = mse_loss(pred_uncertainty, Target_uncertainty_segment)
            loss = loss_resolution + loss_uncertainty

            total_loss += loss

        # Regularization for spatial smoothness of input weights
        penalty = 0
        for i in range(num_windows):
            w_res = w_input_resolution[i].view(D, D)
            w_unc = w_input_uncertainty[i].view(D, D)
            penalty += (torch.diff(w_res, n=2, dim=0)**2).sum() + (torch.diff(w_res, n=2, dim=1)**2).sum()
            penalty += (torch.diff(w_unc, n=2, dim=0)**2).sum() + (torch.diff(w_unc, n=2, dim=1)**2).sum()

        total_loss = total_loss + lambda_derivative * penalty
        loss_history.append(total_loss.item())

        current_loss = total_loss.item()
        if best_loss - current_loss > min_delta:
            best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1} with best loss {best_loss:.6f}")
            break

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Reorganize results directly into a list of dicts, one per window, with arc_center and rt_center
    arc_centers = arc_starts.cpu().numpy() + arc_window_size / 2
    rt_centers = rt_starts.cpu().numpy() + rt_window_size / 2
    weights_res_np = w_input_resolution.detach().cpu().numpy()
    weights_unc_np = w_input_uncertainty.detach().cpu().numpy()
    bias_res_np = bias_resolution.detach().cpu().numpy()
    bias_unc_np = bias_uncertainty.detach().cpu().numpy()

    final_results = []
    idx = 0
    for a in arc_centers:
        for r in rt_centers:
            entry = {
                'arc_center': a,
                'rt_center': r,
                'weights_resolution': weights_res_np[idx],
                'weights_uncertainty': weights_unc_np[idx],
                'bias_resolution': bias_res_np[idx],
                'bias_uncertainty': bias_unc_np[idx]
            }
            final_results.append(entry)
            idx += 1

    if verbose:
        print(f"Training completed: Final loss {loss_history[-1]:.4f}")

    return pd.DataFrame(final_results)


def add_ev_modulation(df_fit, evdir):
    """
    Adds 'ev_mod_resolution' and 'ev_mod_uncertainty' to df_fit.

    Args:
        df_fit (pd.DataFrame): DataFrame with fitted weights.
        evdir (np.ndarray): 1D array of shape (D*D,) representing the evidence direction.
    """
    evdir = evdir.flatten()
    
    ev_mod_resolution = []
    ev_mod_uncertainty = []

    for idx, row in df_fit.iterrows():
        w_res = row['weights_resolution'].reshape(-1)  # Flatten D x D to D*D
        w_unc = row['weights_uncertainty'].reshape(-1)
        ev_mod_resolution.append(np.dot(w_res, evdir))
        ev_mod_uncertainty.append(np.dot(w_unc, evdir))

    df_fit['ev_mod_resolution'] = ev_mod_resolution
    df_fit['ev_mod_uncertainty'] = ev_mod_uncertainty

    return df_fit