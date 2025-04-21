import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import laplace
import pickle
import gzip


def compute_spatial_information(RF):
    """
    Compute spatial information content (SIC) in bits/spike for a single 2D response field.

    Parameters:
        RF: np.ndarray of shape (x, y), firing rate map

    Returns:
        float: SIC in bits/spike
    """
    rf = RF.flatten()
    rf = rf[rf >= 0]  # Ignore negative or nan entries
    rf = rf[~np.isnan(rf)]

    if rf.sum() == 0 or rf.mean() == 0:
        return 0.0

    p = np.ones_like(rf) / len(rf)  # Uniform occupancy
    r_bar = np.mean(rf)
    info = p * (rf / r_bar) * np.log2((rf + 1e-12) / r_bar)
    return np.nansum(info)


def compute_morans_I(RF, sigma=1.0):
    """
    Compute Moran's I for a 2D spatial map using a Gaussian weight matrix.

    Parameters:
        RF: np.ndarray of shape (x, y)
        sigma: float, standard deviation of the Gaussian weight kernel

    Returns:
        float: Moran's I index
    """
    if np.isnan(RF).all():
        return np.nan

    RF = np.nan_to_num(RF, nan=0.0)
    x = RF - np.mean(RF)
    n = x.size

    # Gaussian weight kernel
    size = int(np.ceil(3 * sigma)) * 2 + 1
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    gaussian_weights = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gaussian_weights[size // 2, size // 2] = 0  # remove self-weight

    # Normalize weights
    gaussian_weights /= gaussian_weights.sum()

    # Convolve using zero-padding to handle edge effects
    from scipy.signal import convolve2d
    num = np.sum(x * convolve2d(x, gaussian_weights, mode='same', boundary='fill', fillvalue=0))
    denom = np.sum(x**2)

    return num / denom if denom > 0 else np.nan


def choice_selectivity_binary(choice_0, choice_1, arc_window=0.2):
    """
    Compute choice selectivity as (1 - Pearson correlation)/2 in a running window over arc length.
    Only computes valid values when a full window is available. 
    Arc-length indices at the boundaries (less than half_window at each end) are returned as NaN.
    
    Parameters:
        choice_0: np.ndarray of shape (neurons, arc_length, RT)
        choice_1: np.ndarray of shape (neurons, arc_length, RT)
        arc_window: float, fraction of arc_length to use as the window size (default=0.2)
        
    Returns:
        selectivity: np.ndarray of shape (neurons, arc_length, RT), with valid values computed only 
                     where the full window is available and NaN for the boundary points.
    """
    # Check shapes
    assert choice_0.shape == choice_1.shape, "Input arrays must have the same shape."
    N, L, B = choice_0.shape
    
    # Define window size (number of arc points)
    window_size = max(1, int(arc_window * L))
    half_window = window_size // 2

    # Prepare output array; default is NaN everywhere.
    selectivity = np.full((N, L, B), np.nan, dtype=np.float32)

    # Only compute for indices where the complete window is available:
    for l in range(half_window, L - half_window):
        start = l - half_window
        end = l + half_window + 1  # window size is end - start = window_size
        # Extract the window from the original arrays (no padding)
        x = choice_0[:, start:end, :]  # shape: (N, window_size, B)
        y = choice_1[:, start:end, :]  # same shape
        
        # Compute the mean over the arc-window dimension
        x_mean = np.nanmean(x, axis=1, keepdims=True)
        y_mean = np.nanmean(y, axis=1, keepdims=True)
        
        # Center the data
        xm = x - x_mean
        ym = y - y_mean
        
        # Compute numerator (covariance) and denominator (std-products)
        num = np.nansum(xm * ym, axis=1)  # shape: (N, B)
        denom = np.sqrt(np.nansum(xm**2, axis=1) * np.nansum(ym**2, axis=1))  # shape: (N, B)
        
        # Compute Pearson correlation, avoid division by zero
        corr = np.where(denom > 0, num / denom, np.nan)
        
        # Calculate selectivity: (1 - corr) / 2 and assign to this arc index l
        selectivity[:, l, :] = (1 - corr) / 2

    return selectivity


def get_response_fields(df, shift=0.15, window=0.085, NSide=31, mindeg=-15.0, maxdeg=15.0, sigma=3.0, mode='forward'):
    """
    Compute smooth 2D response fields for each neuron using Gaussian kernel smoothing.

    Parameters:
        df: pandas DataFrame with per-trial keys: 'targetsOn', 'spCellPop', 'targ1Pos'
        shift: float, seconds after 'targetsOn' for spike window center
        window: float, half-width of spike counting window in seconds
        NSide: int, number of bins along x and y
        mindeg, maxdeg: bounds of spatial grid in degrees
        sigma: bandwidth of Gaussian smoothing kernel (in degrees)
        mode: 'forward' or 'backward', whether to align the response window to 'targetsOn' or 'saccadeDetected'

    Returns:
        xarray.Dataset with dims (cell, x, y) and data var 'FR_smooth'
    """
    # Spike count → firing rate
    duration = 2 * window
    fr_table = []

    for _, row in df.iterrows():
        xpos, ypos = float(row['targ1Pos'][0])/10.0, float(row['targ1Pos'][1])/10.0
        if mode == 'forward':
            t_center = row['targetsOn'] + shift
        elif mode == 'backward':
            t_center = row['saccadeDetected'] - shift
        else:
            raise ValueError("mode must be 'forward' or 'backward'")
        t0 = t_center - window
        t1 = t_center + window

        spikes = row['spCellPop']  # list of arrays (len = n_cells)
        for cell_idx, spike_train in enumerate(spikes):
            nspikes = np.sum((spike_train > t0) & (spike_train < t1))
            fr_table.append([xpos, ypos, cell_idx, nspikes / duration])

    df_fr = pd.DataFrame(fr_table, columns=['x', 'y', 'cell', 'FR'])

    # Set up grid
    x_vals = np.linspace(mindeg, maxdeg, NSide)
    y_vals = np.linspace(mindeg, maxdeg, NSide)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Output array
    n_cells = df_fr['cell'].max() + 1
    Z = np.zeros((n_cells, NSide, NSide))

    for cell in range(n_cells):
        sub = df_fr[df_fr['cell'] == cell]
        cx = sub['x'].values[:, None]
        cy = sub['y'].values[:, None]
        fr = sub['FR'].values[:, None]

        dx = grid_points[:, 0][None, :] - cx
        dy = grid_points[:, 1][None, :] - cy
        kernel = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

        weighted_sum = np.sum(kernel * fr, axis=0)
        weight_sum = np.sum(kernel, axis=0)
        Z[cell] = (weighted_sum / weight_sum).reshape(NSide, NSide)

    sic = np.array([compute_spatial_information(Z[i]) for i in range(Z.shape[0])])
    moran = np.array([compute_morans_I(Z[i], sigma=5.0) for i in range(Z.shape[0])])
    com_spread = np.array([compute_center_of_mass_spread(Z[i], side_length=maxdeg-mindeg) for i in range(Z.shape[0])])
    lap_energy = np.array([compute_normalized_laplacian_energy(Z[i]) for i in range(Z.shape[0])])

    rf_range = np.nanmax(Z, axis=(1, 2)) - np.nanmin(Z, axis=(1, 2))
    rf_mean = np.nanmean(Z, axis=(1, 2))

    ds = xr.Dataset(
        data_vars={'ResponseField': (('cell', 'x', 'y'), Z)},
        coords={'cell': np.arange(n_cells), 'x': x_vals, 'y': y_vals}
    )

    # Add metadata attributes for session 6
    ds.attrs.update({
        'Monkey': 'Jones',
        'Date': pd.Timestamp('2021-10-11'),
        'Session': 'S6',
        'NCells': 138,
        'TinC': [1, 2, 4, 7, 14, 52, 77, 90, 92, 101, 116, 120],
        'TinI': [11, 20, 25, 27, 28, 29, 30, 33, 34, 35, 41, 42, 51, 70, 105, 112, 128],
        'MinC': [10, 31, 102, 115],
        'MinI': [24, 36, 40, 48, 103],
        'Units': {
            'ResponseField': 'Hz',
            'x': 'dva',
            'y': 'dva'
        },
        'ChoiceTargets': {
            'Contra': [-5.1, -6.1],
            'Ipsi': [5.1, 6.1]
        },
        'ResponseFieldRange': rf_range,
        'ResponseFieldMean': rf_mean,
        'SpatialInformation': sic,
        'MoransI': moran,
        'CenterOfMassSpread': com_spread,
        'LaplacianEnergy': lap_energy,
    })

    return ds


def get_weighted_average_response_field(property, Z, heat=None):
    """
    Compute a weighted average response field using a neuron-level property and sharpening via heat scaling.

    Parameters:
        property: np.ndarray of shape (neurons,)
            Array of property values to weight the response field.
        Z: np.ndarray of shape (neurons, pixels)
            Flattened 2D response field per neuron.
        heat: float or None
            Optional sharpening factor applied as exp(heat * Z_normalized). If None, no sharpening.

    Returns:
        np.ndarray of shape (pixels,) representing the weighted average response field.
    """


    if heat is not None:
        # Normalize each neuron's field
        Z_max = np.nanmax(Z, axis=1, keepdims=True)
        Z_normalized = np.where(Z_max > 0, Z / Z_max, 0)
        Z_sharp = np.exp(heat * Z_normalized)
    else:
        Z_sharp = Z

    #Z_sharp = Z_sharp/np.linalg.norm(Z_sharp, axis=1, keepdims=True)
    #Z_sharp = Z_sharp/np.nansum(Z_sharp, axis=1, keepdims=True)
    

    # Weighted average
    numerator = np.nansum(property[:, None] * Z_sharp, axis=0)
    denominator = np.nansum(Z_sharp, axis=0)

    weighted_avg = np.where(denominator > 0, numerator / denominator, 0)
    return weighted_avg


def get_response_field_significance(ds, measure='SpatialInformation', p_lim=0.05, n_shuffles=100):
    """
    Assess significance of spatial structure in response fields using shuffling.

    Parameters:
        ds: xarray.Dataset containing 'ResponseField' and attribute measure per cell
        measure: str, either 'SpatialInformation' or 'MoransI'
        p_lim: float, significance threshold (e.g., 0.05)
        n_shuffles: int, number of permutations per cell

    Adds:
        ds.attrs[f'{measure}Significance']: boolean array of shape (n_cells,)
    """
    assert measure in ['SpatialInformation', 'MoransI'], "Measure must be 'SpatialInformation' or 'MoransI'"
    Z = ds['ResponseField'].values
    n_cells = Z.shape[0]

    if measure == 'SpatialInformation':
        func = compute_spatial_information
    else:
        func = compute_morans_I

    observed = np.array(ds.attrs[measure])
    significance = np.zeros(n_cells, dtype=bool)

    for i in range(n_cells):
        rf = Z[i]
        shuffled_values = []
        for _ in range(n_shuffles):
            shuffled_rf = np.random.permutation(rf.flatten()).reshape(rf.shape)
            shuffled_values.append(func(shuffled_rf))
        shuffled_values = np.array(shuffled_values)
        p_value = np.mean(shuffled_values >= observed[i])
        significance[i] = p_value < p_lim

    ds.attrs[f'{measure}Significance'] = significance

def compute_center_of_mass_spread(RF, side_length=None):
    """
    Compute the spatial spread of mass around the center of mass (CoM) for a 2D response field.
    
    Parameters:
        RF: np.ndarray of shape (x, y), the response field.
        side_length: float or None, optional length of the side of the spatial grid.
    
    Returns:
        float: Spread measure (variance of mass distribution around the CoM).
               Lower values indicate more spatial concentration.
    """
    if np.isnan(RF).all() or np.sum(RF) == 0:
        return np.nan

    RF = np.nan_to_num(RF, nan=0.0)

    total_mass = np.sum(RF)
    x_dim, y_dim = RF.shape
    if side_length is not None:
        x_coords = np.linspace(0, side_length, x_dim)
        y_coords = np.linspace(0, side_length, y_dim)
    else:
        x_coords = np.arange(x_dim)
        y_coords = np.arange(y_dim)
        
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')

    # Center of mass
    com_x = np.sum(xx * RF) / total_mass
    com_y = np.sum(yy * RF) / total_mass

    # Distance from center of mass
    dist_sq = (xx - com_x) ** 2 + (yy - com_y) ** 2

    # Weighted average of squared distance
    spread = np.sum(RF * dist_sq) / total_mass

    return np.sqrt(spread)

def compute_normalized_laplacian_energy(RF, epsilon=1e-6):
    """
    Compute normalized Laplacian energy of a 2D response field.

    Parameters:
        RF: np.ndarray of shape (x, y)
        epsilon: small constant to avoid division by zero

    Returns:
        float: normalized Laplacian energy
    """
    if np.isnan(RF).all():
        return np.nan

    RF = np.nan_to_num(RF, nan=0.0)
    lap = laplace(RF)
    numerator = np.sum(lap**2)
    denominator = np.sum(RF**2) + epsilon

    return numerator / denominator

def find_indices_in_window(time_array, saccade_time, start_offset=0, end_offset=10):
    """
    Helper function to find indices in time_array within a window around saccade_time.
    """
    start_time = saccade_time + start_offset / 1000
    end_time = saccade_time + end_offset / 1000
    return np.where((time_array >= start_time) & (time_array <= end_time))[0]

def get_target_retinotopy(session, target_position):
    """
    Estimate the retinotopic location of a given target based on saccade endpoints of the visually guided task.
    
    Parameters:
        session (str): Session name (e.g., 'S6')
        target_position (list or np.ndarray): [x, y] screen location of the target (in tenths of dva)
    
    Returns:
        list [x, y]: Mean eye position (retinotopic location) or None if file not found
    """

    from src.io_utils import download_session
    path = download_session(f'{session}_visually_guided_task')
    if path is None:
        return None
    with gzip.open(path, 'rb') as f:
        df = pickle.load(f)

    df_filtered = df[df['targ1Pos'].apply(lambda x: np.allclose(x, target_position))]
    if df_filtered.empty:
        return None
    if len(df_filtered) < 10:
        return None

    mean_eye_x_list = []
    mean_eye_y_list = []

    for idx in df_filtered.index:
        eye_data = df_filtered.at[idx, 'eye']
        saccade_time = df_filtered.at[idx, 'saccadeComplete']
        indices = find_indices_in_window(eye_data['t'], saccade_time,start_offset=10, end_offset=80)
        if len(indices) > 0:
            eye_x = eye_data['eyeX'][indices]
            eye_y = eye_data['eyeY'][indices]
            mean_eye_x_list.append(np.nanmean(eye_x))
            mean_eye_y_list.append(np.nanmean(eye_y))

    if len(mean_eye_x_list) == 0:
        return None

    return [np.mean(mean_eye_x_list), np.mean(mean_eye_y_list)]

def get_saccade_endpoints(outlier_percentile=1, filter_correct_trials=True):
    """
    Load and aggregate saccade endpoint data from all sessions.

    Parameters:
        outlier_percentile: float, percentile of RT outliers to remove from both ends (default=1%)
        filter_correct_trials: bool, whether to only include correct trials (default=True)

    Returns:
        pd.DataFrame: Combined trial-wise DataFrame with saccade endpoints and RT.
                      Includes attribute 'targets_screen' with target info per session.
    """
    from collections import Counter
    from src.io_utils import download_session, load_dataframe_with_metadata, suppress_output


    combined_df = []
    targets = []
    sessions = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']

    for session in sessions:
        print(f'Processing session {session}...')

        with suppress_output():
            _ = download_session(session)
            df = load_dataframe_with_metadata(session)

        # Most common target positions
        df['targ1Pos_tuple'] = df['targ1Pos'].apply(lambda x: tuple(x))
        df['targ2Pos_tuple'] = df['targ2Pos'].apply(lambda x: tuple(x))
        T1 = Counter(df['targ1Pos_tuple']).most_common(1)[0][0]
        T2 = Counter(df['targ2Pos_tuple']).most_common(1)[0][0]
        df = df[(df['targ1Pos_tuple'] == T1) & (df['targ2Pos_tuple'] == T2)]
        with suppress_output():
            T1_retino = get_target_retinotopy(session, T1)
            T2_retino = get_target_retinotopy(session, T2)

        # Remove RT outliers
        low = df['RT'].quantile(outlier_percentile / 100)
        high = df['RT'].quantile(1 - outlier_percentile / 100)
        df = df[(df['RT'] >= low) & (df['RT'] <= high)]

        # Filter correct trials if requested
        if filter_correct_trials:
            df = df[(df['coh'] == 0) | (df['correct'] == 1)]

        targets.append({
            'session': session,
            'location_screen': list(np.array(T1) / 10.0),
            'location_retinotopy': T1_retino
        })
        targets.append({
            'session': session,
            'location_screen': list(np.array(T2) / 10.0),
            'location_retinotopy': T2_retino
        })

        # Compute eye positions
        df['EyeX'] = np.nan
        df['EyeY'] = np.nan
        for idx, row in df.iterrows():
            eye = row['eye']
            indices = find_indices_in_window(eye['t'], row['saccadeComplete'])
            if len(indices) > 0:
                df.at[idx, 'EyeX'] = np.mean(eye['eyeX'][indices])
                df.at[idx, 'EyeY'] = np.mean(eye['eyeY'][indices])

        session_df = df[['RT', 'EyeX', 'EyeY', 'choice']].copy()
        session_df = session_df.rename(columns={'choice': 'Choice'})
        session_df['Session'] = session
        combined_df.append(session_df)

    combined_df = pd.concat(combined_df, ignore_index=True)
    combined_df.attrs['targets'] = targets
    # Compute centroids per choice and session and filter trials
    centroids = []
    clean_rows = []
    for session in sessions:
        for choice in [0, 1]:
            subset = combined_df[(combined_df['Session'] == session) & (combined_df['Choice'] == choice)]
            if not subset.empty:
                centroid_x = np.nanmedian(subset['EyeX'])
                centroid_y = np.nanmedian(subset['EyeY'])
                dx = subset['EyeX'] - centroid_x
                dy = subset['EyeY'] - centroid_y
                dists = np.sqrt(dx**2 + dy**2)
                dist_std = np.nanstd(dists)
                inlier_mask = dists <= (5 * dist_std)
                inliers = subset[inlier_mask]
                if not inliers.empty:
                    centroid_x = np.nanmedian(inliers['EyeX'])
                    centroid_y = np.nanmedian(inliers['EyeY'])
                    centroids.append({
                        'session': session,
                        'choice': choice,
                        'location_retinotopy': [centroid_x, centroid_y]
                    })
                    clean_rows.append(inliers)

    combined_df = pd.concat(clean_rows, ignore_index=True)
    combined_df.attrs['centroids'] = centroids
    return combined_df

def coordinate_transformation(df, use_targets=False):
    """
    Transform retinotopic coordinates to a normalized frame:
    - Centered at midpoint between centroids or targets
    - Rotated so line connecting them aligns with x-axis
    - Scaled so centroids/targets lie at -0.5 and 0.5 on x-axis

    Parameters:
        df: pandas DataFrame returned from get_saccade_endpoints
        use_targets: bool, whether to use target retinotopy instead of centroids

    Returns:
        pd.DataFrame: Transformed copy of input DataFrame with updated coordinates
    """
    import copy

    df_new = df.copy()
    df_new.attrs = copy.deepcopy(df.attrs)

    if use_targets:
        ref = []
        pairs = {}
        sessions = set(t['session'] for t in df.attrs['targets'])
        for s in sessions:
            session_targets = [t for t in df.attrs['targets'] if t['session'] == s and t['location_retinotopy'] is not None]
            if len(session_targets) == 2:
                pairs[s] = [t['location_retinotopy'] for t in session_targets]
                ref.extend(session_targets)
            else:
                session_centroids = [c for c in df.attrs['centroids'] if c['session'] == s]
                c0 = next((r['location_retinotopy'] for r in session_centroids if r['choice'] == 0), None)
                c1 = next((r['location_retinotopy'] for r in session_centroids if r['choice'] == 1), None)
                if c0 is not None and c1 is not None:
                    pairs[s] = [c0, c1]
                    ref.extend(session_centroids)
        anchor_type = 'targets if available else centroids'
    else:
        ref = df.attrs['centroids']
        pairs = {}
        sessions = set(r['session'] for r in ref)
        for s in sessions:
            c0 = next((r['location_retinotopy'] for r in ref if r['session'] == s and r['choice'] == 0), None)
            c1 = next((r['location_retinotopy'] for r in ref if r['session'] == s and r['choice'] == 1), None)
            if c0 is not None and c1 is not None:
                pairs[s] = [c0, c1]
        anchor_type = 'centroids'

    for session, pts in pairs.items():
        if len(pts) != 2:
            continue
        pt1, pt2 = pts
        midpoint = (np.array(pt1) + np.array(pt2)) / 2
        direction = np.array(pt2) - np.array(pt1)
        scale = np.linalg.norm(direction)
        angle = np.arctan2(direction[1], direction[0])%np.pi
        cos_theta = np.cos(-angle)
        sin_theta = np.sin(-angle)

        def transform(coord):
            shifted = np.array(coord) - midpoint
            rotated = np.array([
                shifted[0] * cos_theta - shifted[1] * sin_theta,
                shifted[0] * sin_theta + shifted[1] * cos_theta
            ])
            return (rotated / scale).tolist()

        # Transform trial coordinates
        mask = df_new['Session'] == session
        coords = df_new.loc[mask, ['EyeX', 'EyeY']].values
        transformed = np.array([transform(c) for c in coords])
        df_new.loc[mask, 'EyeX'] = transformed[:, 0]
        df_new.loc[mask, 'EyeY'] = transformed[:, 1]

        # Transform centroids
        for centroid in df_new.attrs['centroids']:
            if centroid['session'] == session:
                centroid['location_retinotopy'] = transform(centroid['location_retinotopy'])

        # Transform targets
        for target in df_new.attrs['targets']:
            if target['session'] == session and target['location_retinotopy'] is not None:
                target['location_retinotopy'] = transform(target['location_retinotopy'])

    return df_new

from scipy.stats import wasserstein_distance


def compute_2d_wasserstein(X, Y):
    import ot
    """
    Compute 2D Wasserstein distance using Sinkhorn approximation from POT.
 
    Parameters:
        X, Y: np.ndarray of shape (n_samples, 2), point clouds
        reg: float, entropy regularization parameter for Sinkhorn
 
    Returns:
        float: Sinkhorn distance between X and Y
    """
    reg = 1e-1
    n = X.shape[0]
    m = Y.shape[0]
 
    a = np.ones((n,)) / n
    b = np.ones((m,)) / m
 
    M = ot.dist(X, Y, metric='euclidean')
    M /= M.max()  # normalize for stability
 
    sinkhorn_dist = ot.sinkhorn2(a, b, M, reg)
    return sinkhorn_dist if np.isscalar(sinkhorn_dist) else sinkhorn_dist[0]
    

from scipy.stats import ks_2samp

def compute_saccade_shift_statistics(df, percentile=10, p_lim=0.05, use_targets=False, n_shuffles=1000):
    """
    Compute session- and choice-specific statistics on RT-dependent shifts in saccade endpoints.
    
    Parameters:
        df: pd.DataFrame from get_saccade_endpoints()
        percentile: float, percentile to define fast and slow RT groups
        p_lim: float, significance level for permutation tests
        use_targets: bool, whether to use target locations for coordinate transformation
        n_shuffles: int, number of shuffles for significance testing
    
    Returns:
        pd.DataFrame with one row per (session, choice) with:
            - target_distance
            - 2D Wasserstein distance + significance
            - KS statistics for projections on the target line (one-sided, testing if slow saccades are inbetween fast and the other target) and orthogonal to the target line (two-sided, testing if slow and fast are different) + significances
    """

    df_trans = coordinate_transformation(df, use_targets=use_targets)
    sessions = df['Session'].unique()
    results = []

    for session in sessions:
        for choice in [0, 1]:
            sub = df[df['Session'] == session]
            sub_trans = df_trans[df_trans['Session'] == session]

            if sub.empty or sub_trans.empty:
                continue

            # Compute target distance (same for both choices)
            targets = df.attrs['targets']
            sess_targets = [t['location_screen'] for t in targets if t['session'] == session]
            if len(sess_targets) != 2:
                continue
            c0, c1 = sess_targets
            target_dist = np.linalg.norm(np.array(c0) - np.array(c1))

            df_choice = sub[sub['Choice'] == choice]
            df_choice_trans = sub_trans[sub_trans['Choice'] == choice]

            if len(df_choice) < 10:
                continue

            rt_sorted = df_choice['RT'].sort_values()
            low_thresh = rt_sorted.quantile(percentile / 100)
            high_thresh = rt_sorted.quantile(1 - percentile / 100)

            slow = df_choice[df_choice['RT'] >= high_thresh]
            fast = df_choice[df_choice['RT'] <= low_thresh]
            slow_trans = df_choice_trans[df_choice_trans['RT'] >= high_thresh]
            fast_trans = df_choice_trans[df_choice_trans['RT'] <= low_thresh]

            if slow.empty or fast.empty:
                continue

            # --- 2D Wasserstein ---
            slow_xy = slow[['EyeX', 'EyeY']].dropna().values
            fast_xy = fast[['EyeX', 'EyeY']].dropna().values
            if len(slow_xy) == 0 or len(fast_xy) == 0:
                continue

            wass2d = compute_2d_wasserstein(slow_xy, fast_xy)

            # Shuffle test for 2D Wasserstein
            combined = np.vstack([slow_xy, fast_xy])
            labels = np.array([0] * len(slow_xy) + [1] * len(fast_xy))
            null_dist = []
            for _ in range(n_shuffles):
                np.random.shuffle(labels)
                s_idx = labels == 0
                f_idx = labels == 1
                if s_idx.sum() > 0 and f_idx.sum() > 0:
                    ws = compute_2d_wasserstein(combined[s_idx], combined[f_idx])
                    null_dist.append(ws)
            wass_signif = np.mean(np.array(null_dist) >= wass2d) < p_lim

            # --- 1D KS stats in transformed coords ---
            xs_slow = slow_trans[['EyeX']].dropna().values.ravel()
            xs_fast = fast_trans[['EyeX']].dropna().values.ravel()
            ys_slow = slow_trans[['EyeY']].dropna().values.ravel()
            ys_fast = fast_trans[['EyeY']].dropna().values.ravel()

            if choice == 0:
                ks_x, p_x = ks_2samp(xs_slow, xs_fast, alternative='less')
            else:
                ks_x, p_x = ks_2samp(xs_slow, xs_fast, alternative='greater')
            ks_y, p_y = ks_2samp(ys_slow, ys_fast)

            results.append({
                'session': session,
                'choice': choice,
                'target_distance': target_dist,
                'n_samples': min(len(slow_xy), len(fast_xy)),
                'wasserstein_2d': wass2d,
                'wasserstein_significant': wass_signif,
                'KS_on_target_line': ks_x,
                'KS_on_target_line_significant': p_x < p_lim,
                'KS_off_target_line': ks_y,
                'KS_off_target_line_significant': p_y < p_lim
            })

    return pd.DataFrame(results)






