
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go

def get_pca(df, column='LFADS', cells=None, components=10, plot=True):
    """
    Performs PCA on neural trajectories stored in df[column], optionally on a subset of cells.

    Parameters:
        df (pd.DataFrame): DataFrame containing neural data.
        column (str): Column name in df to perform PCA on. Each row in this column must be a (neurons × time) numpy array. All arrays must have the same number of neurons, but may have varying numbers of time points.
        cells (list or None): Optional list of cell indices to include.
        components (int): Number of PCA components.
        plot (bool): Whether to plot explained variance using Plotly.

    Returns:
        pca (PCA object): Trained PCA object for further analysis.
    """

    # Accumulate data
    data_list, shapes = [], []
    for trial in df.index:
        data = df.at[trial, column]
        if cells is not None:
            data = data[cells, :]
        data_list.append(data.T)
        shapes.append(data.shape[1])

    data_concat = np.concatenate(data_list, axis=0)

    # Standardize data
    scaler = StandardScaler(with_std=False)
    data_scaled = scaler.fit_transform(data_concat)

    # PCA
    pca = PCA(n_components=components)
    pca_transformed = pca.fit_transform(data_scaled)

    # Add mean back in PCA space
    mean_transformed = pca.transform(scaler.mean_.reshape(1, -1))
    pca_transformed += mean_transformed

    # Store PCA back in dataframe
    cum_shapes = np.insert(np.cumsum(shapes), 0, 0)
    df[f'{column}-PCA'] = None
    for i, trial in enumerate(df.index):
        df.at[trial, f'{column}-PCA'] = pca_transformed[cum_shapes[i]:cum_shapes[i+1], :components].T

    # Explained variance plot
    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(1, components + 1),
            y=np.cumsum(pca.explained_variance_ratio_),
            mode='lines+markers',
            line=dict(color='black')
        ))

        participation_ratio = (np.sum(pca.explained_variance_)**2) / np.sum(pca.explained_variance_**2)
        fig.update_layout(
            title=f'Cumulative Explained Variance ({column})',
            xaxis_title='Principal Component',
            yaxis_title='Cumulative Explained Variance',
            width=500,  # Adjust the width here (e.g., 400–600 for narrow plot)
            height=400, # Adjust height if desired
            annotations=[dict(
                x=components * 0.5, y=0.1, text=f'Participation Ratio: {participation_ratio:.1f}',
                showarrow=False, bgcolor='white'
            )]
        )
        fig.show()


    return pca




def compute_angle(a, b):
    """
    Computes the angles (in radians) between vectors a and b.

    This function supports both single vectors (1D arrays) and multiple vectors (2D arrays).
    For 2D arrays, vectors are expected along the first dimension (rows).

    Parameters:
        a (np.ndarray): A NumPy array representing the first vector or set of vectors.
        b (np.ndarray): A NumPy array representing the second vector or set of vectors.

    Returns:
        np.ndarray or float:
            - If inputs are 1D arrays, returns a single float value (angle in radians).
            - If inputs are 2D arrays, returns a 1D array of angles (radians) between corresponding pairs of vectors.

    Raises:
        ValueError: If the shapes of `a` and `b` do not match.

    Examples:
        >>> compute_angle(np.array([1, 0]), np.array([0, 1]))
        1.5707963267948966  # pi/2 radians (90 degrees)
        
        >>> compute_angle(np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]))
        array([1.57079633, 1.57079633])
    """
    if a.shape != b.shape:
        raise ValueError("The shapes of `a` and `b` must be identical.")

    # Efficient handling for both 1D and 2D using numpy.einsum
    dot_product = np.einsum('...i,...i->...', a, b)
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)

    # Calculate and clamp cosine values
    cos_theta = np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0)

    return np.arccos(cos_theta)


def der(arr, axis=-1):
    """
    Compute the numerical derivative of an N-dimensional array using second-order finite differences
    along the specified axis.

    This method uses central differences for interior points (second-order accurate),
    and forward/backward differences at boundaries.

    Parameters:
        arr (np.ndarray): Input N-dimensional array.
        axis (int): Axis along which to compute the derivative (default: -1, the last axis).

    Returns:
        np.ndarray: Array of derivatives, same shape as input.

    Examples:
        >>> arr = np.array([[1, 2, 4, 7], [0, 1, 3, 6]])
        >>> der(arr, axis=1)
        array([[1. , 1.5, 2.5, 3. ],
               [1. , 1.5, 2.5, 3. ]])
    """
    arr = np.asarray(arr)
    if arr.shape[axis] < 3:
        raise ValueError("Input array must have at least 3 points along the specified axis.")

    derivative = np.empty_like(arr, dtype=float)

    # Prepare slicing
    slice_all = [slice(None)] * arr.ndim

    # Forward difference (2nd-order accuracy) for the first point
    slice_all[axis] = 0
    idx_0 = tuple(slice_all)
    
    slice_all[axis] = 1
    idx_1 = tuple(slice_all)
    
    slice_all[axis] = 2
    idx_2 = tuple(slice_all)

    derivative[idx_0] = (-3 * arr[idx_0] + 4 * arr[idx_1] - arr[idx_2]) / 2

    # Central differences for interior points
    slice_all[axis] = slice(2, None)
    idx_plus = tuple(slice_all)

    slice_all[axis] = slice(None, -2)
    idx_minus = tuple(slice_all)

    slice_all[axis] = slice(1, -1)
    idx_center = tuple(slice_all)

    derivative[idx_center] = (arr[idx_plus] - arr[idx_minus]) / 2

    # Backward difference (2nd-order accuracy) for the last point
    slice_all[axis] = -1
    idx_n = tuple(slice_all)
    
    slice_all[axis] = -2
    idx_n1 = tuple(slice_all)
    
    slice_all[axis] = -3
    idx_n2 = tuple(slice_all)

    derivative[idx_n] = (3 * arr[idx_n] - 4 * arr[idx_n1] + arr[idx_n2]) / 2

    return derivative




import scipy.ndimage

def K(curve, w=1, zoom=1):
    """
    Computes curvature along a given curve using a moving-window chord-length method.

    Optionally resamples the curve for smoother curvature estimation using interpolation.

    Parameters:
        curve (np.ndarray): A 2D array of shape (dimensions, points), representing the curve coordinates.
        w (int): Window size for computing curvature at each point.
        zoom (float): Resampling factor (>1 for increased resolution, 1 for original resolution).

    Returns:
        np.ndarray: Array containing curvature values at each point along the curve.

    Notes:
        - Curvature at boundary points uses asymmetric windows to avoid edge issues.
        - Points with insufficient arc length to reliably estimate curvature are assigned NaN.

    Examples:
        >>> curve = np.array([[0, 1, 2, 3], [0, 1, 0, -1]])  # simple curve
        >>> K(curve, w=1)
        array([nan, 2.30940108, 2.30940108, nan])
    """
    curve = np.asarray(curve)

    if zoom != 1:
        curve = scipy.ndimage.zoom(curve, [1, zoom], order=3, mode='nearest')
        w = int(zoom * w)

    n_points = curve.shape[1]
    k = np.empty(n_points, dtype=float)

    for i in range(n_points):
        # Adjust window size near boundaries
        w_left = min(w, i)
        w_right = min(w, n_points - 1 - i)

        # Segment selection
        seg = curve[:, i - w_left:i + w_right + 1]

        # Calculate arc length (total length along the curve segment)
        deltas = np.diff(seg, axis=1)
        segment_lengths = np.linalg.norm(deltas, axis=0)
        arc_length = np.sum(segment_lengths)

        # Calculate chord length (straight-line distance)
        chord_length = np.linalg.norm(seg[:, -1] - seg[:, 0])

        # Curvature estimation with numerical stability
        if arc_length > 1e-6:
            curvature_sq = 24 * (arc_length - chord_length) / arc_length**3
            k[i] = np.sqrt(curvature_sq) if curvature_sq > 0 else 0
        else:
            k[i] = np.nan  # Insufficient data, unreliable curvature estimation

    if zoom != 1:
        k = scipy.ndimage.zoom(k, 1 / zoom, order=3, mode='nearest')

    return k



import numpy as np
import scipy.ndimage

def Curvatures(dat, s=5):
    """
    Compute generalized curvatures of neural trajectories from multivariate data.

    Parameters:
        dat (np.ndarray): Array of shape (neurons, time), representing the trajectory.
        s (float): Standard deviation for Gaussian smoothing applied after derivatives.

    Returns:
        np.ndarray: Array of shape (neurons-1, time) containing curvature values.

    Notes:
        - Computes derivatives up to order equal to the number of neurons.
        - Uses Gaussian smoothing to stabilize derivatives.
        - Handles numerical instabilities by setting NaNs and infinities to zero.

    Examples:
        >>> data = np.random.rand(3, 100)  # 3 neurons over 100 time points
        >>> K = Curvatures(data, s=3)
        >>> K.shape
        (2, 100)
    """
    num_neurons, num_timepoints = dat.shape

    # Compute derivatives with smoothing
    D = np.empty((num_neurons, num_neurons, num_timepoints))
    derivative_data = dat.copy()

    for order in range(num_neurons):
        derivative_data = np.gradient(derivative_data, axis=1, edge_order=2)
        derivative_data = scipy.ndimage.gaussian_filter1d(derivative_data, sigma=s, axis=1)
        D[order] = derivative_data

    # Compute Gram determinants
    G = np.empty((num_neurons, num_timepoints))
    for t in range(num_timepoints):
        for p in range(num_neurons):
            mat = D[:p+1, :, t]
            G[p, t] = np.linalg.det(mat @ mat.T)

    # Compute curvatures from Gram determinants
    K = np.zeros((num_neurons - 1, num_timepoints))
    epsilon = 1e-10  # small constant to avoid division by zero
    for p in range(num_neurons - 1):
        with np.errstate(divide='ignore', invalid='ignore'):
            if p == 0:
                numerator = np.sqrt(np.abs(G[1]))
                denominator = G[0] ** 1.5 + epsilon
            else:
                numerator = np.sqrt(np.abs(G[p - 1] * G[p + 1]))
                denominator = (G[p] ** 2 + epsilon) / G[0]

            curvature = numerator / denominator
            curvature[~np.isfinite(curvature)] = 0  # Set NaN and inf to zero
            K[p] = curvature

    return K



import numpy as np
import scipy.ndimage

def Tortuosity(curvature, zoom=10, smoothing_sigma=75):
    """
    Computes the tortuosity of a curve from its curvature profile.

    Tortuosity captures the complexity of the curvature, indicating rapid changes.

    Parameters:
        curvature (np.ndarray): 1D array containing curvature values along the curve.
        zoom (int): Resampling factor for increasing resolution during intermediate computations.
        smoothing_sigma (float): Sigma for Gaussian smoothing applied to tortuosity.

    Returns:
        np.ndarray: Tortuosity values, same shape as the input curvature array.

    Notes:
        - Uses spline interpolation for stable up/down sampling.
        - Gaussian smoothing reduces noise and stabilizes derivative computation.

    Examples:
        >>> curvature = np.random.rand(100)
        >>> tort = Tortuosity(curvature)
        >>> tort.shape
        (100,)
    """
    # Upsample curvature to improve derivative stability
    curvature_zoomed = scipy.ndimage.zoom(curvature, zoom, order=3, mode='nearest')

    # Compute derivative using generalized `der()` function
    dk = der(curvature_zoomed)

    # Compute tortuosity with stability handling
    epsilon = 1e-10  # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tort = (dk / (curvature_zoomed + epsilon)) ** 2
        tort[~np.isfinite(tort)] = 0  # Replace NaNs and inf with zero

    # Smooth tortuosity profile
    tort_smooth = scipy.ndimage.gaussian_filter1d(tort, sigma=smoothing_sigma)

    # Downsample back to original resolution
    tort_final = scipy.ndimage.zoom(tort_smooth, 1 / zoom, order=3, mode='nearest')

    return tort_final



def Dis(r):
    """
    Computes distances between consecutive points along a trajectory and returns averaged distances at each point.

    Parameters:
        r (np.ndarray): A 2D array of shape (dimensions, points), representing points along a trajectory.

    Returns:
        np.ndarray: Array of averaged distances for each point, length equal to the number of points.

    Examples:
        >>> r = np.array([[0, 1, 2], [0, 1, 0]])
        >>> Dis(r)
        array([1.41421356, 1.70710678, 1.41421356])
    """
    # Compute pairwise distances between consecutive points
    d = np.linalg.norm(np.diff(r, axis=1), axis=0)
    
    # Pad distances at the start and end
    d_padded = np.pad(d, (1, 1), 'edge')
    
    # Return averaged distances for smooth representation
    return 0.5 * (d_padded[:-1] + d_padded[1:])




def CumDis(r):
    """
    Computes cumulative distances along a trajectory from the starting point.

    Parameters:
        r (np.ndarray): A 2D array of shape (dimensions, points), representing points along a trajectory.

    Returns:
        np.ndarray: Array of cumulative distances at each point along the trajectory.

    Examples:
        >>> r = np.array([[0, 1, 2], [0, 1, 0]])
        >>> CumDis(r)
        array([0.        , 1.41421356, 2.82842712])
    """
    # Compute pairwise distances between consecutive points and cumulative sum
    cumulative_distances = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(r, axis=1), axis=0))))

    return cumulative_distances



def tangent_space(data, axis=-1):
    """
    Compute the unit tangent vectors along a specified axis of an N-dimensional array.

    Parameters:
        data (np.ndarray): Input array of arbitrary shape (..., points, ...).
        axis (int): Axis along which to compute tangent vectors. Defaults to the last axis.

    Returns:
        np.ndarray: Array of unit tangent vectors with the same shape as `data`.

    Notes:
        - Uses the previously defined `der()` function for consistent derivative estimation.
        - Tangent vectors at each point are normalized to unit magnitude.
        - Handles zero-magnitude vectors gracefully by leaving them as zero.

    Examples:
        >>> data = np.array([[1, 2, 4, 7], [0, 1, 3, 6]])
        >>> tangent_space(data, axis=1)
        array([[0.70710678, 0.70710678, 0.70710678, 0.70710678],
               [0.70710678, 0.70710678, 0.70710678, 0.70710678]])
    """
    # Compute derivatives using existing der() function
    tangent_vectors = der(data, axis=axis)

    # Compute magnitudes along the specified axis
    magnitudes = np.linalg.norm(tangent_vectors, axis=0, keepdims=True)

    # Handle zero-magnitude vectors to avoid division by zero
    magnitudes[magnitudes == 0] = 1.0

    # Normalize to get unit tangent vectors
    unit_tangent_vectors = tangent_vectors / magnitudes

    return unit_tangent_vectors
    
    


import numpy as np

def compute_unit_orthogonal_vectors(a, b, axis=0):
    """
    Compute unit vectors orthogonal to vectors `a` that lie within the plane spanned by vectors `a` and `b`.

    This function works for arrays of arbitrary dimensionality, computing orthogonalization along a specified axis.

    Parameters:
        a (np.ndarray): First set of vectors. Shape should match `b`.
        b (np.ndarray): Second set of vectors, same shape as `a`.
        axis (int): Axis along which vectors are defined (default is 0).

    Returns:
        np.ndarray: Array of unit vectors orthogonal to `a`, lying in the plane spanned by `a` and `b`. 
                    The returned array has the same shape as `a` and `b`.

    Notes:
        - Handles division by zero gracefully by returning zero vectors where orthogonalization is undefined.
        - Input arrays must be broadcast-compatible and have matching shapes.

    Examples:
        >>> a = np.array([[1, 0], [0, 1]])
        >>> b = np.array([[1, 1], [-1, 1]])
        >>> compute_unit_orthogonal_vectors(a, b, axis=0)
        array([[ 0.        ,  0.70710678],
               [-1.        ,  0.70710678]])
    """
    # Compute projection of b onto a
    dot_a_b = np.sum(a * b, axis=axis, keepdims=True)
    dot_a_a = np.sum(a * a, axis=axis, keepdims=True)

    # Avoid division by zero in projection calculation
    dot_a_a_safe = np.where(dot_a_a == 0, 1, dot_a_a)

    projection = (dot_a_b / dot_a_a_safe) * a

    # Orthogonal component of b relative to a
    orthogonal_vector = b - projection

    # Compute norms along the specified axis
    norm = np.linalg.norm(orthogonal_vector, axis=axis, keepdims=True)

    # Handle division by zero for normalization
    norm_safe = np.where(norm == 0, 1, norm)

    # Normalize to obtain unit orthogonal vectors
    unit_orthogonal_vector = orthogonal_vector / norm_safe

    return unit_orthogonal_vector



import scipy.interpolate

def compute_geometry_measures(df, column='LFADS-PCA', timeres=10, arc_res=101):
    """
    Computes geometric measures and interpolates them along normalized arc-length.

    Parameters:
        df (pd.DataFrame): DataFrame containing neural trajectories.
        column (str): Column name with neural trajectories.
        timeres (float): Temporal resolution (ms per point).
        arc_res (int): Points for interpolation along normalized arc-length.

    Returns:
        None. Modifies DataFrame in place.
    """
    newx = np.linspace(0, 1, arc_res, endpoint=True)

    # Initialize columns explicitly as object types to hold arrays
    array_cols = ['Time', 'Speed', 'Arc-Length-Proper', 'Arc-Length',
                  f'{column}-Arc', 'Speed-Arc', 'Time-Arc',
                  'Curvature-Arc', 'Tortuosity-Arc']

    for col in array_cols:
        df[col] = [None] * len(df)

    for idx, row in df.iterrows():
        traj = row[column]

        # Compute measures
        time = np.arange(traj.shape[1]) * timeres
        speed = Dis(traj) / timeres
        arc_length_proper = CumDis(traj)
        arc_length_norm = arc_length_proper / (arc_length_proper[-1] if arc_length_proper[-1] != 0 else 1)

        # Store measures in DataFrame
        df.at[idx, 'Time'] = time
        df.at[idx, 'Speed'] = speed
        df.at[idx, 'Arc-Length-Proper'] = arc_length_proper
        df.at[idx, 'Arc-Length'] = arc_length_norm

        # Helper interpolation function
        def interp(y):
            f = scipy.interpolate.interp1d(
                arc_length_norm, y, axis=-1, kind='cubic', fill_value=np.nan, bounds_error=False
            )
            return f(newx)

        # Interpolated measures
        df.at[idx, f'{column}-Arc'] = interp(traj)
        df.at[idx, 'Speed-Arc'] = interp(speed)
        df.at[idx, 'Time-Arc'] = interp(time)

        # Curvature and Tortuosity computations
        curvature = K(df.at[idx, f'{column}-Arc'], w=5)
        tortuosity = Tortuosity(curvature)

        df.at[idx, 'Curvature-Arc'] = curvature
        df.at[idx, 'Tortuosity-Arc'] = tortuosity





def get_components(df,
                   tangent_ref,
                   conditions,
                   behavioral_axis: str = 'RT',
                   new_behavioral_axis=None,
                   trajectories: str = 'LFADS-PCA-Arc',
                   name: str = 'Resolution'):
    """
    Compute the projection of each trial’s unit tangent vector onto a reference tangent space,
    and store both the raw and squared projections as new columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): Trial-wise DataFrame with each row containing a time-varying
                           neural trajectory in column `trajectories`. Each value must be a
                           NumPy array of shape (N, T), where:
                               - N = number of dimensions (e.g., PCA components),
                               - T = number of timepoints (e.g., 101 arc-length steps).
        
        tangent_ref (np.ndarray): Reference tangent space of shape (N, T, B), where:
                                  - N = dimensions,
                                  - T = arc-length points,
                                  - B = number of bins along the behavioral axis.

        conditions (list of pd.Series): List of boolean masks for selecting relevant trials.
                                        They will be combined with logical AND.

        behavioral_axis (str): Name of the column in `df` that holds a scalar behavioral
                               value per trial (e.g., 'RT').

        new_behavioral_axis (np.ndarray): 1D array of behavioral bin values (length B),
                                          used to align each trial to the nearest slice in
                                          the reference tangent space.

        trajectories (str): Name of the column in `df` that contains neural trajectories.
                            Each entry must be a (N × T) NumPy array.

        name (str): Base name used for new columns:
                    - f"{name}Projection": the dot product per timepoint (length T).
                    - f"{name}SquareProjection": the squared dot product per timepoint.

    Returns:
        None. Modifies `df` in place by adding two new columns per trial.
    """

    # Initialize output columns
    df[f"{name}Projection"] = None
    df[f"{name}SquareProjection"] = None

    combined_condition = np.logical_and.reduce(conditions)

    for idx in df[combined_condition].index:

        traj = df.at[idx, trajectories]  # shape: N x T
        trial_tangent = tangent_space(traj, axis=1)  # shape: N x T

        bval = df.loc[idx, behavioral_axis]
        b_idx = np.argmin(np.abs(new_behavioral_axis - bval))
        ref_tangent = tangent_ref[:, :, b_idx]  # shape: N x T

        projection = np.sum(trial_tangent * ref_tangent, axis=0)  # shape: T
        df.at[idx, f"{name}Projection"] = projection
        df.at[idx, f"{name}SquareProjection"] = projection**2




################################ OTHER TOOLS #########################################


import numpy as np

def uniformize(x):
    """
    Transforms an array's values to a uniform distribution, preserving the original order of values.

    Useful for normalizing data, making it suitable for algorithms that perform better with uniformly distributed input.

    Parameters:
        x (np.ndarray): 1D input array to transform.

    Returns:
        np.ndarray: Transformed array with values uniformly distributed between 0 and 1.

    Examples:
        >>> x = np.array([3, 1, 2])
        >>> uniformize(x)
        array([1.        , 0.33333333, 0.66666667])
    """
    # Obtain ranks for each element
    ranks = np.argsort(np.argsort(x))

    # Normalize ranks to [0, 1]
    uniform_values = (ranks + 1) / len(x)

    return uniform_values



import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
from tqdm import tqdm

def local_average(df, column, behavioral_axis='RT', frac=0.5, ba_res=101, method='lowess', remove_outliers=0.01):
    """
    Compute smoothed local averages of any continuous neural measure (e.g., trajectories, curvature, speed),
    aligned to a continuous behavioral measure, parameterized by normalized arc-length.

    Parameters:
        df (pd.DataFrame): DataFrame containing neural measures parameterized by arc-length.
        column (str): Column with neural measure data (e.g., 'LFADS-PCA-Arc', 'Curvature-Arc', 'Speed-Arc').
                      Shape: (neurons/components × arc-length points) or (arc-length points,).
        behavioral_axis (str): Column containing scalar behavioral measure per trial (e.g., 'RT').
        frac (float): Fraction (0–1) of data for LOWESS smoothing.
        ba_res (int): Number of points for interpolation along behavioral axis (behavioral axis resolution).
        method (str): Smoothing method: 'lowess' or 'cdf-lowess'.
        remove_outliers (float or None): Fraction of extreme behavioral values to remove from both tails (top and bottom).
                                         If None or 0, no outliers are removed.

    Returns:
        Tuple:
            - interpolated_array: Interpolated neural measures aligned to behavioral axis.
                                  Shape: (neurons/components, arc-length points, ba_res) or (arc-length points, ba_res).
            - behavioral_axis_new: Interpolated behavioral axis values.

    Examples:
        >>> interpolated_data, new_behavioral_axis = local_average(
                df=df,
                column='Curvature-Arc',
                behavioral_axis='RT',
                frac=0.4,
                ba_res=100,
                remove_outliers=0.01
            )
    """

    # Optionally remove top and bottom quantile outliers based on behavioral measure
    if remove_outliers and remove_outliers > 0:
        lower_threshold = df[behavioral_axis].quantile(remove_outliers)
        upper_threshold = df[behavioral_axis].quantile(1 - remove_outliers)
        df_filtered = df[(df[behavioral_axis] >= lower_threshold) &
                         (df[behavioral_axis] <= upper_threshold)]
    else:
        df_filtered = df

    # Define new behavioral axis for interpolation
    behavioral_min = df_filtered[behavioral_axis].min()
    behavioral_max = df_filtered[behavioral_axis].max()
    behavioral_axis_new = np.linspace(behavioral_min, behavioral_max, ba_res)

    # Determine data shape from a sample entry
    sample_array = df_filtered[column].iloc[0]
    if np.ndim(sample_array) == 1:
        N, L = 1, len(sample_array)
        interpolated_array = np.zeros((1, L, ba_res))
    else:
        N, L = sample_array.shape
        interpolated_array = np.zeros((N, L, ba_res))

    total_iterations = N * L
    with tqdm(total=total_iterations, desc="Interpolating") as pbar:
        for n in range(N):
            for l in range(L):
                # Extract values and behavioral measure for each dimension and arc-length point
                values = np.array([
                    trial[n, l] if N > 1 else trial[l]
                    for trial in df_filtered[column]
                ])
                behaviors = df_filtered[behavioral_axis].values

                # Apply LOWESS smoothing
                if method == 'lowess':
                    smoothed = lowess(values, behaviors, frac=frac, return_sorted=True)
                    behaviors_smoothed, values_smoothed = smoothed[:, 0], smoothed[:, 1]
                elif method == 'cdf-lowess':
                    behaviors_uniform = uniformize(behaviors)
                    smoothed = lowess(values, behaviors_uniform, frac=frac, return_sorted=True)
                    behaviors_smoothed, values_smoothed = np.sort(behaviors), smoothed[:, 1]
                else:
                    raise ValueError(f"Unknown method '{method}'. Choose 'lowess' or 'cdf-lowess'.")

                # Cubic spline interpolation of the smoothed data
                interp = interp1d(
                    behaviors_smoothed,
                    values_smoothed,
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )

                interpolated_values = interp(behavioral_axis_new)

                interpolated_array[n, l, :] = interpolated_values
                pbar.update(1)

    if N == 1:
        interpolated_array = interpolated_array.squeeze(axis=0)

    return interpolated_array, behavioral_axis_new

