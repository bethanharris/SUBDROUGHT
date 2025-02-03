import iris
from iris.util import equalise_attributes
import os
import numpy as np
import xarray as xr
import cf_units
import dask.array as da
from tqdm import tqdm
from scipy.stats import linregress


def detrend_missing_values(data):
    """
    Detrend data array containing missing values.

    Parameters
    ----------
    data: 1D numpy array or dask array
        Data to detrend
    Returns
    -------
    Detrended array of same type as data
    """
    x = np.arange(data.size)
    valid_idx = ~np.isnan(data)
    if valid_idx.sum() > 0:
        m, b, r_val, p_val, std_err = linregress(x[valid_idx], data[valid_idx])
        detrended_data = data - (m*x + b)
    else:
        detrended_data = data
    return detrended_data


def detrend_cube(cube, dimension='time'):
    """
    Detrend Iris cube of data along a given dimension. Permits missing values.

    Parameters
    ----------
    cube: iris.cube.Cube
        Cube containing data to detrend
    dimension: str
        Dimension along which to detrend
    Returns
    -------
    iris.cube.Cube
        Detrended cube
    """
    coord = cube.coord(dimension)
    axis = cube.coord_dims(coord)[0]
    detrended = da.apply_along_axis(
        detrend_missing_values,
        axis=axis,
        arr=cube.lazy_data().rechunk([-1, 80, 80]),
        shape=(cube.shape[axis],)
    )
    return cube.copy(detrended)


def daily_anomalies_standardised_rolling(cube, detrend=False, mean_rolling_window=7, std_rolling_window=7):
    """
    Compute standardised anomalies for daily data, with options to detrend data first and to 
    set windows for running mean and running standard deviation.

    Parameters
    ----------
    cube: iris.cube.Cube
        Cube containing data from which to compute standardised anomalies. Also accepts xarray DataArray.
    detrend (kwarg): bool, default True
        Apply linear detrending in time before computing anomalies
    mean_rolling_window (kwarg): int, default 7
        Width of rolling window for which to compute climatological mean for anomalies.
        Window is centred on date (e.g. on 15th June, a window of 7 will use data from 12th June to 18th June inclusive
                                   to compute the climatological mean).
    std_rolling_window (kwarg): int, default 7
        Width of rolling window for which to compute climatological standard deviation for standardising anomalies.
        Window is centred on date (e.g. on 15th June, a window of 7 will use data from 12th June to 18th June inclusive
                                   to compute the climatological standard deviation).
    Returns
    -------
    xarray.DataArray
        Array of standardised anomalies.
    """
    if isinstance(cube, xr.core.dataset.Dataset) or isinstance(cube, xr.core.dataarray.DataArray):
        cube = cube.to_iris()
    if detrend:
        cube = detrend_cube(cube)
    # else:
    #     cube = cube.lazy_data().rechunk([-1, 80, 80])
    # Compute standardised anomalies using rolling windows for climatological mean and standard deviation
    dxr = xr.DataArray.from_iris(cube)
    month_day_str = xr.DataArray(dxr.indexes['time'].strftime('%m-%d'), coords=dxr.coords['time'].coords, name='month_day_str')
    rolling_dxr_mean = dxr.rolling(min_periods=1, center=True, time=mean_rolling_window).construct("window_mean")
    rolling_dxr_std = dxr.rolling(min_periods=1, center=True, time=std_rolling_window).construct("window_std")
    grouped_mean_rolling = rolling_dxr_mean.groupby(month_day_str)
    grouped_std_rolling = rolling_dxr_std.groupby(month_day_str)
    climatology_mean = grouped_mean_rolling.mean(["time", "window_mean"])
    climatology_std = grouped_std_rolling.std(["time", "window_std"])
    anomalies_xr = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        dxr.groupby(month_day_str),
        climatology_mean,
        climatology_std, dask='allowed'
    )
    # Fix coordinate names in xarray to prevent naming mismatches
    coord_names = anomalies_xr.coords._names
    if 'lat' in coord_names:
        anomalies_xr = anomalies_xr.rename({'lat': 'latitude'})
    if 'lon' in coord_names:
        anomalies_xr = anomalies_xr.rename({'lon': 'longitude'})
    # Ensure final cube of standardised anomalies has consistent metadata
    anomalies = anomalies_xr.to_iris()
    anomalies.standard_name = cube.standard_name
    anomalies.long_name = cube.long_name
    anomalies.units = cube.units
    calendar = anomalies.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    anomalies.coord('time').convert_units(common_time_unit)
    for coord_key in ['time', 'latitude', 'longitude']:
        if anomalies.coord(coord_key).points.size > 1:
            anomalies.coord(coord_key).bounds = None
            anomalies.coord(coord_key).guess_bounds()
            anomalies.coord(coord_key).bounds = np.round(anomalies.coord(coord_key).bounds, 3)
            anomalies.coord(coord_key).points = np.round(anomalies.coord(coord_key).points, 3)
    return xr.DataArray.from_iris(anomalies).drop_vars('month_day_str')


def daily_anomalies_rolling(cube, detrend=False, mean_rolling_window=7):
    """
    Compute standardised anomalies for daily data, with options to detrend data first and to 
    set windows for running mean and running standard deviation.

    Parameters
    ----------
    cube: iris.cube.Cube
        Cube containing data from which to compute standardised anomalies. Also accepts xarray DataArray.
    detrend (kwarg): bool, default True
        Apply linear detrending in time before computing anomalies
    mean_rolling_window (kwarg): int, default 7
        Width of rolling window for which to compute climatological mean for anomalies.
        Window is centred on date (e.g. on 15th June, a window of 7 will use data from 12th June to 18th June inclusive
                                   to compute the climatological mean).
    Returns
    -------
    xarray.DataArray
        Array of standardised anomalies.
    """
    if isinstance(cube, xr.core.dataset.Dataset) or isinstance(cube, xr.core.dataarray.DataArray):
        cube = cube.to_iris()
    if detrend:
        cube = detrend_cube(cube)
    # else:
    #     cube = cube.lazy_data().rechunk([-1, 80, 80])
    # Compute standardised anomalies using rolling windows for climatological mean and standard deviation
    dxr = xr.DataArray.from_iris(cube)
    month_day_str = xr.DataArray(dxr.indexes['time'].strftime('%m-%d'), coords=dxr.coords['time'].coords, name='month_day_str')
    rolling_dxr_mean = dxr.rolling(min_periods=1, center=True, time=mean_rolling_window).construct("window_mean")
    grouped_mean_rolling = rolling_dxr_mean.groupby(month_day_str)
    climatology_mean = grouped_mean_rolling.mean(["time", "window_mean"])
    anomalies_xr = xr.apply_ufunc(
        lambda x, m: (x - m),
        dxr.groupby(month_day_str),
        climatology_mean, dask='allowed'
    )
    # Fix coordinate names in xarray to prevent naming mismatches
    coord_names = anomalies_xr.coords._names
    if 'lat' in coord_names:
        anomalies_xr = anomalies_xr.rename({'lat': 'latitude'})
    if 'lon' in coord_names:
        anomalies_xr = anomalies_xr.rename({'lon': 'longitude'})
    # Ensure final cube of standardised anomalies has consistent metadata
    anomalies = anomalies_xr.to_iris()
    anomalies.standard_name = cube.standard_name
    anomalies.long_name = cube.long_name
    anomalies.units = cube.units
    calendar = anomalies.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    anomalies.coord('time').convert_units(common_time_unit)
    for coord_key in ['time', 'latitude', 'longitude']:
        if anomalies.coord(coord_key).points.size > 1:
            anomalies.coord(coord_key).bounds = None
            anomalies.coord(coord_key).guess_bounds()
            anomalies.coord(coord_key).bounds = np.round(anomalies.coord(coord_key).bounds, 3)
            anomalies.coord(coord_key).points = np.round(anomalies.coord(coord_key).points, 3)
    return xr.DataArray.from_iris(anomalies).drop_vars('month_day_str')


def rolling_mean_std_for_standardisation(cube, detrend=False, mean_rolling_window=7, std_rolling_window=7):
    """
    Compute the climatological mean and standard deviation that are used to calculate 
    the standardised anomalies by daily_anomalies_standardised_rolling().

    Parameters
    ----------
    cube: iris.cube.Cube
        Cube containing data from which standardised anomalies are computed. Also accepts xarray DataArray.
    detrend (kwarg): bool, default True
        Apply linear detrending in time before computing anomalies
    mean_rolling_window (kwarg): int, default 7
        Width of rolling window for which to compute climatological mean for anomalies.
        Window is centred on date (e.g. on 15th June, a window of 7 will use data from 12th June to 18th June inclusive
                                   to compute the climatological mean).
    std_rolling_window (kwarg): int, default 7
        Width of rolling window for which to compute climatological standard deviation for standardising anomalies.
        Window is centred on date (e.g. on 15th June, a window of 7 will use data from 12th June to 18th June inclusive
                                   to compute the climatological standard deviation).
    Returns
    -------
    means: xarray.DataArray
        Array of climatological means that would be used to compute standardised anomalies.
    stdevs: xarray.DataArray
        Array of climatological means that would be used to compute standardised anomalies.
    """
    if isinstance(cube, xr.core.dataset.Dataset) or isinstance(cube, xr.core.dataarray.DataArray):
        cube = cube.to_iris()
    if detrend:
        cube = detrend_cube(cube)
    # Compute climatological means and standard deviations using appropriate rolling windows
    dxr = xr.DataArray.from_iris(cube)
    month_day_str = xr.DataArray(dxr.indexes['time'].strftime('%m-%d'), coords=dxr.coords['time'].coords, name='month_day_str')
    rolling_dxr_mean = dxr.rolling(min_periods=1, center=True, time=mean_rolling_window).construct("window_mean")
    rolling_dxr_std = dxr.rolling(min_periods=1, center=True, time=std_rolling_window).construct("window_std")
    grouped_mean_rolling = rolling_dxr_mean.groupby(month_day_str)
    grouped_std_rolling = rolling_dxr_std.groupby(month_day_str)
    climatology_mean = grouped_mean_rolling.mean(["time", "window_mean"])
    climatology_std = grouped_std_rolling.std(["time", "window_std"])
    mean_xr = xr.apply_ufunc(
        lambda x, m: (x/x) * m,
        dxr.groupby(month_day_str),
        climatology_mean, dask='allowed'
    )
    std_xr = xr.apply_ufunc(
        lambda x, s: (x/x) * s,
        dxr.groupby(month_day_str),
        climatology_std, dask='allowed'
    )
    # Fix coordinate names in xarray to prevent naming mismatches
    coord_names = std_xr.coords._names
    if 'lat' in coord_names:
        mean_xr = mean_xr.rename({'lat': 'latitude'})
        std_xr = std_xr.rename({'lat': 'latitude'})
    if 'lon' in coord_names:
        mean_xr = mean_xr.rename({'lon': 'longitude'})
        std_xr = std_xr.rename({'lon': 'longitude'})
    # Ensure final cubes of climatological means and standard deviations have consistent metadata
    means = mean_xr.to_iris()
    stdevs = std_xr.to_iris()
    means.standard_name = cube.standard_name
    stdevs.standard_name = cube.standard_name
    means.long_name = cube.long_name
    stdevs.long_name = cube.long_name
    means.units = cube.units
    stdevs.units = cube.units
    calendar = stdevs.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    means.coord('time').convert_units(common_time_unit)
    stdevs.coord('time').convert_units(common_time_unit)
    for coord_key in ['time', 'latitude', 'longitude']:
        if stdevs.coord(coord_key).points.size > 1:
            stdevs.coord(coord_key).bounds = None
            means.coord(coord_key).bounds = None
            stdevs.coord(coord_key).guess_bounds()
            means.coord(coord_key).guess_bounds()
            stdevs.coord(coord_key).bounds = np.round(stdevs.coord(coord_key).bounds, 3)
            means.coord(coord_key).bounds = np.round(means.coord(coord_key).bounds, 3)
            stdevs.coord(coord_key).points = np.round(stdevs.coord(coord_key).points, 3)
            means.coord(coord_key).points = np.round(means.coord(coord_key).points, 3)
    return xr.DataArray.from_iris(means).drop_vars('month_day_str'), xr.DataArray.from_iris(stdevs).drop_vars('month_day_str')
