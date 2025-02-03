import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import dask.array
from dask.diagnostics import ProgressBar
import time
from save_global_standardised_anomalies import process_by_tiles


tile_scratch_dir = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/scratch'


def rolling_min_in_subsequent_pentads(dxr, following_pentads):
    following_days = 5*following_pentads
    dxr_pad = dxr.pad(pad_width={'time': following_days}, mode='constant', constant_values=np.nan)
    min_in_pentads_after_pad = dxr_pad.shift(time=-1*(following_days)).rolling(min_periods=1, center=False, time=following_days).min()
    min_in_pentads_after = min_in_pentads_after_pad.isel(time=slice(following_days, -1*following_days))
    return min_in_pentads_after


def rolling_mean_surrounding_days(dxr, window, min_periods=1):
    dxr_pad = dxr.pad(pad_width={'time': window}, mode='constant', constant_values=np.nan)
    mean_in_window_pad = dxr_pad.rolling(min_periods=min_periods, center=True, time=window).mean()
    mean_in_window = mean_in_window_pad.isel(time=slice(window, -1*window))
    return mean_in_window


def rolling_min_surrounding_days(dxr, window, min_periods=1):
    dxr_pad = dxr.pad(pad_width={'time': window}, mode='constant', constant_values=np.nan)
    min_in_window_pad = dxr_pad.rolling(min_periods=min_periods, center=True, time=window).min()
    min_in_window = min_in_window_pad.isel(time=slice(window, -1*window))
    return min_in_window


def running_sm_pentad_mins_after(sm_anom_pentad_means):
    min_subsequent_4_pentads = rolling_min_in_subsequent_pentads(sm_anom_pentad_means, 4).sm
    return min_subsequent_4_pentads


def running_sm_pentad_means_centred(sm_anom):
    mean_around = rolling_mean_surrounding_days(sm_anom, 5, min_periods=2).sm
    return mean_around


def rolling_mean_in_preceding_days(dxr, preceding_days):
    # compute mean over the n days before the timestep (note this includes timestep itself)
    mean_in_days_before = dxr.rolling(min_periods=5, center=False, time=preceding_days).mean()
    return mean_in_days_before


def running_means_before(sm_anom):
    mean_previous_30_days = rolling_mean_in_preceding_days(sm_anom, 30).sm
    return mean_previous_30_days


def running_t2m_pentad_means_centred(t2m_raw):
    mean_around = rolling_mean_surrounding_days(t2m_raw, 5).t2m
    return mean_around


def running_t2m_mins_centred(t2m_raw):
    min_around = rolling_min_surrounding_days(t2m_raw, 61, min_periods=1).t2m
    return min_around


def save_sm_running_pentad_means(number_lat_tiles, number_lon_tiles, save_by_year=False, cleanup=True):
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/soil_moisture_running_mean_5d_centred/running_mean_5d_centred.nc'
    ssm_anom_dir = '/prj/nceo/bethar/SUBDROUGHT/soil_moisture_standardised_anomalies'
    ssm = xr.open_mfdataset(f'{ssm_anom_dir}/*.nc', 
                               chunks={"time": -1, "lat": 40, "lon": 40}, parallel=True)
    ssm = ssm.chunk(chunks={'time': -1, "latitude": 280, "longitude": 240}) # ALWAYS match tile size
    process_by_tiles('sm', running_sm_pentad_means_centred, ssm, number_lat_tiles, number_lon_tiles, 
                     tile_scratch_dir, final_save_name, save_by_year=save_by_year, cleanup=cleanup)
    

def save_sm_preceding_means(number_lat_tiles, number_lon_tiles, save_by_year=False, cleanup=True):
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/soil_moisture_running_mean_30d_before/running_mean_30d_before.nc'
    ssm_anom_dir = '/prj/nceo/bethar/SUBDROUGHT/soil_moisture_standardised_anomalies'
    ssm = xr.open_mfdataset(f'{ssm_anom_dir}/*.nc', 
                               chunks={"time": -1, "latitude": 40, "longitude": 40}, parallel=True)
    ssm = ssm.chunk(chunks={'time': -1, "latitude": 280, "longitude": 240}) # ALWAYS match tile size
    process_by_tiles('sm', running_means_before, ssm, number_lat_tiles, number_lon_tiles, 
                     tile_scratch_dir, final_save_name, save_by_year=save_by_year, cleanup=cleanup)


def save_sm_min_in_pentads_after(number_lat_tiles, number_lon_tiles, save_by_year=False, cleanup=True):
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/soil_moisture_running_pentad_min_20d_after/running_pentad_min_20d_after.nc'
    pentad_mean_dir = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/soil_moisture_running_mean_5d_centred'
    pentad_means = xr.open_mfdataset(f'{pentad_mean_dir}/*.nc', 
                                chunks={"time": -1, "latitude": 40, "longitude": 40}, parallel=True)
    pentad_means = pentad_means.chunk(chunks={'time': -1, "latitude": 280, "longitude": 240})
    process_by_tiles('sm', running_sm_pentad_mins_after, pentad_means, number_lat_tiles, number_lon_tiles, 
                     tile_scratch_dir, final_save_name, save_by_year=save_by_year, cleanup=cleanup)


def save_t2m_running_pentad_means(number_lat_tiles, number_lon_tiles, save_by_year=False, cleanup=True):
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/max_t2m_running_mean_5d/running_mean_5d.nc'
    t2m_dir = '/prj/nceo/bethar/ERA5/2m_temperature/local_time/afternoon/chunked/'
    t2m = xr.open_mfdataset(f'{t2m_dir}/*.nc', 
                               chunks={"time": -1, "latitude": 40, "longitude": 40}, parallel=True)
    t2m = t2m.sel(latitude=slice(-60, 80))
    t2m = t2m.chunk(chunks={'time': -1, "latitude": 280, "longitude": 240})
    process_by_tiles('t2m', running_t2m_pentad_means_centred, t2m, number_lat_tiles, number_lon_tiles, 
                     tile_scratch_dir, final_save_name, save_by_year=save_by_year, cleanup=cleanup)


def save_t2m_pentad_min_in_61d_window(number_lat_tiles, number_lon_tiles, save_by_year=False, cleanup=True):
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/t2m_running_pentad_min_61d/t2m_running_pentad_min_61d.nc'
    pentad_mean_dir = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/max_t2m_running_mean_5d'
    pentad_means = xr.open_mfdataset(f'{pentad_mean_dir}/*.nc', 
                                chunks={"time": -1, "latitude": 40, "longitude": 40}, parallel=True)
    pentad_means = pentad_means.chunk(chunks={'time': -1, "latitude": 280, "longitude": 240})
    process_by_tiles('t2m', running_t2m_mins_centred, pentad_means, number_lat_tiles, number_lon_tiles, 
                     tile_scratch_dir, final_save_name, save_by_year=save_by_year, cleanup=cleanup)


if __name__ == '__main__':
    # save running 5-day mean of standardised soil moisture anomalies. Must have 2 obs in pentad
    save_sm_running_pentad_means(2, 6, save_by_year=True, cleanup=True)
    # save running mean of standardised soil moisture anomaly in the 30 days before timestamp
    save_sm_preceding_means(2, 6, save_by_year=True, cleanup=True)
    # save the minimum running pentad mean in 20 days after timestamp
    save_sm_min_in_pentads_after(2, 6, save_by_year=True, cleanup=True)
    # save the minimum T2m running pentad mean in the 30d either side of the timestamp (for masking frozen pixels)
    save_t2m_running_pentad_means(2, 6, save_by_year=True, cleanup=True)
    save_t2m_pentad_min_in_61d_window(2, 6, save_by_year=True, cleanup=True)
