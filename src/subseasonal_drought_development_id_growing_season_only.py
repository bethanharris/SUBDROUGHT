import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import dask.array
from dask.diagnostics import ProgressBar


def save_flash_droughts():
    
    # load SSM and T2m lagged means/mins needed for flash drought identification
    # these are all saved by save_rolling_pentad_means_mins.py

    rolling_pentad_mean_dir = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/soil_moisture_running_mean_5d_centred/'
    rolling_30d_mean_dir = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/soil_moisture_running_mean_30d_before'
    rolling_minimum_after_dir = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/soil_moisture_running_pentad_min_20d_after/'
    rolling_t2m_min_dir = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/t2m_running_pentad_min_61d/'

    mean_in_pentad = xr.open_mfdataset(f'{rolling_pentad_mean_dir}/*.nc', 
                                    chunks={"time": -1, "latitude": 40, "longitude": 40}, parallel=True)
    min_in_20d_after = xr.open_mfdataset(f'{rolling_minimum_after_dir}/*.nc', 
                                        chunks={"time": -1, "latitude": 40, "longitude": 40}, parallel=True)
    mean_in_30d_before = xr.open_mfdataset(f'{rolling_30d_mean_dir}/*.nc', 
                                        chunks={"time": -1, "latitude": 40, "longitude": 40}, parallel=True)
    min_t2m_pentad = xr.open_mfdataset(f'{rolling_t2m_min_dir}/*.nc', 
                                    chunks={"time": -1, "latitude": 40, "longitude": 40}, parallel=True)

    # identify peak growing season based on NDVI climatology
    ndvi = xr.open_mfdataset('/prj/nceo/bethar/MODIS-NDVI-16day/v061/modis_terra_*').NDVI
    ndvi = ndvi.sel(lat=slice(-60, 80), time=slice('2000-01-01', '2020-12-31'))
    annual_cycle_ndvi = ndvi.groupby('time.month').mean('time')
    number_ndvi_obs = ndvi.count('time')
    max_ndvi_month_filled = annual_cycle_ndvi.fillna(-99).argmax(dim='month')
    max_ndvi_month_masked = max_ndvi_month_filled.where(number_ndvi_obs>0.)
    max_ndvi_month = (max_ndvi_month_masked + 1).data.compute()
    months = mean_in_pentad.sm.time.dt.month.data

    # set onset conditions for flash drought identification
    drought_event_starts = xr.where(dask.array.logical_and(mean_in_30d_before>-0.25, min_in_20d_after<-1.0), 1, 0)
    drought_reached = xr.where(mean_in_pentad<-1.0, 1, 0)
    # identify where frozen ground is likely
    frozen = xr.where(min_t2m_pentad<278.15, 1, 0)

    with ProgressBar():
        drought_start_data = drought_event_starts.sm.astype(bool).data.compute()
        drought_reached_data = drought_reached.sm.astype(bool).data.compute()
        frozen_data = frozen.t2m.astype(bool).data.compute()

    drought_over = ~drought_reached_data # drought over when no longer below -1stdev

    number_lats = mean_in_pentad.latitude.size
    number_lons = mean_in_pentad.longitude.size

    final_drought_dates = np.zeros_like(drought_start_data)

    # loop over pixels to ID flash droughts
    for lat_idx in tqdm(range(number_lats)):
        for lon_idx in range(number_lons):
            max_ndvi_month_px = max_ndvi_month[lat_idx, lon_idx]
            drought_starts_px = np.where(drought_start_data[:, lat_idx, lon_idx])[0]
            drought_reached_px = np.where(drought_reached_data[:, lat_idx, lon_idx])[0]
            drought_over_px = np.where(drought_over[:, lat_idx, lon_idx])[0]
            frozen_px = np.where(frozen_data[:, lat_idx, lon_idx])[0]
            for drought_start in drought_starts_px:
                if not np.isin(drought_start, frozen_px): # frozen ground not likely
                    drought_ends_after = drought_reached_px[drought_reached_px>drought_start]
                    if drought_ends_after.size > 0: #don't count the drought if we never see the end of it
                        # find first drought timestamp in FD
                        next_drought_reached = np.min(drought_reached_px[drought_reached_px>drought_start])
                        max_month_offset = np.abs(months[next_drought_reached] - max_ndvi_month_px)
                        growing_season = np.logical_or(max_month_offset <= 1, max_month_offset == 11) # account for Dec/Jan being adjacent
                        if growing_season:
                            drought_over_after = drought_over_px[drought_over_px>next_drought_reached]
                            if drought_over_after.size > 0.:
                                next_drought_over = np.min(drought_over_px[drought_over_px>next_drought_reached])
                                if next_drought_over - next_drought_reached >= 15.: # only save FDs lasting 15 days+
                                    final_drought_dates[next_drought_reached:next_drought_over, lat_idx, lon_idx] = 1

    final_drought_xr = xr.DataArray(final_drought_dates, name='sm',
                                    coords={'time': mean_in_pentad.time.data,
                                            'latitude': mean_in_pentad.latitude.data,
                                            'longitude': mean_in_pentad.longitude.data}, 
                                    dims=["time", "latitude", "longitude"])

    final_drought_xr = final_drought_xr.chunk({'time':366, 'latitude':40, 'longitude':40})
    encoding_settings = dict(contiguous=False, chunksizes=(366, 40, 40))
    encoding = {'sm': encoding_settings}
    final_drought_xr.to_netcdf('/prj/nceo/bethar/SUBDROUGHT/HESS_paper/subseasonal_drought_development_events_mask_frozen_peak_growing_season.nc', encoding=encoding)
