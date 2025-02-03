import xarray as xr
import numpy as np
from tqdm import tqdm
import dask.array
from dask.diagnostics import ProgressBar
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.feature as feat
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


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

    final_drought_xr = final_drought_xr.chunk({'time': 366, 'latitude': 40, 'longitude': 40})
    encoding_settings = dict(contiguous=False, chunksizes=(366, 40, 40))
    encoding = {'sm': encoding_settings}
    final_drought_xr.to_netcdf('/prj/nceo/bethar/SUBDROUGHT/HESS_paper/subseasonal_drought_development_events_mask_frozen.nc', encoding=encoding)


def plot_n_map():
    drought_days = xr.open_dataset('/prj/nceo/bethar/SUBDROUGHT/HESS_paper/subseasonal_drought_development_events_mask_frozen.nc', chunks={"time": 365}).sm
    drought_diff_day_before = drought_days - drought_days.shift(time=1)
    start_of_drought = xr.where(drought_diff_day_before==1, 1, 0).astype(bool)
    number_droughts = start_of_drought.sum('time')
    with ProgressBar():
        n = number_droughts.data.compute()
    n = n.astype(float)
    n[n==0.] = np.nan

    lons = np.arange(-180, 180, 0.25) + 0.5*0.25
    lats = np.arange(-60, 80, 0.25) + 0.5*0.25
    lon_bounds = np.hstack((lons - 0.5*0.25, np.array([lons[-1]+0.5*0.25])))
    lat_bounds = np.hstack((lats - 0.5*0.25, np.array([lats[-1]+0.5*0.25])))
    cmap = cm.Oranges
    cmap.set_bad('white')
    cmap.set_under('#cccccc')
    # cmap, norm = binned_cmap(np.arange(10), 'Oranges', fix_colours=[(0, 'w')], extend='max')
    sm_mask = xr.open_dataset("/prj/nceo/bethar/VODCA_global/ESA-CCI-SOILMOISTURE-LAND_AND_RAINFOREST_MASK-fv04.2.nc")
    sm_mask = sm_mask.isel(lat=slice(None, None, -1))
    sm_mask = sm_mask.sel(lat=slice(-60, 80))
    no_sm = np.logical_or(~(sm_mask.land.data), sm_mask.rainforest.data)
    n_masked = np.copy(n)
    n_masked[no_sm] = -999

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(6, 4.5)) 
    ax = plt.axes(projection=projection)
    p = plt.pcolormesh(lon_bounds, lat_bounds, n_masked, cmap=cmap, vmin=0, vmax=10, transform=ccrs.PlateCarree())
    cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0-0.12, ax.get_position().width, 0.03])
    cbar = fig.colorbar(p, orientation='horizontal', cax=cax, aspect=40, pad=0.12, extend='max')
    cbar.set_ticks(np.arange(0, 11, 2))
    endash = u'\u2013'
    cbar.ax.set_xlabel(f'number of flash drought events 2000{endash}2020', fontsize=15)
    cbar.ax.tick_params(labelsize=16)
    ax.coastlines(color='black', linewidth=1)
    ax.set_xticks(np.arange(-180, 181, 90), crs=projection)
    ax.set_yticks(np.arange(-60, 61, 60), crs=projection)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=14)
    ax.tick_params(axis='x', pad=5)
    plt.savefig('../figures/number_events_esa_cci_sm_mask.png', dpi=800, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # save_flash_droughts()
    plot_n_map()