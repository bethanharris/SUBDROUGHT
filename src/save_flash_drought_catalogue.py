import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar
import pandas as pd
from tqdm import tqdm


def save_drought_events_catalogue():
    drought_days = xr.open_dataset('/prj/nceo/bethar/SUBDROUGHT/HESS_paper/subseasonal_drought_development_events_mask_frozen.nc', chunks={"time": -1,'latitude':40,'longitude':40}).sm.chunk({'time': -1})
    sm_anom_pentad_means = xr.open_mfdataset('/prj/nceo/bethar/SUBDROUGHT/HESS_paper/soil_moisture_running_mean_5d_centred/*.nc', parallel=True, chunks={'time':-1,'latitude':40,'longitude':40}).sm

    drought_diff_day_before = drought_days - drought_days.shift(time=1)
    start_of_drought = xr.where(drought_diff_day_before==1, 1, 0).astype(bool)

    dates = np.datetime_as_string(drought_days.time.data, unit='D')

    drought_events = []

    for tile_lat_south in tqdm(np.arange(-60, 11, 70)):
        tile_lat_north = tile_lat_south + 70
        for tile_lon_west in np.arange(-180, 1, 180):
            tile_lon_east = tile_lon_west + 180            
            drought_days_tile = drought_days.sel(latitude=slice(tile_lat_south, tile_lat_north),
                                                 longitude=slice(tile_lon_west, tile_lon_east))
            start_days_tile = start_of_drought.sel(latitude=slice(tile_lat_south, tile_lat_north),
                                                   longitude=slice(tile_lon_west, tile_lon_east))
            pentad_means_tile = sm_anom_pentad_means.sel(latitude=slice(tile_lat_south, tile_lat_north),
                                                         longitude=slice(tile_lon_west, tile_lon_east))
            with ProgressBar():
                drought_days_data = drought_days_tile.data.compute()
                start_days_data = start_days_tile.data.compute()
                pentad_means_data = pentad_means_tile.data.compute()
            latitudes = drought_days_tile.latitude.data
            longitudes = drought_days_tile.longitude.data
            for lat_idx, lat in tqdm(enumerate(latitudes)):
                for lon_idx, lon in enumerate(longitudes):
                    starts_px = start_days_data[:, lat_idx, lon_idx]
                    if starts_px.sum()>0:
                        drought_days_px = drought_days_data[:, lat_idx, lon_idx]
                        pentad_means_px = pentad_means_data[:, lat_idx, lon_idx]
                        for start_day in np.where(starts_px)[0]:
                            start_date = dates[start_day]
                            ends = np.where(~drought_days_px)[0]
                            ends_after = ends[ends>start_day]
                            end_day = ends_after[0]
                            end_date = dates[end_day]
                            drought_length = end_day - start_day
                            maximum_drought_intensity = pentad_means_px[start_day:end_day].min()
                            drought_events.append((lat, lon, start_date, end_date, drought_length, maximum_drought_intensity))

    df = pd.DataFrame(drought_events, columns=['latitude (degrees north)', 'longitude (degrees east)', 'start date', 'end date', 'length', 'maximum intensity'])
    df.to_csv('/prj/nceo/bethar/SUBDROUGHT/HESS_paper/flash_drought_event_catalogue.csv', index=False)


if __name__ == '__main__':
    save_drought_events_catalogue()
