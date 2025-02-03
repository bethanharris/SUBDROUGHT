import xarray as xr
import numpy as np
import os
import pandas as pd
import dask.array
from dask.diagnostics import ProgressBar


save_dir = f'/prj/nceo/bethar/SUBDROUGHT/HESS_paper/max_shf_quartile_maps/'
os.system(f'mkdir -p {save_dir}')


combined_land_covers = {'all_cropland_rainfed': ['cropland_rainfed', 'cropland_rainfed_herbaceous_cover', 'cropland_rainfed_tree_or_shrub_cover'],
                        'all_shrubland': ['shrubland', 'shrubland_evergreen', 'shrubland_deciduous'],
                        'all_sparse_vegetation': ['sparse_vegetation', 'sparse_tree', 'sparse_shrub', 'sparse_herbaceous'],
                        'all_tree_cover': ['tree_broadleaved_evergreen_closed_to_open',' tree_broadleaved_deciduous_closed_to_open',
                                           'tree_broadleaved_deciduous_closed', 'tree_broadleaved_deciduous_open',
                                           'tree_needleleaved_evergreen_closed_to_open', 'tree_needleleaved_evergreen_closed',
                                           'tree_needleleaved_evergreen_open', 'tree_needleleaved_deciduous_closed_to_open',
                                           'tree_needleleaved_deciduous_closed', 'tree_needleleaved_deciduous_open', 'tree_mixed'],
                        'all_broadleaved': ['tree_broadleaved_evergreen_closed_to_open',' tree_broadleaved_deciduous_closed_to_open',
                                           'tree_broadleaved_deciduous_closed', 'tree_broadleaved_deciduous_open'],
                        'all_needleleaved': ['tree_needleleaved_evergreen_closed_to_open',' tree_needleleaved_deciduous_closed_to_open',
                                           'tree_needleleaved_deciduous_closed', 'tree_needleleaved_deciduous_open'],
                        'all_closed_tree_cover': ['tree_broadleaved_deciduous_closed',
                                                  'tree_needleleaved_evergreen_closed',
                                                  'tree_needleleaved_deciduous_closed'],                                          
                        'all_broadleaved_deciduous': ['tree_broadleaved_deciduous_closed_to_open', 'tree_broadleaved_deciduous_closed', 'tree_broadleaved_deciduous_open'],
                        'all_needleleaved_evergreen': ['tree_needleleaved_evergreen_closed_to_open', 'tree_needleleaved_evergreen_closed', 'tree_needleleaved_evergreen_open'],
                        'all_needleleaved_deciduous': ['tree_needleleaved_deciduous_closed_to_open', 'tree_needleleaved_deciduous_closed', 'tree_needleleaved_deciduous_open'],
                        }


def rolling_max_in_subsequent_days(dxr, following_days):
    # compute max over the n days after the timestep (note this includes timestep itself)
    max_in_days_after = dxr.shift(time=-1*(following_days-1)).rolling(min_periods=2, center=False, time=following_days).max()
    return max_in_days_after


def read_land_cover(drought_event_tile, land_cover_mask='all'):
    if land_cover_mask in combined_land_covers.keys():
        land_covers = combined_land_covers[land_cover_mask]
    else:
        land_covers = land_cover_mask
    lc = pd.read_csv("/prj/nceo/bethar/CCI_landcover/land_cover_key.csv")
    lc_dict = lc.set_index('land_cover')['flag'].to_dict()
    land_cover_annual = xr.open_mfdataset('/prj/nceo/bethar/CCI_landcover/0pt25deg/*.nc', chunks={'time': -1, 'lat': 40, 'lon': 40}).lccs_class
    land_cover_annual = land_cover_annual.sel(lat=slice(drought_event_tile.latitude.min()-1e-6, drought_event_tile.latitude.max()+1e-6),
                                              lon=slice(drought_event_tile.longitude.min()-1e-6, drought_event_tile.longitude.max()+1e-6))
    years = land_cover_annual.time.dt.year.data
    mid_year_times = [np.datetime64(f'{year}-07-02', 'ns') for year in years]
    land_cover_annual = land_cover_annual.assign_coords({'time': mid_year_times})
    land_cover_daily = land_cover_annual.interp_like(drought_event_tile, method='nearest', kwargs={"fill_value": "extrapolate"})
    if isinstance(land_covers, list):
        valid_indices = [lc_dict[lc] for lc in land_covers]
        mask = xr.where(land_cover_daily.isin(valid_indices), 1, 0).astype(bool)
    else:
        mask = xr.where(land_cover_daily==lc_dict[land_cover_mask], 1, 0).astype(bool)
    mask = mask.rename({'lat': 'latitude'})
    mask = mask.rename({'lon': 'longitude'})
    mask = mask.assign_coords({'longitude': mask.longitude.data.astype(np.float32), 'latitude': mask.latitude.data.astype(np.float32)})
    return mask


def save_event_quartiles(lc):
    shf_anom = xr.open_mfdataset('/prj/nceo/bethar/ESA_CCI_LST/MODIS_AQUA_LST-T2m/v4.00/*.nc', 
                                chunks={"time": -1, "latitude": 40, "longitude": 40}, parallel=True)
    shf_anom = shf_anom.where(shf_anom['n']>= 625./5.)
    drought_events = xr.open_dataset("/prj/nceo/bethar/SUBDROUGHT/HESS_paper/subseasonal_drought_development_events_mask_frozen.nc")
    earliest_common_date = max(shf_anom.time.data[0], drought_events.time.data[0])
    latest_common_date = min(shf_anom.time.data[-1], drought_events.time.data[-1])
    shf_anom = shf_anom.sel(time=slice(earliest_common_date, latest_common_date)).chunk({'time': -1, 'latitude': 40, 'longitude': 40})
    drought_events = drought_events.sel(time=slice(earliest_common_date, latest_common_date)).chunk({'time': -1, 'latitude': 40, 'longitude': 40})

    max_shf_anom = rolling_max_in_subsequent_days(shf_anom, 20)
    max_shf_anom = max_shf_anom.assign_coords({'latitude': np.round(max_shf_anom.latitude.data, 3), 'longitude': np.round(max_shf_anom.longitude.data, 3)})

    drought_diff = drought_events.astype(int) - drought_events.astype(int).shift(time=1)

    max_shf_anom_after_drought_starts = max_shf_anom['lst-t2m'].where(drought_diff==1)

    save_filename = f'{save_dir}/max_shf_20_quartiles_{lc}_mask20pct.nc'
    file_already_exists = os.path.isfile(save_filename)
    if file_already_exists:
        print(f'Already saved SHF quartiles for {lc}, skipping')
    else:
        if lc == 'all':
            max_shf_anom_after_drought_starts_land_cover = max_shf_anom_after_drought_starts.sm
        else:
            lc_mask = read_land_cover(drought_events, land_cover_mask=lc)
            max_shf_anom_after_drought_starts_land_cover = max_shf_anom_after_drought_starts.sm.where(lc_mask)
        flat = max_shf_anom_after_drought_starts_land_cover.data.ravel()
        flatvalid = flat[~(dask.array.isnan(flat))]
        with ProgressBar():
            all_valid_data = flatvalid.compute()
        percentiles_to_save = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        pc = np.percentile(all_valid_data, percentiles_to_save)
        save_df = pd.DataFrame({'percentile':percentiles_to_save, 'value':pc})
        save_df.to_csv(save_filename, index=False)

        pc_dict = {str(p): v for p, v in zip(percentiles_to_save, pc)}
        quartile_1 = xr.where(max_shf_anom_after_drought_starts_land_cover<=pc_dict['25'], 1, 0)
        quartile_2 = xr.where(dask.array.logical_and(max_shf_anom_after_drought_starts_land_cover>pc_dict['25'], max_shf_anom_after_drought_starts_land_cover<=pc_dict['50']), 2, 0)
        quartile_3 = xr.where(dask.array.logical_and(max_shf_anom_after_drought_starts_land_cover>pc_dict['50'], max_shf_anom_after_drought_starts_land_cover<=pc_dict['75']), 3, 0)
        quartile_4 = xr.where(max_shf_anom_after_drought_starts_land_cover>pc_dict['75'], 4, 0)
        quartile_map = quartile_1 + quartile_2 + quartile_3 + quartile_4
        save_ds = quartile_map.to_dataset(name='shf_max_quartile')
        encoding = {'shf_max_quartile': {'zlib': True, 'contiguous': False, 'chunksizes': (quartile_map.time.size, 40, 40)}}
        with ProgressBar():
            save_ds.to_netcdf(save_filename, encoding=encoding)


if __name__ == '__main__':
    save_event_quartiles('all_cropland_rainfed')