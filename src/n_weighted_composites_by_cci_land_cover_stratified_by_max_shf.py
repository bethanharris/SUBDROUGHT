import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import os.path
import string
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import dask.array
from save_global_standardised_anomalies import get_tile_indices


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

std_anom_directories = {
                        'tdiff_aqua': '/prj/nceo/bethar/ESA_CCI_LST/MODIS_AQUA_LST-T2m/v4.00/',
                        'lst_aqua': "/prj/nceo/bethar/ESA_CCI_LST/MODIS_Aqua_LST_std_anoms/"
                        }


netcdf_variable_names = {
                         'tdiff_aqua': 'lst-t2m',
                         'lst_aqua': 'lst_aqua'
                         }


def time_series_around_date(data_grid, lat_idx, lon_idx, date_idx, days_range=60, zero_pad=False, as_int=False):
    box_whole_time_series = data_grid[:, lat_idx, lon_idx]
    if zero_pad:
        end_buffer = np.zeros(days_range)
        if as_int:
            end_buffer = end_buffer.astype(int)
    else:
        end_buffer = np.ones(days_range)*np.nan
    data_pad = np.hstack((end_buffer, box_whole_time_series, end_buffer))
    time_series = data_pad[date_idx+days_range-days_range:date_idx+days_range+(days_range+1)]
    return time_series


def composite_events(events_bool_xr, data_xr, total_obs_xr, days_range=120, land_cover='all', quartile_land_cover='all'):
    days_around = np.arange(-days_range, days_range+1)
    shf_max_quartile_xr = get_quartile_map(events_bool_xr, land_cover=quartile_land_cover)
    # fix it if I didn't make the t2m quartile data for exact years the data spans. events_bool_xr and data_xr should
    # already match
    earliest_common_date = max(shf_max_quartile_xr.time.data[0], events_bool_xr.time.data[0])
    latest_common_date = min(shf_max_quartile_xr.time.data[-1], events_bool_xr.time.data[-1])
    events_bool_xr = events_bool_xr.sel(time=slice(earliest_common_date, latest_common_date))
    data_xr = data_xr.sel(time=slice(earliest_common_date, latest_common_date))
    total_obs_xr = total_obs_xr.sel(time=slice(earliest_common_date, latest_common_date))
    shf_max_quartile_xr = shf_max_quartile_xr.sel(time=slice(earliest_common_date, latest_common_date))
    data_grid = data_xr.data.compute()
    shf_max_quartile = shf_max_quartile_xr.data.compute()
    total_obs_grid = total_obs_xr.data.compute()
    events_bool_array = events_bool_xr.data
    if isinstance(events_bool_array, dask.array.Array):
        events_bool_array = events_bool_array.compute()
    event_indices = np.where(events_bool_array)
    events = [((event_indices[1][i], event_indices[2][i]), event_indices[0][i]) for i in range(event_indices[0].size)]
    if land_cover != 'all':
        lc = read_land_cover(events_bool_xr, land_cover_mask=land_cover).data.compute()
        events = [e for e in events if lc[e[1], e[0][0], e[0][1]]]
    composite = {f'Q{x}': np.zeros_like(days_around)*np.nan for x in range(1,5)}
    n = {f'Q{x}': np.zeros_like(days_around) for x in range(1,5)}
    total_obs = {f'Q{x}': np.zeros_like(days_around) for x in range(1,5)}
    for event in events:
        event_is_start_of_drought = events_bool_array[event[1]-1, event[0][0], event[0][1]] == 0
        q = shf_max_quartile[event[1], event[0][0], event[0][1]]
        event_is_start_of_drought = np.logical_and(event_is_start_of_drought, q>0.)
        if event_is_start_of_drought:
            event_series = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
            total_obs_series = time_series_around_date(total_obs_grid, event[0][0], event[0][1], event[1], days_range=days_range, zero_pad=True, as_int=True)
            # plt.plot(days_around, event_series)
            additional_valid_day = np.logical_and(~np.isnan(event_series), ~np.isnan(composite[f'Q{q}']))
            first_valid_day = np.logical_and(~np.isnan(event_series), np.isnan(composite[f'Q{q}']))
            valid_days = np.logical_or(additional_valid_day, first_valid_day)
            n[f'Q{q}'][valid_days] += 1
            total_obs[f'Q{q}'][valid_days] += total_obs_series[valid_days] # not actually necessary to restrict to valid_days?
            weight_ratio = total_obs_series/total_obs[f'Q{q}']
            composite[f'Q{q}'][additional_valid_day] = composite[f'Q{q}'][additional_valid_day] + weight_ratio[additional_valid_day] * (event_series[additional_valid_day] - composite[f'Q{q}'][additional_valid_day])
            composite[f'Q{q}'][first_valid_day] = event_series[first_valid_day]
    return days_around, composite, n, total_obs


def save_tiles(drought_events, data, total_obs, number_lat_tiles, number_lon_tiles, tile_scratch_dir, land_cover='all', quartile_land_cover='all'):
    # Fix coordinate names in xarray to prevent naming mismatches
    coord_names = data.coords._names
    if 'lat' in coord_names:
        data = data.rename({'lat': 'latitude'})
        total_obs = total_obs.rename({'lat': 'latitude'})
    if 'lon' in coord_names:
        data = data.rename({'lon': 'longitude'})
        total_obs = total_obs.rename({'lon': 'longitude'})
    lat_tile_slices, lon_tile_slices = get_tile_indices(data.latitude.size, data.longitude.size, 
                                                        number_lat_tiles, number_lon_tiles)
    os.system(f'mkdir -p {tile_scratch_dir}')
    os.system(f'rm {tile_scratch_dir}/*.csv')
    for i in tqdm(range(number_lat_tiles), desc='Processing tiles'):
        for j in range(number_lon_tiles):
            tile_id = i*number_lon_tiles + j
            drought_event_tile = drought_events.isel(latitude=lat_tile_slices[i], longitude=lon_tile_slices[j])
            data_tile = data.isel(latitude=lat_tile_slices[i], longitude=lon_tile_slices[j])
            total_obs_tile = total_obs.isel(latitude=lat_tile_slices[i], longitude=lon_tile_slices[j])
            days_since_drought, tile_composite, tile_n, tile_total_obs = composite_events(drought_event_tile, data_tile, total_obs_tile, land_cover=land_cover, quartile_land_cover=quartile_land_cover)
            for q in range(1,5):
                df = pd.DataFrame({"days_since_drought_start": days_since_drought, "composite_mean": tile_composite[f'Q{q}'], "composite_n": tile_n[f'Q{q}'], "composite_total_obs": tile_total_obs[f'Q{q}']})
                df.to_csv(f"{tile_scratch_dir}/tile-output-Q{q}-{str(tile_id).zfill(3)}.csv", index=False)


def gather_tiles(tile_scratch_dir, final_save_name, cleanup=False):
    for q in range(1,5):
        csv_tiles = os.listdir(tile_scratch_dir)
        csv_tiles = [f for f in csv_tiles if f'Q{q}' in f]
        for i, tile_filename in enumerate(csv_tiles):
            tile_data = pd.read_csv(f'{tile_scratch_dir}/{tile_filename}')
            if i==0:
                days_since_drought_start = tile_data['days_since_drought_start']
                compiled_composite = tile_data['composite_mean']
                compiled_total_obs = tile_data['composite_total_obs']
                compiled_n = tile_data['composite_n']
            else:
                new_composite = tile_data['composite_mean']
                new_n = tile_data['composite_n']
                new_total_obs = tile_data['composite_total_obs']
                compiled_composite = (compiled_composite.fillna(0)*compiled_total_obs + new_composite.fillna(0)*new_total_obs)/(compiled_total_obs+new_total_obs)
                compiled_n = compiled_n + new_n
                compiled_total_obs = compiled_total_obs + new_total_obs
        df = pd.DataFrame({"days_since_drought_start": days_since_drought_start, "composite_mean": compiled_composite, "composite_n": compiled_n, "composite_total_obs": compiled_total_obs})
        save_dir = '/'.join(final_save_name.split('/')[:-1])
        q_save_name = final_save_name.split('.csv')[0] + f'_Q{q}' + '.csv'
        os.system(f'mkdir -p {save_dir}')
        df.to_csv(q_save_name, index=False)
    if cleanup:
        os.system(f'rm {tile_scratch_dir}/*.csv')


def composite_by_tiles(drought_events, data_xr, total_obs_xr, final_save_name, land_cover='all', quartile_land_cover='all'):
    tile_scratch_dir = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/scratch/dask_tiling_workspace/'
    save_tiles(drought_events, data_xr, total_obs_xr, 7, 18, tile_scratch_dir, land_cover=land_cover, quartile_land_cover=quartile_land_cover)
    gather_tiles(tile_scratch_dir, final_save_name, cleanup=True)


def read_land_cover(drought_event_tile, land_cover_mask='all'):
    lc = pd.read_csv("/prj/nceo/bethar/CCI_landcover/land_cover_key.csv")
    lc_dict = lc.set_index('land_cover')['flag'].to_dict()
    land_cover_annual = xr.open_mfdataset('/prj/nceo/bethar/CCI_landcover/0pt25deg/*.nc', chunks={'time': -1, 'lat': 40, 'lon': 40}).lccs_class
    land_cover_annual = land_cover_annual.sel(lat=slice(drought_event_tile.latitude.min()-1e-6, drought_event_tile.latitude.max()+1e-6),
                                              lon=slice(drought_event_tile.longitude.min()-1e-6, drought_event_tile.longitude.max()+1e-6))
    years = land_cover_annual.time.dt.year.data
    mid_year_times = [np.datetime64(f'{year}-07-02', 'ns') for year in years]
    land_cover_annual = land_cover_annual.assign_coords({'time': mid_year_times})
    land_cover_daily = land_cover_annual.interp_like(drought_event_tile, method='nearest', kwargs={"fill_value": "extrapolate"})
    if isinstance(land_cover_mask, list):
        valid_indices = [lc_dict[lc] for lc in land_cover_mask]
        mask = xr.where(land_cover_daily.isin(valid_indices), 1, 0).astype(bool)
    else:
        mask = xr.where(land_cover_daily==lc_dict[land_cover_mask], 1, 0).astype(bool)
    return mask


def get_quartile_map(drought_event_tile, land_cover='all'):
    quartiles = xr.open_dataset(f'/prj/nceo/bethar/SUBDROUGHT/HESS_paper/max_shf_quartile_maps/max_shf_20_quartiles_{land_cover}_mask20pct.nc', chunks={'time': -1, 'lat': 40, 'lon': 40}).shf_max_quartile
    quartiles = quartiles.sel(latitude=slice(drought_event_tile.latitude.min()-1e-6, drought_event_tile.latitude.max()+1e-6),
                              longitude=slice(drought_event_tile.longitude.min()-1e-6, drought_event_tile.longitude.max()+1e-6))
    return quartiles


def save_global_composite(variable_name, std_anom_dir, final_save_name, land_cover='all', quartile_land_cover='all'):
    anom = xr.open_mfdataset(f'{std_anom_dir}/*.nc', 
                        chunks={"time": -1, "latitude": 40, "longitude": 40}, parallel=True)
    anom = anom.sel(latitude=slice(-60, 80), longitude=slice(-180, 180))
    anom[variable_name] = anom[variable_name].where(anom['n']>= 625./5.)
    anom['n'] = xr.where(anom['n']>= 625./5., anom['n'], 0)
    drought_events = xr.open_dataset("/prj/nceo/bethar/SUBDROUGHT/HESS_paper/subseasonal_drought_development_events_mask_frozen.nc")
    drought_events = drought_events.sel(latitude=slice(-60, 80), longitude=slice(-180, 180))
    earliest_common_date = max(anom.time.data[0], drought_events.time.data[0])
    latest_common_date = min(anom.time.data[-1], drought_events.time.data[-1])
    anom = anom.sel(time=slice(earliest_common_date, latest_common_date))
    drought_events = drought_events.sel(time=slice(earliest_common_date, latest_common_date))
    composite_by_tiles(drought_events.sm, anom[variable_name], anom['n'], final_save_name, land_cover=land_cover, quartile_land_cover=quartile_land_cover)


def save_single_composite_for_land_cover(variable_abbrev, land_cover='all'):
    variable_name = netcdf_variable_names[variable_abbrev]
    std_anom_dir = std_anom_directories[variable_abbrev]
    final_save_name = f'/prj/nceo/bethar/SUBDROUGHT/HESS_paper/ESA_CCI_land_cover_composites/stratified_by_max_shf/mask20pct/{variable_abbrev}_n-weighted_composite_{land_cover}.csv'
    save_stem = final_save_name.split('.csv')[0]
    quartile_save_names = [f"{save_stem}_Q{q}.csv" for q in range(1, 5)]
    files_already_exist = [os.path.isfile(q) for q in quartile_save_names]
    quartile_land_cover = land_cover
    if all(files_already_exist):
        print(f'Composites already saved for {variable_abbrev} in {land_cover}. Skipping.')
    else:
        if land_cover in combined_land_covers.keys():
            land_covers = combined_land_covers[land_cover]
        else:
            land_covers = land_cover
        save_global_composite(variable_name, std_anom_dir, final_save_name, land_cover=land_covers, quartile_land_cover=quartile_land_cover)


def save_all_composites(land_cover='all'):
    for variable_abbrev in netcdf_variable_names.keys():
        save_single_composite_for_land_cover(variable_abbrev, land_cover=land_cover)


if __name__ == '__main__':
    save_all_composites('all_cropland_rainfed')
