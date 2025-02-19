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


std_anom_directories = {'ssm_pentad_means': "/prj/nceo/bethar/SUBDROUGHT/soil_moisture_running_mean_5d_centred/",
                        'ssm': '/prj/nceo/bethar/SUBDROUGHT/soil_moisture_standardised_anomalies/',
                        'ssm_var': '/prj/nceo/bethar/SUBDROUGHT/soil_moisture_running_variance_11d/',
                        'ssm_GLEAM': '/prj/nceo/bethar/SUBDROUGHT/GLEAM_surface_soil_moisture_standardised_anomalies/',
                        'vod': "/prj/nceo/bethar/SUBDROUGHT/VOD_standardised_anomalies/",
                        'vod_v2': '/prj/nceo/bethar/SUBDROUGHT/VOD_v2_standardised_anomalies/',
                        'vod_v2_SIFyears': '/prj/nceo/bethar/SUBDROUGHT/VOD_v2_SIFyears_standardised_anomalies/',
                        'vod_v1_nofilter': '/prj/nceo/bethar/SUBDROUGHT/VOD_v1_nofilter_standardised_anomalies/',
                        'vpd': "/prj/nceo/bethar/SUBDROUGHT/vpd_standardised_anomalies/",
                        'rzsm': "/prj/nceo/bethar/SUBDROUGHT/root_zone_soil_moisture_standardised_anomalies/",
                        'rzsm_cci': "/prj/nceo/bethar/SUBDROUGHT/ESA_CCI_RZSM_standardised_anomalies/1m/",
                        'rzsm_cci_40cm': "/prj/nceo/bethar/SUBDROUGHT/ESA_CCI_RZSM_standardised_anomalies/40cm/",
                        'rzsm_cci_10cm': "/prj/nceo/bethar/SUBDROUGHT/ESA_CCI_RZSM_standardised_anomalies/10cm/",
                        'rzsm_ERA5_7cm': "/prj/nceo/bethar/SUBDROUGHT/ERA5_soil_moisture_climatological_anomalies/swvl1/",
                        't2m': "/prj/nceo/bethar/SUBDROUGHT/T2m_standardised_anomalies/",
                        'evap': "/prj/nceo/bethar/SUBDROUGHT/GLEAM_v42a_E_standardised_anomalies/",
                        'lst_mw': "/prj/nceo/bethar/SUBDROUGHT/MW-LST_standardised_anomalies/",
                        'lst_aqua': "/prj/nceo/bethar/ESA_CCI_LST/MODIS_Aqua_LST_std_anoms/",
                        'precip': "/prj/nceo/bethar/SUBDROUGHT/precip_standardised_anomalies/",
                        'tdiff_mw_18': "/prj/nceo/bethar/SUBDROUGHT/LST-T2m_MW_18_standardised_anomalies/",
                        'vimd_mean': '/prj/nceo/bethar/SUBDROUGHT/vimd_standardised_anomalies/mean/',
                        'sw_down': '/prj/nceo/bethar/SUBDROUGHT/CERES_sw_down_rad_standardised_anomalies/',
                        'rad': '/prj/nceo/bethar/SUBDROUGHT/CERES_rad_standardised_anomalies/',
                        'wind_speed': '/prj/nceo/bethar/SUBDROUGHT/wind_speed_10m_standardised_anomalies',
                        'SESR_GLEAM': '/prj/nceo/bethar/SUBDROUGHT/GLEAM_SESR/',
                        'SESR_ERA5': '/prj/nceo/bethar/SUBDROUGHT/ERA5_SESR/',
                        'SIF_JJ': '/prj/nceo/bethar/SUBDROUGHT/SIF-GOME2_JJ_standardised_anomalies/',
                        'SIF_PK': '/prj/nceo/bethar/SUBDROUGHT/SIF-GOME2_PK_standardised_anomalies/',
                        'ESR_GLEAM': '/prj/nceo/bethar/GLEAMv38a/ESR/'
                        }


netcdf_variable_names = {'ssm_pentad_means': 'sm',
                         'ssm': 'sm',
                         'ssm_GLEAM': 'SMsurf',
                         'vod': 'vod',
                         'vod_v2': 'VODCA_CXKu',
                         'vod_v2_SIFyears': 'VODCA_CXKu',
                         'vod_v1_nofilter': 'vod',
                         'vpd': 'vpd',
                         'rzsm': 'SMroot',
                         'rzsm_cci': 'rzsm_1m',
                         'rzsm_cci_40cm': 'rzsm_40cm',
                         'rzsm_cci_10cm': 'rzsm_10cm',
                         'rzsm_ERA5_7cm': 'swvl1',
                         't2m': 't2m',
                         'evap': 'E',
                         'lst_mw': 'lst_time_corrected',
                         'lst_aqua': 'lst_aqua',
                         'precip': 'precipitationCal',
                         'tdiff_mw_18': 'lst-t2m_MW_18',
                         'vimd_mean': 'vimd',
                         'sw_down': 'downwelling_surface_sw_rad',
                         'rad': 'net_surface_rad',
                         'wind_speed': 'wind_speed',
                         'SESR_GLEAM': 'ESR',
                         'SESR_ERA5': 'ESR',
                         'SIF_PK': 'SIF',
                         'SIF_JJ': 'SIF',
                         'ESR_GLEAM': 'ESR'
                         }


def time_series_around_date(data_grid, lat_idx, lon_idx, date_idx, days_range=60):
    box_whole_time_series = data_grid[:, lat_idx, lon_idx]
    end_buffer = np.ones(days_range)*np.nan
    data_pad = np.hstack((end_buffer, box_whole_time_series, end_buffer))
    time_series = data_pad[date_idx+days_range-days_range:date_idx+days_range+(days_range+1)]
    return time_series


def composite_events(events_bool_xr, data_xr, days_range=240, land_cover='all'):
    days_around = np.arange(-days_range, days_range+1)
    data_grid = data_xr.data.compute()
    events_bool_array = events_bool_xr.data
    if isinstance(events_bool_array, dask.array.Array):
        events_bool_array = events_bool_array.compute()
    event_indices = np.where(events_bool_array)
    events = [((event_indices[1][i], event_indices[2][i]), event_indices[0][i]) for i in range(event_indices[0].size)]
    if land_cover != 'all':
        lc = read_land_cover(events_bool_xr, land_cover_mask=land_cover).data.compute()
        events = [e for e in events if lc[e[1], e[0][0], e[0][1]]]
    if len(events) == 0:
        composite = np.zeros_like(days_around)*np.nan
        n = np.zeros_like(days_around)
    else:
        event = events[0]
        composite = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
        n = (~np.isnan(composite)).astype(float)
        for event in events[1:]:
            event_is_start_of_drought = events_bool_array[event[1]-1, event[0][0], event[0][1]] == 0
            if event_is_start_of_drought:
                event_series = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
                # plt.plot(days_around, event_series)
                additional_valid_day = np.logical_and(~np.isnan(event_series), ~np.isnan(composite))
                first_valid_day = np.logical_and(~np.isnan(event_series), np.isnan(composite))
                valid_days = np.logical_or(additional_valid_day, first_valid_day)
                n[valid_days] += 1
                composite[additional_valid_day] = composite[additional_valid_day] + (event_series[additional_valid_day] - composite[additional_valid_day])/n[additional_valid_day]
                composite[first_valid_day] = event_series[first_valid_day]
    return days_around, composite, n


def composite_events_spread(events_bool_xr, data_xr, days_range=240, land_cover='all'):
    days_around = np.arange(-days_range, days_range+1)
    data_grid = data_xr.data.compute()
    events_bool_array = events_bool_xr.data
    event_indices = np.where(events_bool_array)
    events = [((event_indices[1][i], event_indices[2][i]), event_indices[0][i]) for i in range(event_indices[0].size)]
    if land_cover != 'all':
        lc = read_land_cover(events_bool_xr, land_cover_mask=land_cover)
        events = [e for e in events if lc[e[1], e[0][0], e[0][1]]]
    if len(events) == 0:
        composite = np.zeros_like(days_around)*np.nan
        composite_std = np.ones_like(days_around) * np.nan
        n = np.zeros_like(days_around)
    else:
        event = events[0]
        composite = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
        n = (~np.isnan(composite)).astype(float)
        squaresum = np.ones_like(n, dtype=np.float64) * np.nan
        squaresum[~np.isnan(composite)] = 0.
        for event in events[1:]:
            event_is_start_of_drought = events_bool_array[event[1]-1, event[0][0], event[0][1]] == 0
            if event_is_start_of_drought:
                event_series = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
                additional_valid_day = np.logical_and(~np.isnan(event_series), ~np.isnan(composite))
                first_valid_day = np.logical_and(~np.isnan(event_series), np.isnan(composite))
                valid_days = np.logical_or(additional_valid_day, first_valid_day)
                n[valid_days] += 1
                old_mean = np.copy(composite)
                composite[additional_valid_day] = composite[additional_valid_day] + (event_series[additional_valid_day] - composite[additional_valid_day])/n[additional_valid_day]
                composite[first_valid_day] = event_series[first_valid_day]
                squaresum[first_valid_day] = 0.
                squaresum[additional_valid_day] = squaresum[additional_valid_day] + ((event_series[additional_valid_day] - old_mean[additional_valid_day]) * (event_series[additional_valid_day] - composite[additional_valid_day]))
        composite_std = np.sqrt(squaresum/n)
    return days_around, composite, composite_std, n



def save_tiles(drought_events, data, number_lat_tiles, number_lon_tiles, tile_scratch_dir, land_cover='all'):
    # Fix coordinate names in xarray to prevent naming mismatches
    coord_names = data.coords._names
    if 'lat' in coord_names:
        data = data.rename({'lat': 'latitude'})
    if 'lon' in coord_names:
        data = data.rename({'lon': 'longitude'})
    lat_tile_slices, lon_tile_slices = get_tile_indices(data.latitude.size, data.longitude.size, 
                                                        number_lat_tiles, number_lon_tiles)
    os.system(f'mkdir -p {tile_scratch_dir}')
    os.system(f'rm {tile_scratch_dir}/*.csv')
    for i in tqdm(range(number_lat_tiles), desc='Processing tiles'):
        for j in range(number_lon_tiles):
            tile_id = i*number_lon_tiles + j
            drought_event_tile = drought_events.isel(latitude=lat_tile_slices[i], longitude=lon_tile_slices[j])
            data_tile = data.isel(latitude=lat_tile_slices[i], longitude=lon_tile_slices[j])
            days_since_drought, tile_composite, tile_n = composite_events(drought_event_tile, data_tile, land_cover=land_cover)
            df = pd.DataFrame({"days_since_drought_start": days_since_drought, "composite_mean": tile_composite, "composite_n": tile_n})
            df.to_csv(f"{tile_scratch_dir}/tile-output-{str(tile_id).zfill(3)}.csv", index=False)


def gather_tiles(tile_scratch_dir, final_save_name, cleanup=False):
    csv_tiles = os.listdir(tile_scratch_dir)
    for i, tile_filename in enumerate(csv_tiles):
        tile_data = pd.read_csv(f'{tile_scratch_dir}/{tile_filename}')
        if i==0:
            days_since_drought_start = tile_data['days_since_drought_start']
            compiled_composite = tile_data['composite_mean']
            compiled_n = tile_data['composite_n']
        else:
            new_composite = tile_data['composite_mean']
            new_n = tile_data['composite_n']
            compiled_composite = (compiled_composite.fillna(0)*compiled_n + new_composite.fillna(0)*new_n)/(compiled_n+new_n)
            compiled_n = compiled_n + new_n
    df = pd.DataFrame({"days_since_drought_start": days_since_drought_start, "composite_mean": compiled_composite, "composite_n": compiled_n})
    save_dir = '/'.join(final_save_name.split('/')[:-1])
    os.system(f'mkdir -p {save_dir}')
    df.to_csv(final_save_name, index=False)
    if cleanup:
        os.system(f'rm {tile_scratch_dir}/*.csv')


def composite_by_tiles(drought_events, data_xr, final_save_name, days_around=60, land_cover='all'):
    tile_scratch_dir = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/scratch/dask_tiling_workspace/'
    save_tiles(drought_events, data_xr, 7, 18, tile_scratch_dir, land_cover=land_cover)
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


def save_global_composite(variable_name, std_anom_dir, final_save_name, land_cover='all'):
    anom = xr.open_mfdataset(f'{std_anom_dir}/*.nc', 
                        chunks={"time": -1, "lat": 40, "lon": 40}, parallel=True)
    try:
        anom = anom.sel(lat=slice(-60, 80))
    except:
        anom = anom.sel(latitude=slice(-60, 80))
    drought_events = xr.open_dataset('/prj/nceo/bethar/SUBDROUGHT/HESS_paper/subseasonal_drought_development_events_mask_frozen.nc')
    earliest_common_date = max(anom.time.data[0], drought_events.time.data[0])
    latest_common_date = min(anom.time.data[-1], drought_events.time.data[-1])
    anom = anom.sel(time=slice(earliest_common_date, latest_common_date))
    drought_events = drought_events.sel(time=slice(earliest_common_date, latest_common_date))
    composite_by_tiles(drought_events.sm, anom[variable_name], final_save_name, land_cover=land_cover)


def save_single_composite_for_land_cover(variable_abbrev, land_cover='all'):
    variable_name = netcdf_variable_names[variable_abbrev]
    std_anom_dir = std_anom_directories[variable_abbrev]
    final_save_name = f'/prj/nceo/bethar/SUBDROUGHT/HESS_paper/ESA_CCI_land_cover_composites/all_seasons/{variable_abbrev}_composite_{land_cover}.csv'
    file_already_exists = os.path.isfile(final_save_name)
    if file_already_exists:
        print(f'Composite already saved for {variable_abbrev} in {land_cover}. Skipping.')
    else:
        if land_cover in combined_land_covers.keys():
            land_covers = combined_land_covers[land_cover]
        else:
            land_covers = land_cover
        save_global_composite(variable_name, std_anom_dir, final_save_name, land_cover=land_covers)


def save_all_composites(land_cover='all'):
    for variable_abbrev in ['ssm_pentad_means', 'rad', 'sw_down', 'evap', 'tdiff_mw_18', 'precip', 'wind_speed']:
        save_single_composite_for_land_cover(variable_abbrev, land_cover=land_cover)


if __name__ == '__main__':
    top_4_lcs = ['all_cropland_rainfed', 'all_shrubland', 'grassland', 'all_broadleaved_deciduous']
    for lc in top_4_lcs:
        save_all_composites(land_cover=lc)
