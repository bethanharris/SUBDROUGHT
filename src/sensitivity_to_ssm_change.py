import numpy as np
import numpy.ma as ma
import xarray as xr
import os
import calendar
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, RawArray


# config section
area_deg = 2.5
save_dir = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/sensitivity/T2m_sensitivity'
sensitivity_variable_std_anom_dir = "/prj/nceo/bethar/SUBDROUGHT/T2m_standardised_anomalies/"
sensitivity_variable_name = 't2m'
variable_increases_during_drought = True

# load drought days, SSM & sensitivity variable standardised anomalies
drought_events = xr.open_dataset("/prj/nceo/bethar/SUBDROUGHT/HESS_paper/subseasonal_drought_development_events_mask_frozen.nc")
ssm_std_anom_dir = '/prj/nceo/bethar/SUBDROUGHT/soil_moisture_standardised_anomalies/'
ssm_std_anom = xr.open_mfdataset(f'{ssm_std_anom_dir}/*.nc', chunks={"time": -1, "lat": 40, "lon": 40}, parallel=True)
sensitivity_variable_std_anom = xr.open_mfdataset(f'{sensitivity_variable_std_anom_dir}/*.nc', chunks={"time": -1, "lat": 40, "lon": 40}, parallel=True)
# make sure all datasets are for common time span
earliest_common_date = max(max(ssm_std_anom.time.data[0], drought_events.time.data[0]), sensitivity_variable_std_anom.time.data[0])
latest_common_date = min(min(ssm_std_anom.time.data[-1], drought_events.time.data[-1]), sensitivity_variable_std_anom.time.data[-1])
drought_events = drought_events.sel(time=slice(earliest_common_date, latest_common_date))
ssm_std_anom = ssm_std_anom.sel(time=slice(earliest_common_date, latest_common_date))
sensitivity_variable_std_anom = sensitivity_variable_std_anom.sel(time=slice(earliest_common_date, latest_common_date))

init_dict = {}
lat_chunk_size = 80
lon_chunk_size = 360


def initialise_save_files(save_dir, area_deg):
    os.system(f'mkdir -p {save_dir}')
    sensitivity_filename = f'{save_dir}/sensitivity_{area_deg}deg.npy'
    n_filename = f'{save_dir}/n_{area_deg}deg.npy'
    save_array_lats = int(np.ceil(140./float(area_deg)))
    save_array_lons = int(np.ceil(360./float(area_deg)))
    save_array = np.ones((save_array_lats, save_array_lons)) * np.nan
    np.save(n_filename, save_array)
    np.save(sensitivity_filename, save_array)


def make_shared_array(data_array, dtype=np.float64):
    data_shared = RawArray('d', data_array.size)
    data_shared_np = np.frombuffer(data_shared, dtype=dtype).reshape(data_array.shape)
    np.copyto(data_shared_np, data_array)
    return data_shared, data_array.shape


def init_worker(drought_events_data, ssm_std_anom_data, sensitivity_variable_std_anom_data, gridbox_shape, px):
    init_dict['drought_event_data'] = drought_events_data
    init_dict['ssm_std_anom_data'] = ssm_std_anom_data
    init_dict['sensitivity_variable_std_anom_data'] = sensitivity_variable_std_anom_data
    init_dict['gridbox_shape'] = gridbox_shape
    init_dict['px'] = px[0]


def time_series_around_date(data_grid, lat_idx, lon_idx, date_idx, days_range=60):
    box_whole_time_series = data_grid[:, lat_idx, lon_idx]
    end_buffer = np.ones(days_range)*np.nan
    data_pad = np.hstack((end_buffer, box_whole_time_series, end_buffer))
    time_series = data_pad[date_idx+days_range-days_range:date_idx+days_range+(days_range+1)]
    return time_series


def composite_events(events, data_to_composite, days_range=120):
    days_around = np.arange(-days_range, days_range+1)
    data_grid = data_to_composite
    events_bool_array = events.astype(bool)
    event_indices = np.where(events_bool_array)
    events = [((event_indices[1][i], event_indices[2][i]), event_indices[0][i]) for i in range(event_indices[0].size)]
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
                additional_valid_day = np.logical_and(~np.isnan(event_series), ~np.isnan(composite))
                first_valid_day = np.logical_and(~np.isnan(event_series), np.isnan(composite))
                valid_days = np.logical_or(additional_valid_day, first_valid_day)
                n[valid_days] += 1
                composite[additional_valid_day] = composite[additional_valid_day] + (event_series[additional_valid_day] - composite[additional_valid_day])/n[additional_valid_day]
                composite[first_valid_day] = event_series[first_valid_day]
    return days_around, composite, n
 

def compute_coarse_gridbox(coords):
    i, j = coords
    drought_event_data = np.frombuffer(init_dict['drought_event_data']).reshape(init_dict['gridbox_shape'])
    ssm_std_anom_data = np.frombuffer(init_dict['ssm_std_anom_data']).reshape(init_dict['gridbox_shape'])
    sensitivity_variable_std_anom_data = np.frombuffer(init_dict['sensitivity_variable_std_anom_data']).reshape(init_dict['gridbox_shape'])
    droughts_coarse_gridbox = drought_event_data[:, i*px:(i+1)*px, j*px:(j+1)*px]
    ssm_coarse_gridbox = ssm_std_anom_data[:, i*px:(i+1)*px, j*px:(j+1)*px]
    sensitivity_variable_coarse_gridbox = sensitivity_variable_std_anom_data[:, i*px:(i+1)*px, j*px:(j+1)*px]
    _, ssm_composite, ssm_n = composite_events(droughts_coarse_gridbox, ssm_coarse_gridbox, days_range=60)
    _, sensitivity_variable_composite, sensitivity_variable_n = composite_events(droughts_coarse_gridbox, sensitivity_variable_coarse_gridbox, days_range=60)
    enough_obs_before = (np.nansum(ssm_n[0:31])>50) and (np.nansum(sensitivity_variable_n[0:31])>50) #require both vars to have 50 obs from day -60 to -30
    enough_obs_after = (np.nanmax(ssm_n[60:]) > 20) and (np.nanmax(sensitivity_variable_n[60:]) > 20) #require both to have some day after onset with 20 obs in composite
    if (enough_obs_before and enough_obs_after):
        ssm_composite[60:][ssm_n[60:]<20] = np.nan #mask days after onset with not enough obs in composite
        sensitivity_variable_composite[60:][sensitivity_variable_n[60:]<20] = np.nan
        ssm_before = np.nanmean(ssm_composite[0:31]) #baseline anomaly is mean from day -60 to -30
        sensitivity_variable_before = np.nanmean(sensitivity_variable_composite[0:31])
        ssm_after = np.nanmin(ssm_composite[60:]) # value for "after" is minimum after onset (has to be based on >=20 obs)
        if variable_increases_during_drought:
            sensitivity_variable_after = np.nanmax(sensitivity_variable_composite[60:])
            sensitivity_variable_peak_idx = np.nanargmax(sensitivity_variable_composite[60:])
        else:
            sensitivity_variable_after = np.nanmin(sensitivity_variable_composite[60:])
            sensitivity_variable_peak_idx = np.nanargmin(sensitivity_variable_composite[60:])
        n = sensitivity_variable_n[60:][sensitivity_variable_peak_idx]
        sensitivity = (sensitivity_variable_after - sensitivity_variable_before)/(ssm_after - ssm_before)
    else:
        sensitivity = np.nan
        n = np.nan
    return sensitivity, n


def save_multiprocessing_output(lat_coarse_grid_idx, lon_coarse_grid_idx, lat_coarse_gridboxes, lon_coarse_gridboxes, 
                                results, save_dir, area_deg):
    results_stack = np.dstack([results])
    sensitivities = results_stack[:,0].reshape(lat_coarse_gridboxes, lon_coarse_gridboxes, order='F')
    n = results_stack[:,1].reshape(lat_coarse_gridboxes, lon_coarse_gridboxes, order='F')

    sensitivity_filename = f'{save_dir}/sensitivity_{area_deg}deg.npy'
    n_filename = f'{save_dir}/n_{area_deg}deg.npy'
    try:
        sensitivity_array = np.load(sensitivity_filename)
        n_array = np.load(n_filename)
    except:
        initialise_save_files(save_dir, area_deg)
        sensitivity_array = np.load(sensitivity_filename)
        n_array = np.load(n_filename)

    lat_start = int(lat_coarse_grid_idx * lat_chunk_size / (float(area_deg)/0.25))
    lon_start = int(lon_coarse_grid_idx * lon_chunk_size / (float(area_deg)/0.25))
    lat_end = int((lat_coarse_grid_idx+1) * lat_chunk_size / (float(area_deg)/0.25))
    lon_end = int((lon_coarse_grid_idx+1) * lon_chunk_size / (float(area_deg)/0.25))
    sensitivity_array[lat_start:lat_end, lon_start:lon_end] = sensitivities
    n_array[lat_start:lat_end, lon_start:lon_end] = n

    np.save(sensitivity_filename, sensitivity_array)
    np.save(n_filename, n_array)


if __name__ == '__main__':
    global_lats = drought_events.latitude.size
    global_lons = drought_events.longitude.size
    number_lat_chunks = int(np.ceil(float(global_lats)/float(lat_chunk_size)))
    number_lon_chunks = int(np.ceil(float(global_lons)/float(lon_chunk_size)))

    for lat_coarse_grid_idx in tqdm(range(number_lat_chunks)):
        for lon_coarse_grid_idx in range(number_lon_chunks):
            lat_slice_from_global = slice(lat_coarse_grid_idx*lat_chunk_size, (lat_coarse_grid_idx+1)*lat_chunk_size)
            lon_slice_from_global = slice(lon_coarse_grid_idx*lon_chunk_size, (lon_coarse_grid_idx+1)*lon_chunk_size)

            drought_events_tile = drought_events.sm[:, lat_slice_from_global, lon_slice_from_global]
            ssm_std_anoms_tile = ssm_std_anom.sm[:, lat_slice_from_global, lon_slice_from_global]
            sensitivity_variable_std_anoms_tile = sensitivity_variable_std_anom[sensitivity_variable_name][:, lat_slice_from_global, lon_slice_from_global]

            drought_events_data = drought_events_tile.data
            ssm_std_anom_data = ssm_std_anoms_tile.data.compute()
            sensitivity_variable_std_anom_data = sensitivity_variable_std_anoms_tile.data.compute()

            lat_coarse_gridboxes = int(np.ceil(drought_events_tile.latitude.size * 0.25/float(area_deg)))
            lon_coarse_gridboxes = int(np.ceil(drought_events_tile.longitude.size * 0.25/float(area_deg)))
            pixels_in_coarse_gridbox = area_deg * 4.
            px = int(pixels_in_coarse_gridbox)
            if np.abs(pixels_in_coarse_gridbox - px) > 1e-6:
                raise ValueError('Terrible choice of coarse box size')

            drought_event_data_shared, _ = make_shared_array(drought_events_data.astype(np.float64))
            ssm_std_anom_data_shared, _ = make_shared_array(ssm_std_anom_data.astype(np.float64))
            sensitivity_variable_std_anom_data_shared, gridbox_shape = make_shared_array(sensitivity_variable_std_anom_data.astype(np.float64))
            px_shared, _ = make_shared_array(np.array([px]).astype(np.float64))
            
            shared_data = (drought_event_data_shared, ssm_std_anom_data_shared, sensitivity_variable_std_anom_data_shared, 
                           gridbox_shape, px_shared)

            i_list = range(lat_coarse_gridboxes)
            j_list = range(lon_coarse_gridboxes)
            I, J = np.meshgrid(i_list, j_list)
            coords = zip(I.ravel(), J.ravel())

            with Pool(processes=8, initializer=init_worker, initargs=shared_data) as pool:
                results = pool.map(compute_coarse_gridbox, coords)
   
            save_multiprocessing_output(lat_coarse_grid_idx, lon_coarse_grid_idx, 
                                        lat_coarse_gridboxes, lon_coarse_gridboxes,
                                        results, save_dir, area_deg)



