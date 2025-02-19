import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar
from standardised_anomalies import daily_anomalies_standardised_rolling
import time
import os
from tqdm import tqdm
import dask


SCRATCH_DIR = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/scratch/'


def rechunk_global_files(data_directory, variable_name, min_year=2000, max_year=2020):
    # variable_name should be what the xarray DataArray name is
    for year in tqdm(range(min_year, max_year+1), desc=f'Rechunking {variable_name} files'):
        if os.path.isfile(f'{SCRATCH_DIR}/rechunked_{variable_name}_{year}.nc'):
            print(f'Rechunked {variable_name} already saved for {year}, skipping')
        else:
            year_files = [f for f in os.listdir(data_directory) if (str(year) in f and f.endswith('.nc'))]
            if len(year_files) == 0:
                raise ValueError(f'No files for {year} in directory')
            global_data = xr.open_mfdataset([f'{data_directory}/{yf}' for yf in year_files])
            days_in_year = global_data.time.size
            if variable_name == 'sm':
                global_data = global_data.isel(lat=slice(None, None, -1))
                global_data = global_data.where(global_data[variable_name] < 1e10)
            elif variable_name == 'vod':
                shifted_times = [t - np.timedelta64(12, 'h') for t in global_data.time.data]
                global_data = global_data.assign_coords(time=shifted_times)
            elif variable_name == 'VODCA_CXKu':
                global_data = global_data.isel(lat=slice(None, None, -1))
                global_data[variable_name].attrs['units'] = 1
            elif variable_name.startswith('rzsm_'):
                global_data = global_data.where(global_data[variable_name] < 1e10)
            elif variable_name == 'SMroot':
                global_data = global_data.isel(lat=slice(None, None, -1))
                global_data[variable_name].attrs['standard_name'] = 'mass_content_of_water_in_soil_layer_defined_by_root_depth'
                global_data = global_data.where(global_data[variable_name] < 1e10)
            elif variable_name == 'SMsurf':
                global_data = global_data.isel(lat=slice(None, None, -1))
                global_data[variable_name].attrs['standard_name'] = 'mass_content_of_water_in_soil_layer'
                global_data = global_data.where(global_data[variable_name] < 1e10)
            elif variable_name == 'E':
                global_data = global_data.isel(lat=slice(None, None, -1))
                global_data[variable_name].attrs['standard_name'] = 'water_evapotranspiration_flux'
                global_data = global_data.where(global_data[variable_name] < 1e10)
            # elif variable_name == 'E' or variable_name == 'SMrz':
            #     global_data = global_data.where(global_data[variable_name] < 1e10)
            #     global_data = global_data.rename({'latitude': 'lat'})
            #     global_data = global_data.rename({'longitude': 'lon'})
            elif variable_name == 'net_surface_rad':
                global_data = global_data.where(global_data[variable_name] < 1e10)
            elif variable_name == 'downwelling_surface_sw_rad':
                global_data = global_data.where(global_data[variable_name] < 1e10)
                global_data[variable_name].attrs['standard_name'] = 'surface_downwelling_shortwave_flux_in_air'
            elif variable_name == 'precipitationCal':
                shifted_times = [t - np.timedelta64(12, 'h') for t in global_data.time.data]
                global_data = global_data.assign_coords(time=shifted_times)
                global_data = global_data.where(global_data[variable_name] < 1e30)
                if 'time_bnds' in global_data.data_vars:
                    global_data = global_data.drop(['time_bnds','lat_bnds','lon_bnds'])
                global_data = global_data.transpose('time', 'lat', 'lon')
            elif variable_name == 'SIF':
                global_data = global_data.transpose('time', 'lat', 'lon')
            elif variable_name == 'lst_time_corrected':
                global_data = global_data.drop(['lst_total_uncertainty', 'qual_flag'])
            elif variable_name == 't2m':
                global_data = global_data.isel(latitude=slice(None, None, -1))
                land = xr.open_dataset("/prj/nceo/bethar/ERA5/ERA5_land_sea_mask.nc").squeeze()
                land = land.isel(latitude=slice(None, None, -1))
                regrid_target = global_data.copy()
                regrid_target = regrid_target.isel(latitude=slice(1, None))
                centred_lats = np.arange(-90, 90, 0.25) + 0.5*0.25
                centred_lons = np.arange(-180, 180, 0.25) + 0.5*0.25
                regrid_target = regrid_target.assign_coords({"latitude": centred_lats, "longitude": centred_lons})
                global_data_interpolated = global_data.interp_like(regrid_target)
                land_interpolated = land.interp_like(regrid_target)
                global_data = global_data_interpolated.where(land_interpolated.lsm>=0.5)
                global_data = global_data.rename({'latitude': 'lat'})
                global_data = global_data.rename({'longitude': 'lon'})
            elif variable_name == 'vimd' or variable_name == 'vpd':
                global_data = global_data.isel(latitude=slice(None, None, -1))
                land = xr.open_dataset("/prj/nceo/bethar/ERA5/ERA5_land_sea_mask.nc").squeeze()
                land = land.isel(latitude=slice(None, None, -1))
                regrid_target = global_data.copy()
                regrid_target = regrid_target.isel(latitude=slice(1, None))
                centred_lats = np.arange(-90, 90, 0.25) + 0.5*0.25
                centred_lons = np.arange(-180, 180, 0.25) + 0.5*0.25
                regrid_target = regrid_target.assign_coords({"latitude": centred_lats, "longitude": centred_lons})
                global_data_interpolated = global_data.interp_like(regrid_target)
                land_interpolated = land.interp_like(regrid_target)
                global_data_masked = global_data_interpolated.where(land_interpolated.lsm>=0.5)
                shifted_times = [t + np.timedelta64(12, 'h') for t in global_data_masked.time.data]
                global_data = global_data_masked.assign_coords(time=shifted_times)
                global_data = global_data.rename({'latitude': 'lat'})
                global_data = global_data.rename({'longitude': 'lon'})
            elif variable_name == 'swvl1':
                global_data = global_data.isel(latitude=slice(None, None, -1))
                land = xr.open_dataset("/prj/nceo/bethar/ERA5/ERA5_land_sea_mask.nc").squeeze()
                land = land.isel(latitude=slice(None, None, -1))
                regrid_target = global_data.copy()
                regrid_target = regrid_target.isel(latitude=slice(1, None))
                centred_lats = np.arange(-90, 90, 0.25) + 0.5*0.25
                centred_lons = np.arange(-180, 180, 0.25) + 0.5*0.25
                regrid_target = regrid_target.assign_coords({"latitude": centred_lats, "longitude": centred_lons})
                global_data_interpolated = global_data.interp_like(regrid_target)
                land_interpolated = land.interp_like(regrid_target)
                global_data_masked = global_data_interpolated.where(land_interpolated.lsm>=0.5)
                shifted_times = [t + np.timedelta64(12, 'h') for t in global_data_masked.time.data]
                global_data = global_data_masked.assign_coords(time=shifted_times)
                global_data = global_data.rename({'latitude': 'lat'})
                global_data = global_data.rename({'longitude': 'lon'})
            elif variable_name == 'wind_speed':
                global_data = global_data.isel(latitude=slice(None, None, -1))
                land = xr.open_dataset("/prj/nceo/bethar/ERA5/ERA5_land_sea_mask.nc").squeeze()
                land = land.isel(latitude=slice(None, None, -1))
                regrid_target = global_data.copy()
                regrid_target = regrid_target.isel(latitude=slice(1, None))
                centred_lats = np.arange(-90, 90, 0.25) + 0.5*0.25
                centred_lons = np.arange(-180, 180, 0.25) + 0.5*0.25
                regrid_target = regrid_target.assign_coords({"latitude": centred_lats, "longitude": centred_lons})
                global_data_interpolated = global_data.interp_like(regrid_target)
                land_interpolated = land.interp_like(regrid_target)
                global_data_masked = global_data_interpolated.where(land_interpolated.lsm>=0.5)
                shifted_times = [t + np.timedelta64(12, 'h') for t in global_data_masked.time.data]
                global_data = global_data_masked.assign_coords(time=shifted_times)
                global_data = global_data.rename({'latitude': 'lat'})
                global_data = global_data.rename({'longitude': 'lon'})
            elif variable_name == 'GPP':
                global_data = global_data.where(global_data[variable_name] > 0.)
                global_data = global_data.transpose('time', 'lat', 'lon')
            data = global_data.sel(lat=slice(-60, 80), lon=slice(-180, 180))
            data.astype(np.float32).to_netcdf(f'{SCRATCH_DIR}/rechunked_{variable_name}_{year}.nc',
                                                encoding={variable_name: {'contiguous': False, 
                                                'chunksizes': (days_in_year, 40, 40)}})


def read_rechunked_global_data(variable_name, min_year=2000, max_year=2020):
    rechunked_files = [f'{SCRATCH_DIR}/rechunked_{variable_name}_{year}.nc' for year in range(min_year, max_year+1)]
    global_data = xr.open_mfdataset(rechunked_files, chunks={"time": -1, "lat": 40, "lon": 40}, parallel=True)
    if variable_name == 'sm':
        global_data = global_data.where(global_data<1e6)
    global_data = global_data.sel(lat=slice(-60, 80), lon=slice(-180, 180))
    global_data = global_data.chunk(chunks={'time': -1, "lat": 280, "lon": 360}) # this should match tile size
    return global_data[variable_name]


def get_tile_indices(total_lats, total_lons, number_lat_tiles, number_lon_tiles):
    lat_tile_size = int(np.ceil(float(total_lats)/float(number_lat_tiles)))
    lon_tile_size = int(np.ceil(float(total_lons)/float(number_lon_tiles)))
    lat_slices = []
    for i in range(number_lat_tiles):
        end_tile = (i+1)*lat_tile_size
        if end_tile>=total_lats:
            end_tile = None
        lat_slices.append(slice(i*lat_tile_size, end_tile))
    lon_slices = []
    for j in range(number_lon_tiles):
        end_tile = (j+1)*lon_tile_size
        if end_tile>=total_lons:
            end_tile = None
        lon_slices.append(slice(j*lon_tile_size, end_tile))
    return lat_slices, lon_slices


def save_tiles(processor_function, data, number_lat_tiles, number_lon_tiles, tile_scratch_dir):
    # Fix coordinate names in xarray to prevent naming mismatches
    coord_names = data.coords._names
    if 'lat' in coord_names:
        data = data.rename({'lat': 'latitude'})
    if 'lon' in coord_names:
        data = data.rename({'lon': 'longitude'})
    lat_tile_slices, lon_tile_slices = get_tile_indices(data.latitude.size, data.longitude.size, 
                                                        number_lat_tiles, number_lon_tiles)
    os.system(f'mkdir -p {tile_scratch_dir}')
    os.system(f'rm {tile_scratch_dir}/*.nc')
    for i in tqdm(range(number_lat_tiles), desc='Processing tiles'):
        for j in range(number_lon_tiles):
            tile_id = i*number_lon_tiles + j
            data_tile = data.isel(latitude=lat_tile_slices[i], longitude=lon_tile_slices[j])
            tile_result = processor_function(data_tile)
            tile_result.to_netcdf(f'{tile_scratch_dir}/tile-output-{str(tile_id).zfill(3)}.nc')


def gather_tiles(variable_name, tile_scratch_dir, final_save_name, save_by_year=True, cleanup=False):
    save_dir = '/'.join(final_save_name.split('/')[:-1])
    os.system(f'mkdir -p {save_dir}')
    all_tiles = xr.open_mfdataset(f'{tile_scratch_dir}/*.nc', parallel=True)
    encoding_settings = dict(zlib=True, contiguous=False, chunksizes=(365, 40, 40))
    encoding = {variable_name: encoding_settings}
    if save_by_year:
        save_name_stem = final_save_name.split('.nc')[0]
        years, datasets = zip(*all_tiles.groupby("time.year"))
        paths = [f"{save_name_stem}_{y}.nc" for y in years]
        write_job = xr.save_mfdataset(datasets, paths, compute=False, encoding=encoding)
    else:
        write_job = all_tiles.to_netcdf(final_save_name, compute=False, encoding=encoding)
    with ProgressBar():
        write_job.compute()
    if cleanup:
        os.system(f'rm {tile_scratch_dir}/*.nc')


def process_by_tiles(variable_name, processor_function, data, number_lat_tiles, number_lon_tiles, 
                     tile_scratch_dir, final_save_name, save_by_year=True, cleanup=False):
    save_tiles(processor_function, data, number_lat_tiles, number_lon_tiles, tile_scratch_dir)
    gather_tiles(variable_name, tile_scratch_dir, final_save_name, save_by_year=save_by_year, cleanup=cleanup)


def save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, min_year=2000, max_year=2020,
                                save_by_year=True, cleanup=False):
    rechunk_global_files(data_directory, variable_name, min_year=min_year, max_year=max_year)
    global_data = read_rechunked_global_data(variable_name, min_year=min_year, max_year=max_year)
    process_by_tiles(variable_name, daily_anomalies_standardised_rolling, global_data, 
                     number_lat_tiles, number_lon_tiles, tile_scratch_dir,
                     final_save_name, save_by_year=save_by_year, cleanup=cleanup)


def save_soil_moisture_standardised_anomalies():
    variable_name = 'sm'
    data_directory = '/prj/swift/ESA_CCI_SM/year_files_v8.1_combined_GLOBAL'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/soil_moisture_standardised_anomalies/ESA-CCI-SM_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)


def save_CCI_RZSM_standardised_anomalies():
    variable_name = 'rzsm_10cm'
    data_directory = '/prj/nceo/bethar/ESA_CCI_RZSM/10cm/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/ESA_CCI_RZSM_standardised_anomalies/10cm/ESA-CCI-RZSM_10cm_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)


def save_ERA5_SSM_standardised_anomalies():
    variable_name = 'swvl1'
    data_directory = '/prj/nceo/bethar/ERA5/swvl1/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/ERA5_soil_moisture_standardised_anomalies/swvl1/ERA5_swvl1_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)


def save_VOD_standardised_anomalies():
    variable_name = 'vod'
    data_directory = '/prj/nceo/bethar/VODCA_global/filtered/filtered_surface_water/X-band'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/VOD_standardised_anomalies/VODCA_X-band_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, max_year=2018, save_by_year=True, cleanup=True)


def save_VOD_v2_standardised_anomalies():
    variable_name = 'VODCA_CXKu'
    data_directory = '/prj/nceo/bethar/VODCA_global/VODCA_CXKu/year_files/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/VOD_v2_standardised_anomalies/VODCA_v2_CXKu_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, max_year=2020, save_by_year=True, cleanup=True)


def save_VOD_v2_standardised_anomalies_SIFtime():
    variable_name = 'VODCA_CXKu'
    data_directory = '/prj/nceo/bethar/VODCA_global/VODCA_CXKu/year_files/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/VOD_v2_SIFyears_standardised_anomalies/VODCA_v2_CXKu_SIFyears_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, min_year=2007, max_year=2018, save_by_year=True, cleanup=True)


def save_VOD_no_surface_water_filter_standardised_anomalies():
    variable_name = 'vod'
    data_directory = '/prj/nceo/bethar/VODCA_global/filtered/X-band/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/VOD_v1_nofilter_standardised_anomalies/VODCA_v1_Xnofilter_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, max_year=2018, save_by_year=True, cleanup=True)



def save_t2m_standardised_anomalies():
    variable_name = 't2m'
    data_directory = '/prj/nceo/bethar/ERA5/2m_temperature/local_time/afternoon/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/T2m_standardised_anomalies/T2m_ERA5_daily_max_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, max_year=2020, save_by_year=True, cleanup=True)



def save_precip_standardised_anomalies():
    variable_name = 'precipitationCal'
    data_directory = '/prj/nceo/bethar/IMERG/regrid_p25_global/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/precip_standardised_anomalies/pr_IMERG_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)


def save_mw_lst_standardised_anomalies():
    variable_name = 'lst_time_corrected'
    data_directory = '/prj/nceo/bethar/ESA_CCI_LST/MW-LST/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/MW-LST_standardised_anomalies/ESA_CCI_MW-LST_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)


def save_lst_t2m_diff_MW_18_standardised_anomalies():
    variable_name = 'lst-t2m_MW_18'
    data_directory = 'SCRATCH_DIR'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/LST-T2m_MW_18_standardised_anomalies/ESA_CCI_MW-LST_diff_T2m_1800_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)


def save_vimd_standardised_anomalies():
    variable_name = 'vimd'
    data_directory = '/prj/nceo/bethar/ERA5/vimd/local_afternoon_max/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/vimd_standardised_anomalies/ERA5_vimd_afternoon_max_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)


def save_vpd_standardised_anomalies():
    variable_name = 'vpd'
    data_directory = '/prj/nceo/bethar/ERA5/vpd/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/vpd_standardised_anomalies/ERA5_vpd_afternoon_mean_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)



def save_SIF_PK_standardised_anomalies():
    variable_name = 'SIF'
    data_directory = '/prj/nceo/bethar/GOME2-SIF/0pt25deg/PK'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/SIF-GOME2_PK_standardised_anomalies/SIF-GOME2_PK_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True, min_year=2007, max_year=2018)


def save_CERES_net_standardised_anomalies():
    variable_name = 'net_surface_rad'
    data_directory = '/prj/nceo/bethar/CERES-rsds/all_components/net_0pt25deg/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/CERES_rad_standardised_anomalies/CERES_net_sfc_rad_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)


def save_CERES_sw_down_standardised_anomalies():
    variable_name = 'downwelling_surface_sw_rad'
    data_directory = '/prj/nceo/bethar/CERES-rsds/all_components/sw_down_0pt25deg/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/CERES_sw_down_rad_standardised_anomalies/CERES_sw_down_sfc_rad_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)


def save_GLEAM_evap_standardised_anomalies():
    variable_name = 'E'
    data_directory = '/prj/nceo/bethar/GLEAMv38a/E/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/GLEAM_E_v38a_testrepl_standardised_anomalies/GLEAM_E_v38a_testrepl_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)
    

def save_GLEAM_v4_evap_standardised_anomalies():
    variable_name = 'E'
    data_directory = '/prj/nceo/bethar/GLEAMv42a/0pt25deg/E/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/GLEAM_v42a_E_standardised_anomalies/GLEAM_E_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)


def save_GLEAM_RZSM_standardised_anomalies():
    variable_name = 'SMroot'
    data_directory = '/prj/nceo/bethar/GLEAMv38a/SMroot/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/root_zone_soil_moisture_standardised_anomalies/GLEAM_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)
    

def save_GLEAM_v4_RZSM_standardised_anomalies():
    variable_name = 'SMrz'
    data_directory = '/prj/nceo/bethar/GLEAMv42a/0pt25deg/SMrz/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/root_zone_soil_moisture_v42a_tandardised_anomalies/GLEAM_v42a_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)


def save_GLEAM_SSM_standardised_anomalies():
    variable_name = 'SMsurf'
    data_directory = '/prj/nceo/bethar/GLEAMv38a/SMsurf/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/GLEAM_surface_soil_moisture_standardised_anomalies/GLEAM_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)


def save_wind_speed_standardised_anomalies():
    variable_name = 'wind_speed'
    data_directory = '/prj/nceo/bethar/ERA5/wind10m/'
    tile_scratch_dir = f'{SCRATCH_DIR}/dask_tiling_workspace'
    final_save_name = '/prj/nceo/bethar/SUBDROUGHT/wind_speed_10m_standardised_anomalies/ERA5_wind_speed_10m_afternoon_mean_standardised_anomaly.nc'
    number_lat_tiles = 2
    number_lon_tiles = 4
    save_standardised_anomalies(data_directory, variable_name, tile_scratch_dir, final_save_name,
                                number_lat_tiles, number_lon_tiles, save_by_year=True, cleanup=True)


if __name__ == '__main__':
    start = time.time()
    save_GLEAM_evap_standardised_anomalies()
    end = time.time()
    print(f'TOTAL PROCESSING TIME: {end-start:.2f} seconds')