import xarray as xr
import numpy as np
import numpy.ma as ma
import pandas as pd
from dask.diagnostics import ProgressBar
from tqdm import tqdm


std_anom_dirs = {'sm': '/prj/nceo/bethar/SUBDROUGHT/soil_moisture_standardised_anomalies/',
                 'lst-t2m': '/prj/nceo/bethar/ESA_CCI_LST/MODIS_AQUA_LST-T2m/v4.00',
                 'lst-t2m_MW_18': '/prj/nceo/bethar/SUBDROUGHT/LST-T2m_MW_18_standardised_anomalies/',
                 'vod': '/prj/nceo/bethar/SUBDROUGHT/VOD_standardised_anomalies/',
                 'VODCA_CXKu': '/prj/nceo/bethar/SUBDROUGHT/VOD_v2_standardised_anomalies/single_precision/',
                 'E': '/prj/nceo/bethar/SUBDROUGHT/GLEAM_E_standardised_anomalies/',
                 't2m': '/prj/nceo/bethar/SUBDROUGHT/T2m_standardised_anomalies/',
                 'wind_speed': '/prj/nceo/bethar/SUBDROUGHT/wind_speed_10m_standardised_anomalies',
                 'precipitationCal': '/prj/nceo/bethar/SUBDROUGHT/precip_standardised_anomalies/',
                 'rzsm_40cm': '/prj/nceo/bethar/SUBDROUGHT/ESA_CCI_RZSM_standardised_anomalies/40cm/',
                 'SMroot': '/prj/nceo/bethar/SUBDROUGHT/root_zone_soil_moisture_standardised_anomalies/',
                 'net_surface_rad': '/prj/nceo/bethar/SUBDROUGHT/CERES_rad_standardised_anomalies/',
                 'downwelling_surface_sw_rad': '/prj/nceo/bethar/SUBDROUGHT/CERES_sw_down_rad_standardised_anomalies/',
                 'lst_time_corrected': '/prj/nceo/bethar/SUBDROUGHT/MW-LST_standardised_anomalies/',
                 'lst_aqua': "/prj/nceo/bethar/ESA_CCI_LST/MODIS_Aqua_LST_std_anoms/",
                 'vimd': '/prj/nceo/bethar/SUBDROUGHT/vimd_standardised_anomalies/mean/',
                 'SIF_JJ': '/prj/nceo/bethar/SUBDROUGHT/SIF-GOME2_JJ_standardised_anomalies/',
                 'SIF_PK': '/prj/nceo/bethar/SUBDROUGHT/SIF-GOME2_PK_standardised_anomalies/',
                 'SESR_ERA5': '/prj/nceo/bethar/SUBDROUGHT/ERA5_SESR/single_precision'
}


def add_anomaly_column(catalogue_filename, save_catalogue_filename, anomaly_variable_name, anomaly_type, days_start, days_end, smooth=None):
    # e.g. for minimum VOD from day 60 to day 30 before onset use VOD, 'min', -60, -30
    drought_event_catalogue = pd.read_csv(catalogue_filename)
    std_anom = xr.open_mfdataset(f'{std_anom_dirs[anomaly_variable_name]}/*.nc')
    if anomaly_variable_name.startswith('SIF_'):
        xr_variable_name = 'SIF'
    elif anomaly_variable_name == 'SESR_ERA5':
        xr_variable_name = 'ESR'
    else:
        xr_variable_name = anomaly_variable_name
    times = std_anom.time.data
    date_strs = [np.datetime_as_string(t, unit='D') for t in times]
    coord_names = std_anom.coords._names
    if 'lat' in coord_names:
        std_anom = std_anom.rename({'lat': 'latitude'})
    if 'lon' in coord_names:
        std_anom = std_anom.rename({'lon': 'longitude'})
    std_anom = std_anom.chunk({'time': 365, 'latitude': 40, 'longitude': 40})
    lons = np.arange(-180, 180, 0.25) + 0.5*0.25
    lats = np.arange(-60, 80, 0.25) + 0.5*0.25
    if smooth is not None:
        std_anom_array = std_anom[xr_variable_name].rolling(time=smooth, center=True, min_periods=2).mean()
    else:
        std_anom_array = std_anom[xr_variable_name] 
    with ProgressBar():
        std_anom_data = std_anom_array.data.compute()
    event_anomalies = np.ones(len(drought_event_catalogue),) * np.nan
    for index, event in tqdm(drought_event_catalogue.iterrows()):
        lat_idx = np.argmin(np.abs(event['latitude (degrees north)'] - lats))
        lon_idx = np.argmin(np.abs(event['longitude (degrees east)'] - lons))
        onset_day_idx = np.where(np.array(date_strs)==event['start date'])[0]
        if len(onset_day_idx)==1:
            onset_day_idx = onset_day_idx[0]
            start_time_idx = onset_day_idx+days_start
            end_time_idx = onset_day_idx+days_end+1
            if start_time_idx>=0 and end_time_idx<times.size:
                px_data = std_anom_data[start_time_idx:end_time_idx, lat_idx, lon_idx]
                if anomaly_type == 'mean':
                    event_anomalies[index] = np.nanmean(px_data)
                elif anomaly_type == 'min':
                    event_anomalies[index] = np.nanmin(px_data)
                elif anomaly_type == 'max':
                    event_anomalies[index] = np.nanmax(px_data)
                elif anomaly_type == 'sum':
                    event_anomalies[index] = np.nansum(px_data)
    column_name = f'std_anom_{anomaly_variable_name}_{anomaly_type}_{days_start}_{days_end}'
    if smooth is not None:
        column_name += f'_smooth{smooth}'
    drought_event_catalogue[column_name] = event_anomalies
    drought_event_catalogue.to_csv(save_catalogue_filename, index=False)


def add_n_weighted_anomaly_column(catalogue_filename, save_catalogue_filename, anomaly_variable_name, anomaly_type, days_start, days_end, smooth=None):
    drought_event_catalogue = pd.read_csv(catalogue_filename)
    cat = {}
    south1 = drought_event_catalogue['latitude (degrees north)'] < -30.
    south2 = np.logical_and(drought_event_catalogue['latitude (degrees north)'] >= -30., drought_event_catalogue['latitude (degrees north)'] < 0.)
    north1 = np.logical_and(drought_event_catalogue['latitude (degrees north)'] >= 0., drought_event_catalogue['latitude (degrees north)'] < 20.)
    north2 = np.logical_and(drought_event_catalogue['latitude (degrees north)'] >= 20., drought_event_catalogue['latitude (degrees north)'] < 40.)
    north3 = np.logical_and(drought_event_catalogue['latitude (degrees north)'] >= 40., drought_event_catalogue['latitude (degrees north)'] < 50.)
    north4 = np.logical_and(drought_event_catalogue['latitude (degrees north)'] >= 50., drought_event_catalogue['latitude (degrees north)'] < 60.)
    north5 = np.logical_and(drought_event_catalogue['latitude (degrees north)'] >= 60., drought_event_catalogue['latitude (degrees north)'] < 70.)
    north6 = drought_event_catalogue['latitude (degrees north)'] >= 70.
    cat['south1'] = drought_event_catalogue.drop(drought_event_catalogue[~south1].index)
    cat['south1'].reset_index(drop=True, inplace=True)
    cat['south2'] = drought_event_catalogue.drop(drought_event_catalogue[~south2].index)
    cat['south2'].reset_index(drop=True, inplace=True)
    cat['north1'] = drought_event_catalogue.drop(drought_event_catalogue[~north1].index)
    cat['north1'].reset_index(drop=True, inplace=True)
    cat['north2'] = drought_event_catalogue.drop(drought_event_catalogue[~north2].index)
    cat['north2'].reset_index(drop=True, inplace=True)
    cat['north3'] = drought_event_catalogue.drop(drought_event_catalogue[~north3].index)
    cat['north3'].reset_index(drop=True, inplace=True)
    cat['north4'] = drought_event_catalogue.drop(drought_event_catalogue[~north4].index)
    cat['north4'].reset_index(drop=True, inplace=True)
    cat['north5'] = drought_event_catalogue.drop(drought_event_catalogue[~north5].index)
    cat['north5'].reset_index(drop=True, inplace=True)
    cat['north6'] = drought_event_catalogue.drop(drought_event_catalogue[~north6].index)
    cat['north6'].reset_index(drop=True, inplace=True)
    std_anom = xr.open_mfdataset(f'{std_anom_dirs[anomaly_variable_name]}/*.nc', chunks={"time": -1, "latitude": 40, "longitude": 40}, parallel=True)
    times = std_anom.time.data
    date_strs = [np.datetime_as_string(t, unit='D') for t in times]
    coord_names = std_anom.coords._names
    if 'lat' in coord_names:
        std_anom = std_anom.rename({'lat': 'latitude'})
    if 'lon' in coord_names:
        std_anom = std_anom.rename({'lon': 'longitude'})
    std_anom = std_anom.chunk({'time': 365, 'latitude': 40, 'longitude': 40})
    if smooth is not None:
        std_anoms_global = std_anom[anomaly_variable_name].rolling(time=smooth, center=True, min_periods=2).mean()
        n_global = std_anom['n'].rolling(time=smooth, center=True, min_periods=2).mean()
    else:
        std_anoms_global = std_anom[anomaly_variable_name]
        n_global = std_anom['n']
    lons = np.arange(-180, 180, 0.25) + 0.5*0.25
    for tile in ['north1', 'north2', 'north3', 'north4', 'north5', 'north6', 'south1', 'south2']:
        if tile == 'north1':
            std_anom_array = std_anoms_global.sel(latitude=slice(0, 20))
            n_array = n_global.sel(latitude=slice(0, 20))
            lats = np.arange(0, 20, 0.25) + 0.5*0.25
        elif tile == 'north2':
            std_anom_array = std_anoms_global.sel(latitude=slice(20, 40))
            n_array = n_global.sel(latitude=slice(20, 40))
            lats = np.arange(20, 40, 0.25) + 0.5*0.25
        elif tile == 'north3':
            std_anom_array = std_anoms_global.sel(latitude=slice(40, 50))
            n_array = n_global.sel(latitude=slice(40, 50))
            lats = np.arange(40, 50, 0.25) + 0.5*0.25
        elif tile == 'north4':
            std_anom_array = std_anoms_global.sel(latitude=slice(50, 60))
            n_array = n_global.sel(latitude=slice(50, 60))
            lats = np.arange(50, 60, 0.25) + 0.5*0.25
        elif tile == 'north5':
            std_anom_array = std_anoms_global.sel(latitude=slice(60, 70))
            n_array = n_global.sel(latitude=slice(60, 70))
            lats = np.arange(60, 70, 0.25) + 0.5*0.25
        elif tile == 'north6':
            std_anom_array = std_anoms_global.sel(latitude=slice(70, 80))
            n_array = n_global.sel(latitude=slice(70, 80))
            lats = np.arange(70, 80, 0.25) + 0.5*0.25
        elif tile == 'south1':
            std_anom_array = std_anoms_global.sel(latitude=slice(-60, -30))
            n_array = n_global.sel(latitude=slice(-60, -30))
            lats = np.arange(-60, -30, 0.25) + 0.5*0.25
        elif tile == 'south2':
            std_anom_array = std_anoms_global.sel(latitude=slice(-30, 0))
            n_array = n_global.sel(latitude=slice(-30, 0))
            lats = np.arange(-30, 0, 0.25) + 0.5*0.25
        with ProgressBar():
            std_anom_data = std_anom_array.data.compute()
            n_data = n_array.data.compute()
        std_anom_data[n_data < 625./5.] = np.nan
        event_anomalies = np.ones(len(cat[tile]),) * np.nan
        event_ns = np.zeros(len(cat[tile]),).astype(int)
        for index, event in tqdm(cat[tile].iterrows()):
            lat_idx = np.argmin(np.abs(event['latitude (degrees north)'] - lats))
            lon_idx = np.argmin(np.abs(event['longitude (degrees east)'] - lons))
            onset_day_idx = np.where(np.array(date_strs)==event['start date'])[0]
            if len(onset_day_idx)==1:
                onset_day_idx = onset_day_idx[0]
                start_time_idx = onset_day_idx+days_start
                end_time_idx = onset_day_idx+days_end+1
                if start_time_idx>=0 and end_time_idx<times.size:
                    px_data = std_anom_data[start_time_idx:end_time_idx, lat_idx, lon_idx]
                    px_n = n_data[start_time_idx:end_time_idx, lat_idx, lon_idx]
                    if np.all(np.isnan(px_data)):
                        event_anomalies[index] = np.nan
                        event_ns[index] = 0
                    else:
                        if anomaly_type == 'mean':
                            weighted_mean = ma.average(ma.masked_invalid(px_data), weights=px_n)
                            total_n = px_n.sum()
                            event_anomalies[index] = weighted_mean
                            event_ns[index] = total_n
                        elif anomaly_type == 'min':
                            min_idx = np.nanargmin(px_data)
                            event_anomalies[index] = px_data[min_idx]
                            event_ns[index] = px_n[min_idx]
                        elif anomaly_type == 'max':
                            max_idx = np.nanargmin(px_data)
                            event_anomalies[index] = px_data[max_idx]
                            event_ns[index] = px_n[max_idx]
        anom_column_name = f'std_anom_{anomaly_variable_name}_{anomaly_type}_{days_start}_{days_end}'
        n_column_name = f'n_{anomaly_variable_name}_{anomaly_type}_{days_start}_{days_end}'
        if anomaly_type == 'mean':
            anom_column_name += 'nweighted'
        if smooth is not None:
            anom_column_name += f'_smooth{smooth}'
            n_column_name += f'_smooth{smooth}'
        cat[tile][anom_column_name] = event_anomalies
        cat[tile][n_column_name] = event_ns
    all_events = pd.concat([cat['north1'], cat['north2'], cat['north3'], cat['north4'], cat['north5'], cat['north6'], cat['south1'], cat['south2']], ignore_index=True)
    all_events.to_csv(save_catalogue_filename, index=False)


if __name__ == '__main__':
    catalogue_filename = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/flash_drought_event_catalogue.csv'
    save_catalogue_filename = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/flash_drought_event_catalogue_with_event_anomalies.csv'
    # add_anomaly_column(catalogue_filename, save_catalogue_filename, 't2m', 'max', 0, 20, smooth=5)
    # add_n_weighted_anomaly_column(save_catalogue_filename, save_catalogue_filename, 'lst-t2m', 'max', 0, 20, smooth=5)
    # add_n_weighted_anomaly_column(save_catalogue_filename, save_catalogue_filename, 'lst-t2m', 'mean', -60, -30)
    add_anomaly_column(save_catalogue_filename, save_catalogue_filename, 'VODCA_CXKu', 'mean', -60, -30)
