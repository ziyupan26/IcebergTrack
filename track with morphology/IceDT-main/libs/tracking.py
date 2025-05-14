import warnings
warnings.filterwarnings("ignore")

import numpy as np
import difflib
import sys
import matplotlib.pyplot as plt
import getMetadata as gmeta # libs
import gc
import myGeoTools as mgt # libs
import pandas as pd
import time
import json

from fastdtw import fastdtw
from scipy import stats
from geopy.distance import geodesic as geodist
from similarityMeasures import Similarity # libs
from datetime import date
from IPython.display import clear_output # progressbar

def check_heuristics(delta_km, delta_days, area, speed, pairs=False):

    speed_tsh = 0.
    days_tsh = 0.

    if 0. < area <= 1.:
        radius = 9. * delta_days
        speed_tsh = 6.5 * 2
        days_tsh = 25

    if 1. < area <= 10.:
        radius = 6.5 * delta_days #* 2 7.5
        speed_tsh = 6.5 * 2
        days_tsh = 40

    if 10. < area <= 100.:
        radius = 4 * delta_days #* 2 5.5
        speed_tsh = 4. * 2
        days_tsh = 60

    if 100. < area <= 1000.:
        radius = 2.3 * delta_days * 1.5
        speed_tsh = 2.3 * 2
        days_tsh = 70 if area < 500 else 120

    if area > 1000.:
        radius = 2.5 * delta_days * 2 # 2.5
        speed_tsh = 2.32 * 2
        days_tsh = 90 if area < 1500 else 180

    #is_dist_max = True if delta_km < max_dist else False #550
    is_dist_max = True
    is_radius = True if delta_km <= radius else False
    is_speed = True if speed <= speed_tsh else False
    is_days = True if delta_days <= days_tsh else False

    return [is_dist_max, is_radius, is_speed, is_days]

def moving_average(x, w):
    return (np.convolve(x, np.ones(w), 'valid') / w) ##.astype(int)

def clean_track(iceberg_to_clean, berg_individual_id, main_bar, smooth_len):

    save_base = iceberg_to_clean.copy()

    total_samples = len(iceberg_to_clean.index)

    last_id_valid = iceberg_to_clean.first_valid_index()
    last_id_valid_check = -1

    indexes_used = []

    while len(iceberg_to_clean.index) > 1:

        clear_output(wait=True)
        bar_length=20
        progress = float(total_samples-len(iceberg_to_clean.index))/(total_samples)
        block = int(round(bar_length * progress))
        progressbarmain = "Global progress: [{0}] {1:.1f}%".format( "#" * int(main_bar[0]) + "-" * (bar_length - int(main_bar[0])), main_bar[1] * 100.)
        progressbar = "Filtering progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100.)
        print(progressbarmain)
        print(progressbar, 'Total detections for current track: ', len(indexes_used))

        if last_id_valid != last_id_valid_check:

            area_base = iceberg_to_clean.loc[last_id_valid,'area_km2']

            iceberg_base_signature = moving_average(np.sort(np.asarray(json.loads(iceberg_to_clean.loc[last_id_valid,'shape']))).tolist(), smooth_len)

            iceberg_to_clean['jcd_to_base'] = iceberg_to_clean.apply(lambda row : measures.jaccard_similarity(iceberg_base_signature, moving_average(np.sort(np.asarray(json.loads(row['shape']))).tolist(),smooth_len)), axis = 1)
            iceberg_to_clean['k_s'] = iceberg_to_clean.apply(lambda row : 1 - stats.ks_2samp(iceberg_base_signature, moving_average(np.sort(np.asarray(json.loads(row['shape']))).tolist(),smooth_len))[0], axis = 1)
            #iceberg_to_clean['vonmisses'] = iceberg_to_clean.apply(lambda row : 1 - stats.cramervonmises_2samp(iceberg_base_signature, moving_average(np.sort(np.asarray(json.loads(row['shape']))).tolist(),smooth_len)).statistic, axis = 1)
            #iceberg_to_clean.loc[iceberg_to_clean['vonmisses'] < 0, 'vonmisses'] = 0

            iceberg_to_clean['sim_score'] = (iceberg_to_clean['jcd_to_base'] + iceberg_to_clean['k_s'])/2
            iceberg_to_clean = iceberg_to_clean.sort_values(by=['sim_score'], ascending=[False])

            if area_base < 1000:
                sim_tsh = 0.9
                sim_id = 1
            else:
                sim_tsh = 0.85
                sim_id = 1

        next_id = iceberg_to_clean.index[1]

        sims = [iceberg_to_clean.loc[next_id,'jcd_to_base'],
                iceberg_to_clean.loc[next_id,'k_s'],
                iceberg_to_clean.loc[next_id,'sim_score']]

        #if sim_pair_k_s > sim_tsh: #0.85
        if sims[sim_id] > sim_tsh:

            indexes_used.append(last_id_valid)
            iceberg_to_clean.drop(last_id_valid, inplace=True)
            last_id_valid_check = last_id_valid
            last_id_valid = next_id

        else:

            iceberg_to_clean.drop(next_id, inplace=True)
            last_id_valid_check = last_id_valid


    if len(indexes_used) >= 5:

        berg_to_save = save_base.loc[indexes_used, :].sort_values(by=['datetime'], ascending=[True])
        berg_to_look = berg_to_save.copy()

        indexes_used_final = []
        while len(berg_to_look.index) > 0:

            indexes_used = berg_to_look.index.tolist()

            idx_start_look = berg_to_look.first_valid_index()
            idx_ant = idx_start_look
            row_ant = berg_to_look.loc[idx_ant, :]

            for index, row in berg_to_save.loc[idx_ant:, :].iterrows():

                droped = False
                if idx_ant != index:

                    dkm = round(geodist((row_ant['latitude'], row_ant['longitude']), (row['latitude'], row['longitude'])).km, 3)
                    ddays = abs((row['datetime'] - row_ant['datetime']).days)

                    ddays = 1. if ddays == 0. else ddays
                    dspeed = round(dkm/ddays, 3)

                    area_change = abs(row['area_km2'] - row_ant['area_km2'])
                    area_dst = area_change / row_ant['area_km2']

                    heuristics = check_heuristics(dkm, ddays, row_ant['area_km2'], dspeed, pairs=True)

                    #if not all(heuristics):
                    if not heuristics[1] or not heuristics[3] or not area_dst < 0.21 :
                        droped = True
                        indexes_used.pop(indexes_used.index(index))

                if not droped:
                    row_ant = row
                    idx_ant = index

            if len(indexes_used) > len(indexes_used_final):
                indexes_used_final = indexes_used

            berg_to_look.drop(idx_start_look, inplace=True)


        berg_to_look = berg_to_save.loc[indexes_used_final]
        if berg_to_look.shape[0] >= 3:
            berg_to_look.to_csv('/content/drive/MyDrive/icebergtrack/sentinel1/'+str(berg_individual_id)+'_'+str(indexes_used_final[0])+'.txt', mode = 'w', columns = ['datetime','latitude','longitude','majoraxis_km', 'area_km2'], sep=' ', index=False)

    else:
        indexes_used_final = indexes_used

    return indexes_used_final

# Start measuring the processing time
t0 = time.process_time()

# Sort the iceberg dataframe by 'datetime' in ascending order
iceberg_control = detected_icebergs.copy().sort_values(by='datetime', ascending=True)
# iceberg_control = detected_icebergs

# Initialize the index to start processing the first iceberg
idx = iceberg_control.first_valid_index()

measures = Similarity()

original_iceberg_data = iceberg_control.copy()  # Save original iceberg data

# Start the main loop to process each iceberg until none remain
while len(iceberg_control.index) > 0:

    # Initialize progress bar variables
    progressbar = 0
    indexes = iceberg_control.index
    idx = iceberg_control.first_valid_index()

    # Perform garbage collection to free memory
    gc.collect()

    # If the current iceberg is still valid, proceed
    if idx in indexes:

        # Clear the previous output and update the progress bar
        clear_output(wait=True)
        bar_length = 20
        progress = float(num_ices - len(indexes)) / num_ices  # Progress as a percentage
        block = int(round(bar_length * progress))
        progressbar = "Global progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100.)
        print(progressbar)

        # Track the progress and display the current block and progress percentage
        main_bar = [block, progress]

        # Retrieve the current iceberg's data
        iceberg_base = iceberg_control.loc[idx, :]

        # Extract the 'datetime' and 'area_km2' of the current iceberg
        date_iceberg_base = iceberg_base['datetime']
        area_iceberg_base = iceberg_base['area_km2']

        # Adjust the smoothing length based on iceberg size
        if area_iceberg_base > 50:
            smooth_len = 3  # Larger icebergs use a larger smoothing window
        else:
            smooth_len = 2  # Smaller icebergs use a smaller smoothing window

        # Adjust the threshold for the number of years to search for similar icebergs based on size
        if area_iceberg_base > 100:
            delta_years_ths = 10  # Larger icebergs have a longer time threshold
        else:
            delta_years_ths = 5  # Smaller icebergs have a shorter time threshold

        # Create a morpho-signature of the iceberg by applying a moving average to its shape
        iceberg_base_morpho_signature = moving_average(np.sort(np.asarray(json.loads(iceberg_base['shape']))).astype(int).tolist(), smooth_len)

        # Filter icebergs based on their longitude (only considering those in the Weddell Sea for now)
        # c2 = iceberg_control['longitude'] > -65  # Longitude threshold for Weddell Sea
        # possible_icebergs_df = iceberg_control.loc[c2]
        possible_icebergs_df = iceberg_control.copy()

        # Further filter by icebergs that are within the time threshold 'delta_years_ths'
        possible_icebergs_df = possible_icebergs_df[((possible_icebergs_df['datetime'] - iceberg_base['datetime']).dt.days) / 365 <= delta_years_ths]

        # If there are any possible matching icebergs, proceed
        if not possible_icebergs_df.empty:

            # Calculate similarity scores using Jaccard similarity and Kolmogorov-Smirnov test
            possible_icebergs_df['jcd_to_base'] = possible_icebergs_df.apply(lambda row: measures.jaccard_similarity(iceberg_base_morpho_signature, moving_average(np.sort(np.asarray(json.loads(row['shape']))).tolist(), smooth_len)), axis=1)
            possible_icebergs_df['k_s'] = possible_icebergs_df.apply(lambda row: 1 - stats.ks_2samp(iceberg_base_morpho_signature, moving_average(np.sort(np.asarray(json.loads(row['shape']))).tolist(), smooth_len))[0], axis=1)

            # Compute the average similarity score between Jaccard and KS similarity measures
            possible_icebergs_df['sim_score'] = (possible_icebergs_df['jcd_to_base'] + possible_icebergs_df['k_s']) / 2

            # Filter icebergs with a similarity score greater than 0.5 and sort by score in descending order
            possible_individual_iceberg_df = possible_icebergs_df[(possible_icebergs_df['sim_score'] > 0.5)].sort_values(by='sim_score', ascending=False)

            # If there are matching icebergs, find the index to drop
            if not len(possible_individual_iceberg_df['jcd_to_base']) == 0:
                print("there are matching icebergs")
                index_to_drop = clean_track(possible_individual_iceberg_df, idx, main_bar, smooth_len)

                # If no matching icebergs found, keep the current index
                if index_to_drop == []:
                    index_to_drop = idx
            else:
                index_to_drop = idx

            # Drop the matched icebergs from the main dataframe
            iceberg_control.drop(index_to_drop, inplace=True)

        else:
            # If no matching icebergs are found, drop the current iceberg
            print("no matching icebergs are found")
            iceberg_control.drop(indexes, inplace=True)

    # Finalize progress after each iteration
    clear_output(wait=True)
    bar_length = 20
    progress = float(num_ices) / num_ices
    block = int(round(bar_length * progress))
    progressbar = "Global progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100.)
    print(progressbar)

# Calculate the total processing time and print it
t1 = (time.process_time() - t0) / 60.
print("Total time elapsed: ", round(t1, 3), 'minutes')  # CPU time elapsed in minutes