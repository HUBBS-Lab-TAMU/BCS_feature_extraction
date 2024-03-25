import numpy as np
import pandas as pd
from numpy import mean
from scipy.signal import butter, filtfilt
import neurokit2 as nk


# Filter EDA data
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


# Extract EDA features
def eda_features(edaPath, ):
    cutoff_freq = 0.1
    edaSamplingRate = 4
    for subNum in range(1, 68):
        finalDF = pd.DataFrame()
        finalList_seg = []
        finalList_eda = []
        finalList_amp = []
        finalList_rise = []
        finalList_reco = []

        # Read the current subject's EDA data and filter it
        edaDF = pd.read_csv(edaPath[subNum])
        eda_data = edaDF['EDA'].values.tolist()
        b, a = butter_lowpass(cutoff_freq, edaSamplingRate, order=5)
        edaDF_filtered = filtfilt(b, a, eda_data)

        # Normalize filtered EDA data
        edaDF_filtered['EDA_norm'] = (edaDF_filtered['EDA_filtered'] - edaDF_filtered['EDA_filtered'].min()) / (
                    edaDF_filtered['EDA_filtered'].max() - edaDF_filtered['EDA_filtered'].min())

        # Go through each segment number 7-23
        for i in range(7, 24):
            # Isolate EDA data from segment i
            eda_segment = edaDF_filtered.loc[edaDF_filtered['Segment'] == i]
            eda_signal = eda_segment["EDA_filtered"].values.tolist()
            edaNormList = eda_segment["EDA_norm"].values.tolist()

            # Skip any empty segments
            if len(eda_signal) == 0:
                continue

            # Analyze the EDA data
            try:
                signals, info = nk.eda_process(eda_signal, sampling_rate=4)
            except:
                finalList_seg.append(i)
                finalList_eda.append(mean(eda_signal))
                finalList_amp.append(np.nan)
                finalList_rise.append(np.nan)
                finalList_reco.append(np.nan)

            finalList_seg.append(i)
            finalList_eda.append(mean(eda_signal))
            finalList_amp.append(mean(info['SCR_Amplitude']))
            finalList_rise.append(mean(info['SCR_RiseTime']))
            finalList_reco.append(mean(info['SCR_RecoveryTime']))

    # Add all features to a dataframe
    finalDF['Segment'] = finalList_seg
    finalDF['EDA_Norm_Avg'] = finalList_eda
    finalDF['Amplitude_Avg'] = finalList_amp
    finalDF['Amplitude_Avg'].fillna((finalDF['Amplitude_Avg'].mean()), inplace=True)
    finalDF['Rise_Time_Avg'] = finalList_rise
    finalDF['Rise_Time_Avg'].fillna((finalDF['Rise_Time_Avg'].mean()), inplace=True)
    finalDF['Recovery_Time_Avg'] = finalList_reco
    finalDF['Recovery_Time_Avg'].fillna((finalDF['Recovery_Time_Avg'].mean()), inplace=True)

    # Can save file to a directory
    return finalDF


def aggregateEDA(eda_path):
    # Concatenate all subjects' EDA features together and then aggregate them
    EDA_full = pd.DataFrame()
    for subN in range(1, 68):
        df_prep = pd.read_excel(eda_path)
        df_prep['Sub'] = subN
        data_frames = [SCR_full, df_prep]
        SCR_full = pd.concat(data_frames, ignore_index=True)
    EDA_By_Segment = EDA_full.groupby(['Segment']).mean()

    # Should all be concatenated into one dataframe
    return EDA_By_Segment