from scipy.signal import find_peaks
import pandas as pd
from scipy.signal import butter, filtfilt
import numpy as np


def imuFilter(IMU_paths):
    # Go through each subject's IMU data
    for currSub in range(1, 68):

        # Read current IMU file
        dir = IMU_paths[currSub]
        acc_df = pd.read_excel(dir)

        # Filter information
        fs = 128
        cutoff = 4
        order = 4
        nyquist_freq = 0.5 * fs
        cutoff_norm = cutoff / nyquist_freq
        b, a = butter(order, cutoff_norm, btype='low')

        # Apply filter to data
        acc_cols = ['Acceleration X (m/s^2)', 'Acceleration Y (m/s^2)', 'Acceleration Z (m/s^2)']
        acc_data = acc_df[acc_cols].values

        # Try applying filter
        try:
            filtered_data = filtfilt(b, a, acc_data, axis=0)
        except ValueError:
            continue
        # Add filtered data to
        filt_cols = ['X_f', 'Y_f', 'Z_f']

        # Replace original acceleration data with filtered data
        acc_df[filt_cols] = filtered_data

        # Save filtered data to a new CSV file
    return acc_df


def strideTime(IMU_paths):
    # Go through each subject's IMU data
    for currSub in range(1, 68):
        stride_df = pd.DataFrame()
        dir = IMU_paths[currSub]
        acc_df = pd.read_excel(dir)
        imuSamplingRate = 128
        segmentList = []
        totalCount = 0

        # maxSegment is the last segment current subject walked through
        maxSegment = acc_df['SegmentNum'].max()
        stride_list = []

        # Go through each segment
        for curr_seg in range(0, maxSegment + 1):
            acc_df_SEGMENT = acc_df.loc[acc_df['SegmentNum'] == curr_seg]

            # X Y Z acc
            acc_norm = np.linalg.norm(acc_df_SEGMENT[['X_f', 'Y_f', 'Z_f']], axis=1)

            # Find the peaks in the acceleration signal
            peaks, _ = find_peaks(acc_norm, height=15)

            # Change to seconds instead of samples
            peaks = peaks / imuSamplingRate
            peaks = peaks[::2]
            stride_times = np.diff(peaks)

            # Average differences between the peaks
            avg_stride = np.average(stride_times)

            stride_list.append(avg_stride)

        stride_df['AvgStride'] = stride_list

        # Save dataframe to a file
        return stride_df