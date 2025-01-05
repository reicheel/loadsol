import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, savgol_filter
from matplotlib.widgets import SpanSelector
from pykalman import KalmanFilter
import matplotlib
matplotlib.use('Qt5Agg')  # Force TkAgg backend

BASE_DIR = r"C:\\filepath"

def get_participant_path(participant_id, subfolder="loadsol"):
    return os.path.join(BASE_DIR, participant_id, subfolder)

def load_data(file_path, sampling_rate=100):
    data = pd.read_csv(file_path, sep='\\s+', skiprows=3, header=0)
    if len(data.columns) > 4:
        data = data.iloc[:, :4]
    data.columns = ['Time_L', 'Force_L', 'Time_R', 'Force_R']
    data['Time'] = np.arange(len(data)) / sampling_rate
    return data

def manual_crop(data):
    while True:
        fig, ax = plt.subplots()
        ax.plot(data['Time'], data['Force_L'], label='Left Force', color='blue')
        ax.plot(data['Time'], data['Force_R'], label='Right Force', color='red')
        ax.set_title("Manual Cropping - Select Region")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (N)")
        plt.legend()

        crop_indices = []

        def onselect(xmin, xmax):
            start_idx = int(xmin * 100)  # Assuming 100 Hz sampling rate
            end_idx = int(xmax * 100)
            crop_indices.extend([start_idx, end_idx])
            plt.close()

        span = SpanSelector(ax, onselect, 'horizontal', useblit=True)
        print("Select the region to crop by dragging the mouse on the plot. Close the plot after selection.")
        plt.show()

        if len(crop_indices) == 2:
            cropped_data = data.iloc[crop_indices[0]:crop_indices[1]]
            plt.figure()
            plt.plot(cropped_data['Time'], cropped_data['Force_L'], label='Left Force', color='blue')
            plt.plot(cropped_data['Time'], cropped_data['Force_R'], label='Right Force', color='red')
            plt.title("Cropped Data - Confirm or Redo?")
            plt.xlabel("Time (s)")
            plt.ylabel("Force (N)")
            plt.legend()
            plt.show()

            confirm = input("Confirm cropping? (y/n): ").strip().lower()
            if confirm == 'y':
                return cropped_data
            else:
                print("Redo cropping.")
        else:
            print("No region selected. Please try again.")


def detect_peaks(data):
    peaks_L, _ = find_peaks(data['Force_L'], height=50, distance=50, prominence=0.1 * data['Force_L'].max())
    peaks_R, _ = find_peaks(data['Force_R'], height=50, distance=50, prominence=0.1 * data['Force_R'].max())

    plt.figure()
    plt.plot(data['Time'], data['Force_L'], label='Left Force', color='blue')
    plt.scatter(data['Time'].iloc[peaks_L], data['Force_L'][peaks_L], color='cyan', label='Detected Peaks (Left)')
    plt.plot(data['Time'], data['Force_R'], label='Right Force', color='red')
    plt.scatter(data['Time'].iloc[peaks_R], data['Force_R'][peaks_R], color='magenta', label='Detected Peaks (Right)')
    plt.title("Detected Peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.show()

    return peaks_L, peaks_R


def detect_phases(data, grf_threshold=50):
    phases = {}
    try:
        prep_start_idx = 0
        prep_end_idx = data[(data['Force_L'] <= 0) & (data['Force_R'] <= 0)].index[0] - 1
        if prep_end_idx > prep_start_idx:
            phases['Prep'] = (prep_start_idx, prep_end_idx)

        landing_start_idx = data[(data.index > prep_end_idx) & ((data['Force_L'] > grf_threshold) | (data['Force_R'] > grf_threshold))].index[0]
        landing_end_idx = data[(data.index > landing_start_idx) & ((data['Force_L'] < grf_threshold) & (data['Force_R'] < grf_threshold))].index[0]
        if landing_end_idx > landing_start_idx:
            phases['1st Landing'] = (landing_start_idx, landing_end_idx)

        jump_start_idx = data[(data.index > landing_end_idx) & ((data['Force_L'] < grf_threshold) & (data['Force_R'] < grf_threshold))].index[0]
        jump_end_idx = data[(data.index > jump_start_idx) & ((data['Force_L'] > grf_threshold) | (data['Force_R'] > grf_threshold))].index[0]
        if jump_end_idx > jump_start_idx:
            phases['Vertical Jump'] = (jump_start_idx, jump_end_idx)

        second_landing_start_idx = jump_end_idx
        second_landing_end_idx = len(data) - 1
        if second_landing_end_idx > second_landing_start_idx:
            phases['2nd Landing'] = (second_landing_start_idx, second_landing_end_idx)

    except IndexError:
        print("Incomplete phase detection due to missing data.")

    return phases


def save_trimmed_data(data, output_dir, trial_name):
    trimmed_file_name = f"{trial_name}_trimmed.csv"
    trimmed_file_path = os.path.join(output_dir, trimmed_file_name)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the trimmed data
    data.to_csv(trimmed_file_path, index=False)
    print(f"Trimmed data saved to: {trimmed_file_path}")


def calculate_metrics(data, force_threshold=20, grf_threshold=10):   
    results = {}

    # Filter data above threshold
    above_threshold_L = data[data['Force_L'] > force_threshold]
    above_threshold_R = data[data['Force_R'] > force_threshold]

    # Detect Peaks
    first_peak_L = above_threshold_L['Force_L'].idxmax() if not above_threshold_L.empty else None
    first_peak_R = above_threshold_R['Force_R'].idxmax() if not above_threshold_R.empty else None
    second_peak_L = (above_threshold_L.loc[first_peak_L + 1:]['Force_L'].idxmax() 
                     if first_peak_L is not None and not above_threshold_L.loc[first_peak_L + 1:].empty 
                     else None)
    second_peak_R = (above_threshold_R.loc[first_peak_R + 1:]['Force_R'].idxmax() 
                     if first_peak_R is not None and not above_threshold_R.loc[first_peak_R + 1:].empty 
                     else None)

    # Calculate Peak Times
    first_peak_time_L = data.loc[first_peak_L, 'Time_L'] if first_peak_L is not None else np.nan
    first_peak_time_R = data.loc[first_peak_R, 'Time_R'] if first_peak_R is not None else np.nan
    second_peak_time_L = data.loc[second_peak_L, 'Time_L'] if second_peak_L is not None else np.nan
    second_peak_time_R = data.loc[second_peak_R, 'Time_R'] if second_peak_R is not None else np.nan

    # Ground Leave Times
    ground_leave_time_L = data[data['Force_L'] <= force_threshold]['Time_L'].iloc[0] \
                          if not data[data['Force_L'] <= force_threshold].empty else np.nan
    ground_leave_time_R = data[data['Force_R'] <= force_threshold]['Time_R'].iloc[0] \
                          if not data[data['Force_R'] <= force_threshold].empty else np.nan

    # Impulse Calculation
    start_idx = data[data['Force_L'] > grf_threshold].index[0]
    end_idx = data[data['Force_L'] > grf_threshold].index[-1]
    impulse_L = trapezoid(data['Force_L'].iloc[start_idx:end_idx], data['Time'].iloc[start_idx:end_idx])
    impulse_R = trapezoid(data['Force_R'].iloc[start_idx:end_idx], data['Time'].iloc[start_idx:end_idx])

    # Peak Impact Force and Loading Rate
    landing_start_idx = data[(data.index >= start_idx) & 
                             ((data['Force_L'] > grf_threshold) | (data['Force_R'] > grf_threshold))].index[0]
    landing_end_idx = landing_start_idx + int(0.25 * (end_idx - start_idx))
    peak_force_L = data['Force_L'].iloc[landing_start_idx:landing_end_idx].max()
    peak_force_R = data['Force_R'].iloc[landing_start_idx:landing_end_idx].max()

    peak_force_time_L = data['Time'].iloc[landing_start_idx + np.argmax(data['Force_L'].iloc[landing_start_idx:landing_end_idx])]
    peak_force_time_R = data['Time'].iloc[landing_start_idx + np.argmax(data['Force_R'].iloc[landing_start_idx:landing_end_idx])]

    loading_rate_L = peak_force_L / (peak_force_time_L - data['Time'].iloc[landing_start_idx])
    loading_rate_R = peak_force_R / (peak_force_time_R - data['Time'].iloc[landing_start_idx])

    # Store Results
    results = {
        'First Peak Force L': data.loc[first_peak_L, 'Force_L'] if first_peak_L is not None else np.nan,
        'First Peak Force R': data.loc[first_peak_R, 'Force_R'] if first_peak_R is not None else np.nan,
        'First Peak Time L': first_peak_time_L,
        'First Peak Time R': first_peak_time_R,
        'Second Peak Force L': data.loc[second_peak_L, 'Force_L'] if second_peak_L is not None else np.nan,
        'Second Peak Force R': data.loc[second_peak_R, 'Force_R'] if second_peak_R is not None else np.nan,
        'Second Peak Time L': second_peak_time_L,
        'Second Peak Time R': second_peak_time_R,
        'Ground Leave Time L': ground_leave_time_L,
        'Ground Leave Time R': ground_leave_time_R,
        'Impulse L': impulse_L,
        'Impulse R': impulse_R,
        'Peak Impact Force L': peak_force_L,
        'Peak Impact Force R': peak_force_R,
        'Loading Rate L': loading_rate_L,
        'Loading Rate R': loading_rate_R
    }

    return results


def process_trials_safe(trial_files, output_dir):
    results = []
    
    for trial_file in trial_files:
        data = load_data(trial_file)
        cropped_data = manual_crop(data)
        metrics = calculate_metrics(cropped_data)
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_dir, "drop_vertical_jump_metrics.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Metrics saved to {results_file}")
    return results_df

def plot_phases(data, phases, output_dir, trial_name):
    phase_colors = {
        'Prep': 'mediumpurple',
        'Drop': 'lightgray',
        '1st Landing': 'lightgreen',
        'Vertical Jump': 'lightcoral',
        '2nd Landing': 'gold'
    }

    plt.figure()
    plt.plot(data['Time'], data['Force_L'], label='Left Force', color='blue')
    plt.plot(data['Time'], data['Force_R'], label='Right Force', color='red')

    for phase, color in phase_colors.items():
        if phases[phase] is not None:
            start_idx, end_idx = phases[phase]
            plt.axvspan(data['Time'].iloc[start_idx], data['Time'].iloc[end_idx], color=color, alpha=0.3, label=phase)

    plt.title(f"DVJ Phases - {trial_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend(loc='upper left')
    plot_file = os.path.join(output_dir, f"{trial_name}_phases.png")
    plt.savefig(plot_file)
    plt.show()
    print(f"phase plot saved to: {plot_file}")


def normalize_time(data, threshold=5):
    zero_force_idx = data[(data['Force_L'] < threshold) & (data['Force_R'] < threshold)].index[0]
    data['Time_Normalized'] = data['Time'] - data['Time'].iloc[zero_force_idx]
    return data


def plot_waveforms(trial_data, output_dir, aclr_limb='Left'):
    normalized_data = [normalize_time(trial) for trial in trial_data]
    normalized_time = np.linspace(
        min(trial['Time_Normalized'].min() for trial in normalized_data),
        max(trial['Time_Normalized'].max() for trial in normalized_data),
        1000
    )

    aclr_forces, contralateral_forces = [], []
    for trial in normalized_data:
        aclr_column = 'Force_L' if aclr_limb == 'Left' else 'Force_R'
        contra_column = 'Force_R' if aclr_limb == 'Left' else 'Force_L'

        aclr_interp = np.interp(normalized_time, trial['Time_Normalized'], trial[aclr_column])
        contra_interp = np.interp(normalized_time, trial['Time_Normalized'], trial[contra_column])

        aclr_forces.append(aclr_interp)
        contralateral_forces.append(contra_interp)

    # Compute average waveforms
    avg_aclr_force = np.mean(aclr_forces, axis=0)
    avg_contra_force = np.mean(contralateral_forces, axis=0)

    # Plot ACLR limb with average waveform
    plt.figure(figsize=(12, 6))
    for i, aclr_force in enumerate(aclr_forces):
        plt.plot(normalized_time, aclr_force, label=f'Trial {i+1}', linestyle='--', alpha=0.7)
    plt.plot(normalized_time, avg_aclr_force, label='Average', linewidth=2, color='black')
    plt.title(f'ACLR Limb ({aclr_limb}) SLH Waveform')
    plt.xlabel('Time (s) (Normalized)')
    plt.ylabel('Force (N)')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid()
    aclr_path = os.path.join(output_dir, f'ACLR_SLH_Waveform_{aclr_limb}.png')
    plt.savefig(aclr_path)
    plt.show()
    print(f"ACLR SLH waveform plot saved to: {aclr_path}")

    # Plot contralateral limb with average waveform
    plt.figure(figsize=(12, 6))
    for i, contra_force in enumerate(contralateral_forces):
        plt.plot(normalized_time, contra_force, label=f'Trial {i+1}', linestyle='--', alpha=0.7)
    plt.plot(normalized_time, avg_contra_force, label='Average', linewidth=2, color='black')
    plt.title(f"Contralateral Limb ({'Right' if aclr_limb == 'Left' else 'Left'}) SLH Waveform")
    plt.xlabel('Time (s) (Normalized)')
    plt.ylabel('Force (N)')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid()
    contra_path = os.path.join(output_dir, f"Contralateral_SLH_Waveform_{'Right' if aclr_limb == 'Left' else 'Left'}.png")
    plt.savefig(contra_path)
    plt.show()
    print(f"Contralateral SLH waveform plot saved to: {contra_path}")

    # Plot average waveforms for ACLR and Contralateral limbs on the same plot
    plt.figure(figsize=(12, 6))
    plt.plot(normalized_time, avg_aclr_force, label=f'Average SLH ACLR Limb ({aclr_limb})', linewidth=2, color='blue')
    plt.plot(normalized_time, avg_contra_force, label=f'Average SLH Contralateral Limb ({("Right" if aclr_limb == "Left" else "Left")})', linewidth=2, color='red')
    plt.title("Average SLH Waveforms - ACLR vs Contralateral")
    plt.xlabel("Time (s) (Normalized)")
    plt.ylabel("Force (N)")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid()
    combined_path = os.path.join(output_dir, "Avg SLH Waveforms.png")
    plt.savefig(combined_path)
    plt.show()
    print(f"Combined average SLH waveform plot saved to: {combined_path}")



################ Main Script #################
if __name__ == "__main__":
    # Participant-specific information
    participant_id = "subject-001"
    aclr_limb = "Right"  # Specify ACLR limb: "Left" or "Right"
    weight = 535  # Participant weight in Newtons

    # Input file information
    file_names = ["Dvj t1.txt", "Dvj t2.txt", "Dvj t3.txt", "Dvj t5.txt", "Dvj t5.txt"] #use renaming script to automate this from pdo files
    output_dir = get_participant_path(participant_id, "loadsol/outputs")

    # Initialize variables
    trial_files = [os.path.join(get_participant_path(participant_id), file) for file in file_names]
    metrics_results = []

    # Process each trial
    for file_name in file_names:
        file_path = os.path.join(get_participant_path(participant_id), file_name)
        print(f"Processing: {file_path}")

        # Load data
        data = load_data(file_path)

        # Manual crop
        cropped_data = manual_crop(data)
        cropped_data.reset_index(drop=True, inplace=True)

        # Save the cropped data
        save_trimmed_data(cropped_data, output_dir, os.path.splitext(file_name)[0])

        # Detect peaks on cropped and filtered data
        peaks_L, peaks_R = detect_peaks(cropped_data)

        # Detect and plot phases on cropped and filtered data
        dynamic_threshold = max(cropped_data['Force_L'].max(), cropped_data['Force_R'].max()) * 0.05
        phases = detect_phases(cropped_data, grf_threshold=max(dynamic_threshold, 50))

        # Ensure all phases are accounted for to avoid KeyError
        for phase in ['Prep', 'Drop', '1st Landing', 'Vertical Jump', '2nd Landing']:
            if phase not in phases:
                phases[phase] = None

        plot_phases(cropped_data, phases, output_dir, os.path.splitext(file_name)[0])

        # Calculate and save metrics
        metrics = calculate_metrics(cropped_data)
        metrics_results.append(metrics)

    # Save metrics results to CSV
    results_df = pd.DataFrame(metrics_results)
    metrics_file_path = os.path.join(output_dir, "drop_vertical_jump_metrics.csv")
    results_df.to_csv(metrics_file_path, index=False)
    print(f"Metrics saved to: {metrics_file_path}")

    # Plot  waveforms
    plot_waveforms(
        [pd.read_csv(os.path.join(output_dir, f"{os.path.splitext(file)[0]}_trimmed.csv")) for file in file_names],
        output_dir,
        aclr_limb=aclr_limb
    )

    # Print final metrics DataFrame
    print(results_df)





