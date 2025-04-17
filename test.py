from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_mysqldb import MySQL
import MySQLdb.cursors
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from scipy.signal import butter, filtfilt, find_peaks
import neurokit2 as nk
from scipy.signal import find_peaks
import os
import pandas as pd
from reportlab.lib.utils import ImageReader
import uuid
from datetime import datetime
from flask import make_response
import re
from functools import wraps
from flask import abort
import pytz
import plotly.graph_objects as go
import json
import plotly
import traceback
import time
from io import BytesIO
from xhtml2pdf import pisa
import matplotlib.pyplot as plt
import base64

app = Flask(__name__)

# Configuration
app.secret_key = "5010dfae019e413f06691431b2e3ba82bbb456c661b0d27332a4dbd5bbd36bd8"
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "452003@hrX"
app.config["MYSQL_DB"] = "hospital_ecg_db"
app.config["MYSQL_CURSORCLASS"] = "DictCursor"

mysql = MySQL(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)

DATASET_PATH = "mit-bih-arrhythmia-database-1.0.0/"
MODEL_PATH = os.path.join('model', 'ecg_arrhythmia_detector_20250331_165229.h5')

CLASSES = {
    0: {'id': 'N', 'name': 'Normal', 'weight': 1, 'color': '#2ecc71'},
    1: {'id': 'S', 'name': 'SVT', 'weight': 100, 'color': '#e67e22'},
    2: {'id': 'AF', 'name': 'Atrial Fibrillation', 'weight': 150, 'color': '#e74c3c'},
    3: {'id': 'VF', 'name': 'Ventricular Fibrillation', 'weight': 200, 'color': '#9b59b6'},
    4: {'id': 'VT', 'name': 'Ventricular Tachycardia', 'weight': 170, 'color': '#c0392b'},
    5: {'id': 'B', 'name': 'Heart Block', 'weight': 120, 'color': '#3498db'},
    6: {'id': 'F', 'name': 'Fusion', 'weight': 80, 'color': '#a67c52'}
}

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
    
    # Try a dummy prediction to verify the model works
    dummy_input = np.random.rand(1, 180, 1)
    dummy_pred = model.predict(dummy_input)
    print("Model test prediction successful")
    
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model: {str(e)}")
    print(traceback.format_exc())
    # In production, you might want to exit here
    # import sys; sys.exit(1)
    model = None


class User(UserMixin):
    def __init__(self, id, username=None, user_type=None):
        self.id = id
        self.username = username
        self.user_type = user_type
    
    def get_type(self):
        return self.user_type

@login_manager.user_loader
def load_user(user_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    try:
        if user_id.startswith('DR-'):
            cursor.execute("SELECT * FROM doctor WHERE Doctor_ID = %s", (user_id,))
            doctor = cursor.fetchone()
            if doctor:
                return User(
                    id=str(doctor["Doctor_ID"]),
                    username=doctor["Username"],
                    user_type="doctor"
                )
        else:
            cursor.execute("SELECT * FROM staff WHERE Staff_ID = %s", (user_id,))
            staff = cursor.fetchone()
            if staff:
                return User(
                    id=staff["Staff_ID"],
                    username=staff["StaffName"],
                    user_type="staff"
                )
    except MySQLdb.Error as e:
        print(f"Database error: {e}")
    finally:
        cursor.close()
    return None

def doctor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.user_type != "doctor":
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

def load_ecg_sample(record_num):
    try:
        record_path = f"{DATASET_PATH}{record_num}"
        # record_path = "E:/mindteck project/mit-bih-arrhythmia-database-1.0.0/100"
        print("record: ",record_path)
        record = wfdb.rdrecord(record_path)
        print(f"Attempting to load record from: {record_path}") # Debug print

        # Check if the necessary files exist (.dat, .hea, .atr)
        if not os.path.exists(f"{record_path}.dat"):
             raise FileNotFoundError(f"ECG data file not found: {record_path}.dat")
        if not os.path.exists(f"{record_path}.hea"):
             raise FileNotFoundError(f"ECG header file not found: {record_path}.hea")
        # Annotation file might be optional depending on your needs, but good to check
        # if not os.path.exists(f"{record_path}.atr"):
        #      print(f"Warning: Annotation file not found: {record_path}.atr")

        record = wfdb.rdrecord(record_path)
        # Only try to read annotations if the file exists or is expected
        try:
            annotation = wfdb.rdann(record_path, 'atr')
            symbols = annotation.symbol
            samples = annotation.sample
        except FileNotFoundError:
            print(f"Annotation file (.atr) not found for record {record_num}, proceeding without annotations.")
            symbols = []
            samples = []
        except Exception as e_ann:
            print(f"Error reading annotation file for {record_num}: {e_ann}")
            symbols = []
            samples = []


        # Ensure p_signal is not None and has columns
        if record.p_signal is None or record.p_signal.shape[1] == 0:
            raise ValueError(f"No signal data found in record {record_num}")

        # Return the first channel (index 0), fs, symbols, and samples
        return record.p_signal[:, 0], record.fs, symbols, samples
    
    except FileNotFoundError as fnf_err:
        print(f"Error loading ECG sample {record_num}: {fnf_err}")
        flash(f"Error: Could not find ECG record files for '{record_num}'. Please check the record number and data path.", "danger")
        return None, None, None, None # Return None for all values on error
    except Exception as e:
        print(f"Error loading ECG sample {record_num}: {e}")
        flash(f"Error loading ECG record '{record_num}': {e}", "danger")
        return None, None, None, None # Return None for all values on error

def check_ecg_quality(ecg_signal, fs):
    """Basic ECG signal quality checks"""
    if len(ecg_signal) < fs:  # Less than 1 second of data
        return False, "Signal too short"
    
    if np.max(ecg_signal) - np.min(ecg_signal) < 0.1:  # Flat line
        return False, "Signal amplitude too low"
    
    return True, "Signal quality OK"

def butterworth_filter(signal, cutoff=50, fs=360, order=4):
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)
    except Exception as e:
        print(f"Error applying Butterworth filter: {e}")
        return signal
    
def detect_r_peaks(ecg_signal, fs):
    try:
        # Clean the ECG signal using NeuroKit2
        processed_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="pantompkins1985")
        
        # Detect R-peaks using a custom threshold
        r_peaks, _ = find_peaks(
            processed_ecg,
            height=np.percentile(processed_ecg, 75),  # Adjust threshold for peak height
            distance=int(0.6 * fs)  # Minimum distance between peaks (600ms)
        )
        
        print(f"Detected R-peaks: {r_peaks}")
        return np.array(r_peaks)
    except Exception as e:
        print(f"Error detecting R-peaks: {e}")
        return np.array([])  # Return an empty array if detection fails
# Return an empty array instead of None

def align_ecg_to_r_peak(ecg_signal, r_peaks, target_length=180):
    """Center the ECG segment around the most prominent R-peak"""
    if len(r_peaks) == 0:
        return center_pad_ecg(ecg_signal, target_length)
    
    # Find the most prominent R-peak (largest amplitude)
    main_r_peak = r_peaks[np.argmax([ecg_signal[i] for i in r_peaks])]
    
    # Calculate start and end indices
    start = max(0, main_r_peak - target_length // 2)
    end = start + target_length
    
    if end > len(ecg_signal):
        end = len(ecg_signal)
        start = end - target_length
    
    segment = ecg_signal[start:end]
    
    # Pad if necessary
    if len(segment) < target_length:
        pad_before = (target_length - len(segment)) // 2
        pad_after = target_length - len(segment) - pad_before
        segment = np.pad(segment, (pad_before, pad_after), mode='reflect')
    
    return segment

def center_pad_ecg(ecg_signal, target_length):
    """Center-pad ECG signal with reflection padding"""
    if len(ecg_signal) >= target_length:
        return ecg_signal[:target_length]
    
    pad_before = (target_length - len(ecg_signal)) // 2
    pad_after = target_length - len(ecg_signal) - pad_before
    return np.pad(ecg_signal, (pad_before, pad_after), mode='reflect')

def check_heart_rate(heart_rate):
   
    normal_min = 60
    normal_max = 100

    if heart_rate < normal_min:
        return "Bradycardia (slow heart rate)"
    elif heart_rate > normal_max:
        return "Tachycardia (fast heart rate)"
    else:
        return "Normal heart rate"



def preprocess_ecg(ecg_signal, target_length=180):
    print("PreProcessing Function !!!")
    """Prepare ECG signal for model input with consistent shape."""
    try:
        # Ensure input is numpy array
        ecg_signal = np.array(ecg_signal, dtype=np.float32)
        
        # Normalize
        ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
        
        # Handle length
        if len(ecg_signal) > target_length:
            ecg_signal = ecg_signal[:target_length]
        else:
            ecg_signal = np.pad(ecg_signal, (0, target_length - len(ecg_signal)), 
                            mode='constant', constant_values=0)
        
        # Reshape to (1, 180, 1) - batch of 1, 180 timesteps, 1 channel
        return np.reshape(ecg_signal, (1, target_length, 1))
    except Exception as e:
        print(f"Error preprocessing ECG: {e}")
        # Return zero array with correct shape if error occurs
        return np.zeros((1, target_length, 1), dtype=np.float32)
    

def downsample_ecg(ecg_signal, target_samples=5000):
    """
    Downsample ECG signal to target_samples while preserving key features
    """
    original_length = len(ecg_signal)
    if original_length <= target_samples:
        return ecg_signal
    
    # Calculate downsampling factor
    factor = int(np.ceil(original_length / target_samples))
    
    # Downsample using mean to preserve general shape
    downsampled = np.zeros(target_samples)
    for i in range(target_samples):
        start = i * factor
        end = min((i + 1) * factor, original_length)
        downsampled[i] = np.mean(ecg_signal[start:end])
    
    return downsampled

def compute_intervals(ecg_signal, r_peaks, fs):
    try:
        if len(r_peaks) < 2:
            return 0, 0, 0, np.array([])
            
        rr_intervals = np.diff(r_peaks) / fs
        heart_rate = 60 / np.mean(rr_intervals)
        
        # Find QRS peaks (using relative height)
        qrs_peaks, _ = find_peaks(ecg_signal, 
                                 height=np.percentile(ecg_signal, 90),
                                 distance=int(fs*0.06))
        
        # Calculate intervals
        qt_interval = (r_peaks[-1] - r_peaks[0]) / fs if len(r_peaks) > 1 else 0
        pr_interval = (r_peaks[1] - r_peaks[0]) / fs if len(r_peaks) > 1 else 0
        
        return float(heart_rate), float(qt_interval), float(pr_interval), qrs_peaks
    except Exception as e:
        print(f"Error computing intervals: {e}")
        return 0.0, 0.0, 0.0, np.array([])

def compute_framingham_risk(age, cholesterol, hdl, systolic_bp, smoker, diabetes):
    try:
        score = (0.02 * age) + (0.03 * cholesterol) - (0.05 * hdl) + (0.04 * systolic_bp) + (0.2 * smoker) + (0.15 * diabetes)
        return min(max(score, 0), 30)
    except Exception as e:
        print(f"Error computing Framingham risk: {e}")
        return 0

def compute_grace_score(age, systolic_bp, heart_rate):
    try:
        score = (0.1 * age) - (0.05 * systolic_bp) + (0.2 * heart_rate)
        return min(max(score, 0), 20)
    except Exception as e:
        print(f"Error computing GRACE score: {e}")
        return 0
    

# def detect_r_peaks(ecg_signal, fs):
#     try:
#         # Clean the ECG signal using NeuroKit2
#         processed_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs)
#         _, r_peaks = nk.ecg_peaks(processed_ecg, sampling_rate=fs)
#         return np.array(r_peaks["ECG_R_Peaks"])
#     except Exception as e:
#         print(f"Error detecting R-peaks: {e}")
#         return np.array([])



#everything proper but need some update in grid lines
def generate_ecg_plot(ecg_signal, r_peaks=None, title="ECG Signal", pred_class_id=None, fs=360, samples_to_show=2000):
    # Convert inputs to numpy arrays
    ecg_signal = np.array(ecg_signal)
    if r_peaks is not None:
        r_peaks = np.array(r_peaks)
    
    # Ensure we don't exceed available samples
    samples_to_show = min(samples_to_show, len(ecg_signal))
    
    # DEBUG: Print signal information
    print(f"ECG signal length: {len(ecg_signal)} samples")
    print(f"First 5 values: {ecg_signal[:5]}")
    
    # Create time axis in seconds
    time_axis = np.linspace(0, samples_to_show/fs, samples_to_show)
    
    # DEBUG: Print time axis information
    print(f"Time axis range: {time_axis[0]:.2f}s to {time_axis[-1]:.2f}s")
    print(f"Time axis length: {len(time_axis)} points")
    
    # Create figure
    fig = go.Figure()
    
    # Plot ECG signal
    fig.add_trace(go.Scatter(
        x=time_axis.tolist(),
        y=ecg_signal[:samples_to_show].tolist(),
        mode='lines',
        line=dict(color='#000000', width=1.5),
        name='ECG Signal',
        hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.2f}mV'
    ))
    
    if r_peaks is not None and len(r_peaks) > 0:
        visible_r_peaks = r_peaks[r_peaks < samples_to_show]
        print(f"Visible R-peaks: {visible_r_peaks}")
        
        class_info = next((v for k, v in CLASSES.items() if v.get('id') == pred_class_id), 
                       {'name': 'Unknown', 'color': '#95a5a6'})
        
        # PRECISE R-PEAK DETECTION - Using windowed refinement
        refined_peaks = []
        window_size = int(0.08 * fs)  # 80ms window around each peak
        
        for peak in visible_r_peaks:
            start = max(0, peak - window_size)
            end = min(len(ecg_signal), peak + window_size)
            window = ecg_signal[start:end]
            exact_peak = start + np.argmax(window)  # Find exact maximum in window
            refined_peaks.append(exact_peak)
        
        refined_peaks = np.array(refined_peaks)
        print(f"Refined peak positions: {refined_peaks}")
        
        # Convert peak positions to time in seconds
        peak_times = refined_peaks / fs
        peak_values = ecg_signal[refined_peaks]
        
        fig.add_trace(go.Scatter(
            x=peak_times.tolist(),
            y=peak_values.tolist(),
            mode='markers',
            marker=dict(
                color=class_info['color'],
                size=10,  # Slightly larger for better visibility
                line=dict(width=2, color='#333333'),
                symbol='diamond'
            ),
            name=f'R-peaks ({len(refined_peaks)} beats)',
            hovertemplate='R-peak at %{x:.3f}s<br>Amplitude: %{y:.2f}mV'
        ))
    
    # Update layout with ECG-standard grid and styling
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title='Time (s)',
        yaxis_title='Amplitude (mV)',
        height=450,
        margin=dict(l=60, r=40, t=80, b=60),
        plot_bgcolor='rgba(255, 235, 235, 0.5)',
        paper_bgcolor='white',
        hovermode='closest',
        showlegend=True,
        xaxis=dict(
            gridcolor='rgba(255, 99, 99, 0.5)',
            showgrid=True,
            gridwidth=1,
            dtick=0.2,  # Major grid every 0.2s (5mm)
            minor=dict(
                showgrid=True,
                gridcolor='rgba(255, 99, 99, 0.2)',
                gridwidth=0.5,
                dtick=0.04  # Minor grid every 0.04s (1mm)
            ),
            rangeslider=dict(visible=True),
            tickformat='.2f'
        ),
        yaxis=dict(
            gridcolor='rgba(255, 99, 99, 0.2)',
            showgrid=True,
            gridwidth=1,
            dtick=0.5,  # Major grid every 0.5mV (5mm)
            minor=dict(
                showgrid=True,
                gridcolor='rgba(255, 99, 99, 0.2)',
                gridwidth=0.5,
                dtick=0.1  # Minor grid every 0.1mV (1mm)
            )
        )
    )
    
    return f"ecg_plot_{int(time.time())}.html", json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# #time axis working fine
# def generate_ecg_plot(ecg_signal, r_peaks=None, title="ECG Signal", pred_class_id=None, fs=360, samples_to_show=2000):
#     # Convert inputs to numpy arrays
#     ecg_signal = np.array(ecg_signal)
#     if r_peaks is not None:
#         r_peaks = np.array(r_peaks)
    
#     # Ensure we don't exceed available samples
#     samples_to_show = min(samples_to_show, len(ecg_signal))
    
#     # DEBUG: Print signal information
#     print(f"ECG signal length: {len(ecg_signal)} samples")
#     print(f"First 5 values: {ecg_signal[:5]}")
    
#     # Create time axis in seconds
#     time_axis = np.linspace(0, samples_to_show/fs, samples_to_show)
    
#     # DEBUG: Print time axis information
#     print(f"Time axis range: {time_axis[0]:.2f}s to {time_axis[-1]:.2f}s")
#     print(f"Time axis length: {len(time_axis)} points")
    
#     # Create figure
#     fig = go.Figure()
    
#     # Plot ECG signal
#     fig.add_trace(go.Scatter(
#         x=time_axis.tolist(),  # Convert to list for Plotly
#         y=ecg_signal[:samples_to_show].tolist(),
#         mode='lines',
#         line=dict(color='#000000', width=1.5),  # Using your preferred black color
#         name='ECG Signal',
#         hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.2f}mV'
#     ))
    
#     if r_peaks is not None and len(r_peaks) > 0:
#         visible_r_peaks = r_peaks[r_peaks < samples_to_show]
#         print(f"Visible R-peaks: {visible_r_peaks}")
        
#         class_info = next((v for k, v in CLASSES.items() if v.get('id') == pred_class_id), 
#                        {'name': 'Unknown', 'color': '#95a5a6'})  # Using your preferred color
        
#         # Convert peak positions to time in seconds
#         peak_times = (visible_r_peaks / fs).tolist()
#         peak_values = ecg_signal[visible_r_peaks].tolist()
        
#         fig.add_trace(go.Scatter(
#             x=peak_times,
#             y=peak_values,
#             mode='markers',
#             marker=dict(
#                 color=class_info['color'],
#                 size=8,
#                 line=dict(width=2, color='#333333'),  # Using your preferred marker style
#                 symbol='diamond'
#             ),
#             name=f'R-peaks ({len(visible_r_peaks)} beats)'
#         ))
    
#     # Update layout with proper indentation
#     fig.update_layout(
#         title=dict(text=title, x=0.5, font=dict(size=18)),
#         xaxis_title='Time (s)',
#         yaxis_title='Amplitude (mV)',
#         height=450,
#         margin=dict(l=60, r=40, t=80, b=60),
#         plot_bgcolor='rgba(255, 235, 235, 0.5)',  # Your preferred background
#         paper_bgcolor='white',
#         hovermode='closest',
#         showlegend=True,
#         xaxis=dict(
#             gridcolor='rgba(255, 99, 99, 0.5)',  # Major gridlines
#             showgrid=True,
#             gridwidth=1,
#             dtick=0.2,  # In seconds now, not samples
#             minor=dict(
#                 showgrid=True,
#                 gridcolor='rgba(255, 99, 99, 0.2)',  # Minor gridlines
#                 gridwidth=0.5,
#                 dtick=0.04  # In seconds now, not samples
#             ),
#             rangeslider=dict(visible=True),
#             tickformat='.2f'
#         ),
#         yaxis=dict(
#             gridcolor='rgba(255, 99, 99, 0.2)',  # Major gridlines
#             showgrid=True,
#             gridwidth=1,
#             dtick=0.5,  # Major grid every 0.5mV
#             minor=dict(
#                 showgrid=True,
#                 gridcolor='rgba(255, 99, 99, 0.2)',  # Minor gridlines
#                 gridwidth=0.5,
#                 dtick=0.1  # Minor grid every 0.1mV
#             )
#         )
#     )
    
#     return f"ecg_plot_{int(time.time())}.html", json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

#working properlybut time axis not in seconds
# def generate_ecg_plot(ecg_signal, r_peaks=None, title="ECG Signal", pred_class_id=None, fs=360, samples_to_show=2000):
    
#     # Convert inputs to numpy arrays
#     ecg_signal = np.array(ecg_signal)
#     if r_peaks is not None:
#         r_peaks = np.array(r_peaks)
    
#     # Create figure
#     fig = go.Figure()
    
#     # Add ECG trace with black line
#     fig.add_trace(go.Scatter(
#         x=list(range(samples_to_show)),
#         y=ecg_signal[:samples_to_show].tolist(),
#         mode='lines',
#         line=dict(color='#000000', width=1.5),
#         name='ECG Signal'
#     ))
    
#     # Highlight R-peaks if available
#     if r_peaks is not None and len(r_peaks) > 0:
#         visible_r_peaks = r_peaks[r_peaks < samples_to_show]
#         class_info = next((v for k, v in CLASSES.items() if v.get('id') == pred_class_id), 
#                          {'name': 'Unknown', 'color': '#95a5a6'})
        
        
#         refined_peaks = []
#         for peak in visible_r_peaks:
#             start = max(0, peak - 5)
#             end = min(len(ecg_signal), peak + 5)
#             window = ecg_signal[start:end]
#             exact_peak = start + np.argmax(window)
#             refined_peaks.append(exact_peak)
        
#         refined_peaks = np.array(refined_peaks)
        
#         fig.add_trace(go.Scatter(
#             x=refined_peaks.tolist(),
#             y=ecg_signal[refined_peaks].tolist(),
#             mode='markers',
#             marker=dict(
#                 color=class_info['color'],
#                 size=8,
#                 line=dict(width=2, color='#333333'),
#                 symbol='diamond'
#             ),
#             name=f'R-peaks ({len(refined_peaks)} beats)'
#         ))
    
    
#     fig.update_layout(
#         title=dict(text=title, x=0.5, font=dict(size=18)),
#         xaxis_title='Time (s)',
#         yaxis_title='Amplitude (mV)',
#         height=450,
#         margin=dict(l=60, r=40, t=80, b=60),
#         plot_bgcolor='rgba(255, 235, 235, 0.5)',
#         paper_bgcolor='white',
#         hovermode='closest',
#         showlegend=True,
#         xaxis=dict(
#             gridcolor='rgba(255, 99, 99, 0.5)',  # Major gridlines
#             showgrid=True,
#             gridwidth=1,
#             dtick=fs*0.2,  # Major grid every 0.2s (5mm at 25mm/s)
#             minor=dict(
#                 showgrid=True,
#                 gridcolor='rgba(255, 99, 99, 0.2)',  # Minor gridlines
#                 gridwidth=0.5,
#                 dtick=fs*0.04  # Minor grid every 0.04s (1mm)
#             ),
#             rangeslider=dict(visible=True),
#             tickformat='.2f'  # Show 2 decimal places for time
#         ),
#         yaxis=dict(
#             gridcolor='rgba(255, 99, 99, 0.2)',  # Major gridlines
#             showgrid=True,
#             gridwidth=1,
#             dtick=0.5,  # Major grid every 0.5mV (5mm at 10mm/mV)
#             minor=dict(
#                 showgrid=True,
#                 gridcolor='rgba(255, 99, 99, 0.2)',  # Minor gridlines
#                 gridwidth=0.5,
#                 dtick=0.1  # Minor grid every 0.1mV (1mm)
#             )
#         )
#     )
    
#     # Convert x-axis from samples to seconds
#     fig.update_xaxes(tickvals=np.arange(0, samples_to_show, fs),
#                     ticktext=[f"{x/fs:.2f}" for x in np.arange(0, samples_to_show, fs)])
    
#     return f"ecg_plot_{int(time.time())}.html", json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_pdf(predicted_class, framingham_risk, grace_score, heart_rate, 
                qt_interval, pr_interval, vfib_detected=None, 
                vfib_confidence=None, ecg_graph_json=None):
    try:
        unique_id = str(uuid.uuid4())[:8]
        pdf_filename = f"ECG_Report_{unique_id}.pdf"
        pdf_path = os.path.join("static", pdf_filename)
        
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "ECG Analysis Report")
        c.setFont("Helvetica", 12)
        
        y_position = 700
        c.drawString(100, y_position, f"Prediction: {predicted_class}")
        y_position -= 30
        c.drawString(100, y_position, f"Framingham Risk: {framingham_risk:.2f}%")
        y_position -= 30
        c.drawString(100, y_position, f"GRACE Score: {grace_score:.2f}%")
        y_position -= 30
        c.drawString(100, y_position, f"Heart Rate: {heart_rate:.2f} BPM")
        
        if vfib_detected is not None:
            y_position -= 30
            status = "Detected" if vfib_detected else "Not Detected"
            c.drawString(100, y_position, f"VFib: {status} ({vfib_confidence:.1f}% confidence)")
        
        y_position -= 30
        c.drawString(100, y_position, f"QT Interval: {qt_interval:.3f}s")
        y_position -= 30
        c.drawString(100, y_position, f"PR Interval: {pr_interval:.3f}s")
        
        if ecg_graph_json:
            try:
                # Save plot as image
                plot_filename = f"ecg_plot_{unique_id}.png"
                plot_path = os.path.join("static", plot_filename)
                
                fig = plotly.io.from_json(ecg_graph_json)
                fig.write_image(plot_path)
                
                # Add to PDF
                c.drawImage(plot_path, 100, 400, width=400, height=200)
            except Exception as e:
                print(f"Error adding ECG plot to PDF: {e}")
        
        c.save()
        return pdf_filename
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None

# Routes
@app.route("/", methods=["GET", "POST"])
def staff_login():
    if request.method == "POST":
        staff_id = request.form.get("Staff_ID")
        password = request.form.get("password")
        
        if not staff_id or not password:
            flash("Please enter Staff ID and password", "danger")
            return render_template("staff_login.html")
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        try:
            cursor.execute("SELECT * FROM staff WHERE Staff_ID = %s", (staff_id,))
            staff = cursor.fetchone()
            
            if staff and bcrypt.check_password_hash(staff["Password"], password):
                staff_obj = User(
                    id=staff["Staff_ID"],  
                    username=staff["StaffName"],
                    user_type="staff"
                )
                login_user(staff_obj)
                return redirect(url_for("patient_registration"))
            else:
                flash("Invalid credentials", "danger")
        except MySQLdb.Error as e:
            flash(f"Database error: {str(e)}", "danger")
        finally:
            cursor.close()
    
    return render_template("staff_login.html")

@app.route("/doctor_login", methods=["GET", "POST"])
def doctor_login():
    if request.method == "POST":
        doctor_id = request.form.get("doctor_id")
        password = request.form.get("password")
        
        if not doctor_id or not password:
            return render_template("doctor_login.html")
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        try:
            cursor.execute("SELECT * FROM doctor WHERE Doctor_ID = %s", (doctor_id,))
            doctor = cursor.fetchone()
            
            if doctor and bcrypt.check_password_hash(doctor["Password"], password):
                doctor_obj = User(
                    id=str(doctor["Doctor_ID"]),
                    username=doctor['Username'],
                    user_type="doctor"
                )
                session['doctor_name'] = doctor['Username']
                login_user(doctor_obj)
                return redirect(url_for("input_form"))
            else:
                flash("Invalid credentials", "danger")
        except MySQLdb.Error as e:
            flash(f"Database error: {str(e)}", "danger")
        finally:
            cursor.close()
    
    return render_template("doctor_login.html")

@app.route("/input_form", methods=["GET", "POST"])
@login_required
def input_form():
    doctor_name = session.get('doctor_name', 'Guest')
    patient = None
    medical_history = None
    
    if request.method == "POST":
        patient_id = request.form.get("Patient_ID")
        
        if not patient_id:
            flash("Please enter a patient ID", "danger")
            return render_template("input_form.html")
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        try:
            cursor.execute("""
                SELECT p.*, s.StaffName 
                FROM patient_profile p
                LEFT JOIN staff s ON p.Staff_Username = s.Staff_ID
                WHERE p.Patient_ID = %s
            """, (patient_id,))
            patient = cursor.fetchone()
            
            if not patient:
                flash("Patient not found", "danger")
                return render_template("input_form.html")
            
            cursor.execute("""
                SELECT * FROM input 
                WHERE Patient_ID = %s 
                ORDER BY Generated_AT DESC
            """, (patient_id,))
            medical_history = cursor.fetchall()
            
        except MySQLdb.Error as e:
            flash(f"Database error: {str(e)}", "danger")
        finally:
            cursor.close()

    return render_template("input_form.html", 
                         patient=patient, 
                         medical_history=medical_history, 
                         doctor_name=doctor_name)

@app.route("/add_medical_data/<patient_id>", methods=["POST"])
@login_required
def add_medical_data(patient_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    try:
        cursor.execute("SELECT * FROM patient_profile WHERE Patient_ID = %s", (patient_id,))
        patient = cursor.fetchone()
        
        if not patient:
            flash("Patient not found", "danger")
            return redirect(url_for("input_form"))
        
        systolic_bp = request.form["systolic_bp"]
        cholesterol = request.form["cholesterol"]
        hdl = request.form["hdl"]
        smoker = 1 if request.form.get("smoker") else 0
        diabetic = 1 if request.form.get("diabetic") else 0
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute("""
            INSERT INTO input 
            (Patient_ID, Doctor_ID, Smoker, Alcoholic, Diabetic, Cholesterol, HDL, 
             Blood_Pressure, Other_Issues, Generated_AT)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            patient_id, 
            current_user.id, 
            smoker, 
            0,  
            diabetic, 
            cholesterol, 
            hdl,
            systolic_bp, 
            "",  
            generated_at
        ))
        mysql.connection.commit()
        # flash("Medical data saved successfully", "success")
        
    except MySQLdb.Error as e:
        flash(f"Error saving medical data: {str(e)}", "danger")
    finally:
        cursor.close()
    
    return redirect(url_for("input_form"))

@app.route("/logout")
@login_required
def logout():
    user_type = current_user.user_type
    logout_user()
    
    if user_type == "doctor":
        return redirect(url_for("doctor_login"))
    elif user_type == "staff":
        return redirect(url_for("staff_login"))
    else:
        return redirect(url_for("staff_login"))

def handle_model_output(predictions):
    try:
        if isinstance(predictions, list) and len(predictions) > 0:
            return np.array(predictions[0][0][:7])
        return np.array(predictions[0][0][:7])
    except Exception as e:
        print(f"Error processing output: {str(e)}")
        return None
    
@app.route("/automatic_analysis/<patient_id>", methods=["GET", "POST"])
@login_required
def automatic_analysis(patient_id):
    print("Testing!!!!")
    """Perform ECG analysis with the multi-output model."""
    try:
        print("Try Catch - 1")
        # Validate patient ID
        if not re.match(r'^PT-\d{5}-\d{4}$', patient_id):
            flash("Invalid Patient ID format", "danger")
            return redirect(url_for("input_form"))

        # Fetch patient data
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        try:
            print("Try - Catch 2")
            cursor.execute("""
                SELECT p.*, d.Username AS Doctor_Name 
                FROM patient_profile p
                LEFT JOIN doctor d ON p.Doctor_ID = d.Doctor_ID
                WHERE p.Patient_ID = %s
            """, (patient_id,))
            patient = cursor.fetchone()
            
            if not patient:
                flash("Patient not found", "danger")
                return redirect(url_for("input_form"))
                
        except Exception as e:
            flash(f"Database error: {str(e)}", "danger")
            return redirect(url_for("input_form"))
        finally:
            cursor.close()

        if request.method == "POST":
            try:
                print("Try Catch 3")
                # Get and validate form data
                record_num = request.form.get("record_num", "100").strip()
                age = int(request.form.get("age", patient.get("Age", 30)))
                cholesterol = int(request.form.get("cholesterol", 150))
                hdl = int(request.form.get("hdl", 40))
                systolic_bp = int(request.form.get("systolic_bp", 120))
                smoker = 1 if request.form.get("smoker") == "on" else 0
                diabetes = 1 if request.form.get("diabetes") == "on" else 0

                # Validate medical parameters
                if not (0 < age <= 120):
                    raise ValueError("Age must be between 1-120")
                if not (0 < cholesterol <= 500):
                    raise ValueError("Invalid cholesterol value")
                if not (0 < hdl <= 100):
                    raise ValueError("Invalid HDL value")
                if not (50 <= systolic_bp <= 250):
                    raise ValueError("Invalid systolic BP")
                print("Before Loading ECG Sample")


                insert_cursor = mysql.connection.cursor()
                try:
                    insert_query = """
                        INSERT INTO input 
                        (Patient_ID, Smoker, Diabetic, Cholesterol, HDL, 
                         Blood_Pressure, Generated_AT, Doctor_ID)
                        VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s)
                    """
                    insert_cursor.execute(insert_query, (
                        patient_id, 
                        smoker, 
                        diabetes, 
                        cholesterol, 
                        hdl, 
                        systolic_bp,
                        patient.get('Doctor_ID')
                    ))
                    mysql.connection.commit()
                    
                    # Verify insertion
                    insert_cursor.execute("""
                        SELECT * FROM input 
                        WHERE Patient_ID = %s
                        ORDER BY Record_ID DESC
                        LIMIT 1
                    """, (patient_id,))
                    latest_record = insert_cursor.fetchone()
                    
                    if not latest_record:
                        raise Exception("Insert verification failed - no record found")

                except Exception as e:
                    mysql.connection.rollback()
                    flash(f"Failed to save medical data: {str(e)}", "danger")
                    return redirect(url_for("automatic_analysis", patient_id=patient_id))
                finally:
                    insert_cursor.close()
                # Load and validate ECG signal
                #print("Return of Load Sample: ",load_ecg_sample(record_num))
                ecg_signal, fs, _, _ = load_ecg_sample(record_num)
                if ecg_signal is None or fs is None:
                 # Flash message is handled inside load_ecg_sample now
                    return redirect(url_for("automatic_analysis", patient_id=patient_id)) 
                if len(ecg_signal) < 180: # Or some reasonable minimum length
                    flash(f"ECG signal for record '{record_num}' is too short.", "warning")
                print("ECG Signal FS: ",fs)
                if ecg_signal is None:
                    raise ValueError("Failed to load ECG record")
                if len(ecg_signal) < 180:
                    raise ValueError("ECG signal too short (min 180 samples)")

                # Process ECG signal
                ecg_filtered = butterworth_filter(ecg_signal, fs=fs)
                print("ECG Filtered: ",ecg_filtered)
                ecg_normalized = (ecg_filtered - np.mean(ecg_filtered)) / np.std(ecg_filtered)
                print("ECG Normalized 1: ",ecg_normalized)
                ecg_normalized = np.nan_to_num(ecg_normalized, nan=0.0)
                print("ECG Normalized 2: ",ecg_normalized)

                # Detect cardiac features
                r_peaks = detect_r_peaks(ecg_filtered, fs)
                print("R Peaks: ",r_peaks)
                print("Compute Intervals",compute_intervals(ecg_filtered, r_peaks, fs))
                heart_rate, qt_interval, pr_interval, _ = compute_intervals(ecg_filtered, r_peaks, fs)
                print("Heart Rate: ",heart_rate)
                print("QT Interval: ",qt_interval)
                print("PR Interval: ",pr_interval)


                heart_rate_status = check_heart_rate(heart_rate)
                print("Heart Rate Status: ",heart_rate_status)
                # Prepare model input (ensure exact 180 samples)
                model_input = np.zeros((1, 180, 1), dtype=np.float32)
                print("Model Input: ",model_input)
                usable_samples = min(180, len(ecg_normalized))
                print("Usable Samples: ",usable_samples)
                model_input[0, :usable_samples, 0] = ecg_normalized[:usable_samples]


                # Make prediction
                main_pred, vfib_pred = model.predict(model_input, verbose=0)
                print("Main Prediction: ",main_pred)
                print("VFIB Prediction: ",vfib_pred)
                main_pred = main_pred.flatten()
                print("Main Prediction Flattened: ",main_pred)
                vfib_pred = vfib_pred.flatten()[0]  # VFIB probability
                print("VFIB Prediction Flattened: ",vfib_pred)

                # Get class with highest probability (excluding VFIB)
                pred_class_idx = np.argmax(main_pred)
                pred_class_id = list(CLASSES.keys())[pred_class_idx]
                pred_class = CLASSES[pred_class_id]['name']
                confidence = main_pred[pred_class_idx] * 100

                # Combine VFIB probability with main predictions
                class_probabilities = {}
                for i, (class_id, info) in enumerate(CLASSES.items()):
                    prob = main_pred[i] * 100
                    if class_id == 'VF':
                        # Use the dedicated VFIB output
                        prob = vfib_pred * 100
                    class_probabilities[class_id] = {
                        'name': info['name'],
                        'probability': prob,
                        'color': info['color'],
                        'weight': info['weight']
                    }

                # Calculate risk scores
                framingham_risk = compute_framingham_risk(age, cholesterol, hdl, systolic_bp, smoker, diabetes)
                grace_score = compute_grace_score(age, systolic_bp, heart_rate)
                print("ECG Normalized: ",ecg_normalized)
                # Generate ECG visualization
                print("Generate Plot: ",generate_ecg_plot(
                    ecg_normalized,
                    r_peaks,
                    f"ECG - Predicted: {pred_class}",
                    pred_class_id
                ))
                # ecg_plot_filename, ecg_plot_json = generate_ecg_plot(
                #     ecg_signal=ecg_filtered, # Plot the filtered (but not normalized) signal for better visual scale
                #     r_peaks=r_peaks,
                #     title=f"ECG Waveform - {len(r_peaks)} Beats Detected (Record: {record_num})",
                #     pred_class_id=pred_class_id,
                #     fs=fs
                # )

    #             if ecg_plot_json is None:
    #             # flash("Failed to generate ECG visualization.", "danger")
    #             # Decide how to handle - maybe show results without plot?
    #             # For now, let's continue but log it
    #             print("Warning: ecg_plot_json is None")

    #             # If you need the plot JSON for web display
    #             if ecg_plot_json:
    # # Store in a template variable for display
    #                 ecg_plot = ecg_plot_json

                # Generate PDF report
                # pdf_filename = generate_pdf(
                #     predicted_class=pred_class,
                #     framingham_risk=framingham_risk,
                #     grace_score=grace_score,
                #     heart_rate=heart_rate,
                #     qt_interval=qt_interval,
                #     pr_interval=pr_interval
                #     #ecg_filename=ecg_plot_filename  # This is now a placeholder filename
                # )

                
                result_data = {
                    'patient': patient,
                    'predicted_class': pred_class,
                    'predicted_class_id': pred_class_id,
                    'confidence': confidence,
                    'age': age,
                    'cholesterol': cholesterol,
                    'hdl': hdl,
                    'systolic_bp': systolic_bp,
                    'smoker': smoker,
                    'diabetes': diabetes,
                    'heart_rate': heart_rate,
                    'heart_rate_status': heart_rate_status,
                    'qt_interval': qt_interval,
                    'pr_interval': pr_interval,
                    # 'ecg_plot_filename': ecg_plot_filename,  # Keep for backward compatibility
                    # 'ecg_plot_json': ecg_plot_json,  # Add this line
                    # 'pdf_filename': pdf_filename,
                    'framingham_risk': framingham_risk,
                    'grace_score': grace_score,
                    'classes': CLASSES,
                    'class_probabilities': class_probabilities,
                    'record_num': record_num,
                    'all_beats_count': len(r_peaks) if r_peaks is not None else 0,
                    'doctor_name': patient.get('Doctor_Name', 'Not Assigned')
}
                print("Result Data: ",result_data)

                return render_template("resulttest.html", **result_data)

            except Exception as e:
                flash(f"Analysis error: {str(e)}", "danger")
                return redirect(url_for("automatic_analysis", patient_id=patient_id))

        return render_template("automatic_analysis.html", patient=patient)

    except Exception as e:
        flash(f"System error: {str(e)}", "danger")
        return redirect(url_for("input_form"))


@app.route('/ecg_waveform/<patient_id>', methods=['GET'])
def ecg_waveform(patient_id):
    record_num = request.args.get('record_num', '100')
    samples_to_show = request.args.get('samples', default=2000, type=int)
    
    try:
        # First get patient data
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("""
            SELECT p.*, d.Username AS Doctor_Name 
            FROM patient_profile p
            LEFT JOIN doctor d ON p.Doctor_ID = d.Doctor_ID
            WHERE p.Patient_ID = %s
        """, (patient_id,))
        patient = cursor.fetchone()
        cursor.close()

        if not patient:
            flash("Patient not found", "danger")
            return redirect(url_for('automatic_analysis', patient_id=patient_id))

        # Load and process ECG data
        ecg_signal, fs, _, _ = load_ecg_sample(record_num)
        if ecg_signal is None or fs is None:
            flash("Failed to load ECG record", "danger")
            return redirect(url_for('automatic_analysis', patient_id=patient_id))
        
        ecg_signal = np.array(ecg_signal)
        ecg_filtered = butterworth_filter(ecg_signal, fs=fs)
        if ecg_filtered is None:
            flash("Failed to filter ECG signal", "danger")
            return redirect(url_for('automatic_analysis', patient_id=patient_id))
        
        r_peaks = detect_r_peaks(ecg_filtered, fs)
        r_peaks = np.array(r_peaks) if r_peaks is not None else np.array([])
        
        print(f"Generating ECG plot for patient {patient_id}...")
        plot_figure, plot_json = generate_ecg_plot(
            ecg_signal=ecg_filtered,
            r_peaks=r_peaks,
            title=f"ECG Waveform - {patient['Patient_Name']}",
            pred_class_id=None,
            fs=fs,
            samples_to_show=samples_to_show
        )
        
        if not plot_json:
            flash("Failed to generate ECG plot data", "danger")
            return redirect(url_for('automatic_analysis', patient_id=patient_id))
        
        try:
            safe_plot_json = json.dumps(plot_json) if not isinstance(plot_json, str) else plot_json
            json.loads(safe_plot_json)  # Validate
        except json.JSONDecodeError as e:
            flash("Invalid JSON data generated for plot", "danger")
            print(f"JSON decode error: {str(e)}")
            return redirect(url_for('automatic_analysis', patient_id=patient_id))
        
        visible_beats = np.sum(r_peaks < samples_to_show) if len(r_peaks) > 0 else 0
        
        return render_template('ecg_waveform.html', 
                            plot_json=safe_plot_json,
                            all_beats_count=len(r_peaks),
                            visible_beats_count=visible_beats,
                            record_num=record_num,
                            patient=patient,  # Pass patient data to template
                            patient_id=patient_id,
                            samples_to_show=samples_to_show,
                            total_samples=len(ecg_filtered))
    
    except Exception as e:
        flash(f"Error generating waveform: {str(e)}", "danger")
        print(f"Error traceback: {traceback.format_exc()}")
        return redirect(url_for('automatic_analysis', patient_id=patient_id))
    
# @app.route('/ecg_waveform/<patient_id>', methods=['GET'])
# def ecg_waveform(patient_id):
#     record_num = request.args.get('record_num', '100')
#     samples_to_show = request.args.get('samples', default=2000, type=int)
    
#     try:
#         # Load and process ECG data
#         ecg_signal, fs, _, _ = load_ecg_sample(record_num)
#         if ecg_signal is None or fs is None:
#             flash("Failed to load ECG record", "danger")
#             return redirect(url_for('automatic_analysis', patient_id=patient_id))
        
#         # Convert to numpy array if not already
#         ecg_signal = np.array(ecg_signal)
        
#         # Apply Butterworth filter
#         ecg_filtered = butterworth_filter(ecg_signal, fs=fs)
#         if ecg_filtered is None:
#             flash("Failed to filter ECG signal", "danger")
#             return redirect(url_for('automatic_analysis', patient_id=patient_id))
        
#         # Detect R-peaks
#         r_peaks = detect_r_peaks(ecg_filtered, fs)
#         r_peaks = np.array(r_peaks) if r_peaks is not None else np.array([])
        
#         # Generate plot data with debug checks
#         # Debug checks (remove after verification)
#         print(f"Signal length: {len(ecg_signal)} samples")
#         print(f"First 5 values: {ecg_signal[:5]}")
#         # print(f"Time axis range: {time_axis[0]:.2f}s to {time_axis[-1]:.2f}s")
#         if r_peaks is not None:
#             print(f"First 5 R-peaks: {r_peaks[:5]} samples")


        
#         print("Generating ECG plot...")
#         plot_figure, plot_json = generate_ecg_plot(
#             ecg_signal=ecg_filtered,
#             r_peaks=r_peaks,
#             title=f"ECG Waveform - {len(r_peaks)} Beats Detected",
#             pred_class_id=None,
#             fs=fs,
#             samples_to_show=samples_to_show
#         )
        
#         # Validate the plot JSON
#         if not plot_json:
#             flash("Failed to generate ECG plot data", "danger")
#             return redirect(url_for('automatic_analysis', patient_id=patient_id))
        
#         try:
#             # Ensure proper JSON encoding
#             if isinstance(plot_json, str):
#                 json.loads(plot_json)  # Validate it's proper JSON
#                 safe_plot_json = plot_json
#             else:
#                 safe_plot_json = json.dumps(plot_json)
#         except json.JSONDecodeError as e:
#             flash("Invalid JSON data generated for plot", "danger")
#             print(f"JSON decode error: {str(e)}")
#             return redirect(url_for('automatic_analysis', patient_id=patient_id))
        
#         # Calculate visible beats
#         visible_beats = np.sum(r_peaks < samples_to_show) if len(r_peaks) > 0 else 0
        
#         print("Rendering template with ECG data...")
#         return render_template('ecg_waveform.html', 
#                             plot_json=safe_plot_json,
#                             all_beats_count=len(r_peaks),
#                             visible_beats_count=visible_beats,
#                             record_num=record_num,
#                             patient_id=patient_id,
#                             samples_to_show=samples_to_show,
#                             total_samples=len(ecg_filtered))
    
#     except Exception as e:
#         flash(f"Error generating waveform: {str(e)}", "danger")
#         print(f"Error traceback: {traceback.format_exc()}")
#         return redirect(url_for('automatic_analysis', patient_id=patient_id))

@app.route('/generate_report/<patient_id>')
@login_required
def generate_report(patient_id):
    try:
        # Debug print all received parameters
        print(f"Received parameters: {request.args}")
        
        # Get patient data with enhanced doctor name handling
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        try:
            cursor.execute("""
                SELECT p.*, 
                       CASE 
                           WHEN p.Doctor_ID IS NULL THEN 'Not assigned'
                           WHEN d.Username IS NULL THEN 'Doctor not found'
                           ELSE CONCAT('Dr. ', d.Username)
                       END AS Doctor_Display_Name,
                       d.Username AS Doctor_Username
                FROM patient_profile p
                LEFT JOIN doctor d ON p.Doctor_ID = d.Doctor_ID
                WHERE p.Patient_ID = %s
            """, (patient_id,))
            patient = cursor.fetchone()
            
            if not patient:
                flash("Patient not found", "danger")
                return redirect(url_for("input_form"))
                
            # Debug output for doctor info
            print(f"Doctor info - ID: {patient.get('Doctor_ID')}, Username: {patient.get('Doctor_Username')}, Display: {patient.get('Doctor_Display_Name')}")
                
        except Exception as e:
            flash(f"Database error: {str(e)}", "danger")
            return redirect(url_for("input_form"))
        finally:
            cursor.close()

        # Get all parameters from request with defaults
        data = {
            'record_num': request.args.get('record_num', ''),
            'predicted_class': request.args.get('predicted_class', 'N/A'),
            'confidence': float(request.args.get('confidence', 0)),
            'heart_rate': float(request.args.get('heart_rate', 0)),
            'qt_interval': float(request.args.get('qt_interval', 0)),
            'pr_interval': float(request.args.get('pr_interval', 0)),
            'framingham_risk': float(request.args.get('framingham_risk', 0)),
            'grace_score': float(request.args.get('grace_score', 0)),
            'systolic_bp': float(request.args.get('systolic_bp', 120)),
            'cholesterol': float(request.args.get('cholesterol', 200)),
            'hdl': float(request.args.get('hdl', 50)),
            'smoker': request.args.get('smoker', '0') == '1',
            'diabetes': request.args.get('diabetes', '0') == '1',
            'all_beats_count': int(request.args.get('all_beats_count', 0)),
            'class_probabilities': {}
        }

        # Process class probabilities
        try:
            data['class_probabilities'] = json.loads(request.args.get('class_probabilities', '{}'))
        except json.JSONDecodeError:
            data['class_probabilities'] = {}

        # Generate ECG plot if record number is provided
        ecg_image = None
        if data['record_num']:
            try:
                ecg_signal, fs, _, _ = load_ecg_sample(data['record_num'])
                if ecg_signal is not None and fs is not None:
                    ecg_filtered = butterworth_filter(ecg_signal, fs=fs)
                    r_peaks = detect_r_peaks(ecg_filtered, fs)
                    
                    # Generate Plotly figure
                    fig = generate_ecg_plot(
                        ecg_signal=ecg_filtered,
                        r_peaks=r_peaks,
                        title=f"ECG Analysis - {patient['Patient_Name']}",
                        pred_class_id=next((k for k, v in CLASSES.items() if v['name'] == data['predicted_class']), None),
                        fs=fs
                    )
                    
                    # Convert Plotly figure to static image
                    img_bytes = fig.to_image(format='png', width=1000, height=500)
                    ecg_image = base64.b64encode(img_bytes).decode('utf-8')
                    
            except Exception as e:
                app.logger.error(f"ECG plot generation error: {str(e)}")
                flash("Error generating ECG plot", "warning")

        # Prepare report context with proper doctor name handling
        report_date = datetime.now()
        context = {
            'patient': patient,
            'doctor_name': patient.get('Doctor_Display_Name', 'Not assigned'),  # Use the pre-formatted display name
            'age': patient.get('Age', 'N/A'),
            'gender': patient.get('Gender', 'N/A'),
            'report_date': report_date,
            'current_year': report_date.year,
            'report_id': f"ECG-{report_date.strftime('%Y%m%d')}-{patient_id}",
            'ecg_image': ecg_image,
            'classes': CLASSES,
            **data  # Unpack all analysis data
        }

        # Debug output
        print(f"Template context: {context}")

        # Check if download was requested
        if request.args.get('download') == 'pdf':
            html = render_template('report.html', **context)
            pdf = BytesIO()
            pisa_status = pisa.CreatePDF(html, dest=pdf)
            
            if pisa_status.err:
                flash("Error generating PDF", "danger")
                return redirect(url_for('automatic_analysis', patient_id=patient_id))
            
            pdf.seek(0)
            response = make_response(pdf.getvalue())
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename={context["report_id"]}.pdf'
            return response

        # Render the report template
        return render_template('report.html', **context)

    except Exception as e:
        flash(f"Error generating report: {str(e)}", "danger")
        app.logger.error(f"Report generation error: {str(e)}\n{traceback.format_exc()}")
        return redirect(url_for('automatic_analysis', patient_id=patient_id))
    
@app.route("/dashboard")
@doctor_required
def dashboard():
    return render_template("dashboard.html", username=current_user.username)
@app.context_processor
def inject_pytz():
   return dict(pytz=pytz)

def generate_patient_id():
    current_year = datetime.now().year
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    try:
        cursor.execute("""
            SELECT Patient_ID 
            FROM patient_profile 
            WHERE Patient_ID LIKE CONCAT('PT-%%', %s) 
            ORDER BY Patient_ID DESC 
            LIMIT 1
        """, (current_year,))
        
        last_patient = cursor.fetchone()
        if last_patient:
            last_num = int(last_patient['Patient_ID'].split('-')[1])
            new_num = last_num + 1
        else:
            new_num = 10001
        
        new_patient_id = f"PT-{new_num}-{current_year}"
        return new_patient_id
    except MySQLdb.Error as e:
        app.logger.error(f"Database error generating patient ID: {str(e)}")
        raise
    finally:
        cursor.close()

def validate_doctor_id(doctor_id):
    pattern = r'^DR-\d{3}-\d{4}$'
    return re.match(pattern, doctor_id) is not None

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

@app.route("/patient_registration", methods=["GET", "POST"])
@login_required
def patient_registration():
    patient = None
    errors = {}
    form_data = {}
    
    if request.method == "POST":
        doctor_id = request.form.get("Doctor_ID", "")
        patient_name = request.form.get("Patient_Name", "")
        age = request.form.get("Age", "")
        gender = request.form.get("Gender", "")
        address = request.form.get("Address", "")
        email_id = request.form.get("Email_ID", "")
        personal_contact = request.form.get("Personal_Contact", "")
        emergency_contact = request.form.get("Emergency_Contact", "")
        
        form_data = {
            "Doctor_ID": doctor_id,
            "Patient_Name": patient_name,
            "Age": age,
            "Gender": gender,
            "Address": address,
            "Email_ID": email_id,
            "Personal_Contact": personal_contact,
            "Emergency_Contact": emergency_contact
        }
        
        # Validation checks
        if not patient_name.strip():
            errors["Patient_Name"] = "Please enter the patient's full name."
        
        try:
            age = int(age)
            if age < 1 or age > 150:
                errors["Age"] = "Please enter a valid age between 1 and 150."
        except (ValueError, TypeError):
            errors["Age"] = "Please enter a valid age."
        
        if not gender:
            errors["Gender"] = "Please select a gender."
        
        if not address.strip():
            errors["Address"] = "Please enter the patient's address."
        
        if not email_id or not validate_email(email_id):
            errors["Email_ID"] = "Please enter a valid email address."
        
        if not personal_contact or not personal_contact.isdigit() or len(personal_contact) != 10:
            errors["Personal_Contact"] = "Please enter a valid 10-digit phone number."
        
        if not emergency_contact or not emergency_contact.isdigit() or len(emergency_contact) != 10:
            errors["Emergency_Contact"] = "Please enter a valid 10-digit phone number."
        
        if personal_contact == emergency_contact:
            errors["Emergency_Contact"] = "Personal contact and emergency contact cannot be the same."
        
        if not doctor_id or not validate_doctor_id(doctor_id):
            errors["Doctor_ID"] = "Please enter a valid doctor ID in the format DR-001-2024."
        
        # Check for duplicate email
        if not errors.get("Email_ID"):
            try:
                cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                cursor.execute("SELECT * FROM patient_profile WHERE Email_ID = %s", (email_id,))
                existing_email = cursor.fetchone()
                if existing_email:
                    errors["Email_ID"] = "Email ID already exists. Please use a different email address."
                cursor.close()
            except MySQLdb.Error as e:
                flash(f"Database error: {str(e)}", "danger")
        
        if errors:
            return render_template("patient_registration.html", errors=errors, form_data=form_data, patient=patient)
        
        try:
            patient_id = generate_patient_id()

            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute(
                """INSERT INTO patient_profile 
                   (Patient_ID, Patient_Name, Age, Gender, Address, Email_ID, Personal_Contact, 
                    Emergency_Contact, Doctor_ID, Created_At, Staff_Username)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (patient_id, patient_name, age, gender, address, email_id, personal_contact,
                 emergency_contact, doctor_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_user.username)
            )
            mysql.connection.commit()
            # flash("Patient registered successfully!", "success")
            
            cursor.execute("SELECT * FROM patient_profile WHERE Patient_ID = %s", (patient_id,))
            patient = cursor.fetchone()
            cursor.close()
            
        except MySQLdb.Error as e:
            flash(f"Error registering patient: {str(e)}", "danger")

    return render_template("patient_registration.html", patient=patient, errors=errors, form_data=form_data)

@app.route("/edit_patient/<patient_id>", methods=["GET", "POST"])
@login_required
def edit_patient(patient_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    try:
        if request.method == "POST":
            patient_name = request.form.get("Patient_Name", "")
            age = request.form.get("Age", "")
            gender = request.form.get("Gender", "")
            address = request.form.get("Address", "")
            email_id = request.form.get("Email_ID", "")
            personal_contact = request.form.get("Personal_Contact", "")
            emergency_contact = request.form.get("Emergency_Contact", "")
            doctor_id = request.form.get("Doctor_ID", "")

            cursor.execute(
                """UPDATE patient_profile 
                   SET Patient_Name = %s, Age = %s, Gender = %s, Address = %s, Email_ID = %s, 
                       Personal_Contact = %s, Emergency_Contact = %s, Doctor_ID = %s
                   WHERE Patient_ID = %s""",
                (patient_name, age, gender, address, email_id, personal_contact,
                 emergency_contact, doctor_id, patient_id)
            )
            mysql.connection.commit()
            # flash("Patient details updated successfully!", "success")
            return redirect(url_for("patient_registration"))
        
        cursor.execute("SELECT * FROM patient_profile WHERE Patient_ID = %s", (patient_id,))
        patient = cursor.fetchone()
        if not patient:
            flash("Patient not found", "danger")
            return redirect(url_for("patient_registration"))
        
        return render_template("edit_patient.html", patient=patient)
    except MySQLdb.Error as e:
        flash(f"Error updating patient details: {str(e)}", "danger")
    finally:
        cursor.close()
    
    return redirect(url_for("patient_registration"))

@app.route("/debug")
def debug():
    return jsonify({
        "user_id": session.get("_user_id"),
        "is_authenticated": current_user.is_authenticated,
        "user_type": current_user.user_type if current_user.is_authenticated else None
    })

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)