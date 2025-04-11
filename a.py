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
from scipy.signal import butter, filtfilt
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

# Load model with verification
# Replace your current model loading code with this:
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
# try:
#     model = load_model(MODEL_PATH)
#     if not hasattr(model, 'optimizer') or model.optimizer is None:
#         model.compile(optimizer='adam',
#                      loss={'main_output': 'categorical_crossentropy', 
#                            'vfib_output': 'binary_crossentropy'},
#                      metrics=['accuracy'])
    
#     # Verify model structure
#     if len(model.outputs) != 2:
#         raise ValueError("Model should have exactly 2 outputs")
#     if model.outputs[0].shape[-1] != len(CLASSES):
#         raise ValueError(f"Main output should have {len(CLASSES)} units")
        
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model = None

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
        processed_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="pantompkins1985")
        _, r_peaks = nk.ecg_peaks(processed_ecg, sampling_rate=fs)
        return np.array(r_peaks["ECG_R_Peaks"])
    except Exception as e:
        print(f"Error detecting R-peaks: {e}")
        return np.array([])  # Return empty array instead of None

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

# def generate_ecg_plot(ecg_signal, beats=None, title="ECG Signal", pred_class_id=None):
#     try:
#         # Downsample the signal for plotting
#         plot_samples = 5000  # Target number of samples for plotting
#         downsampled_signal = downsample_ecg(ecg_signal, plot_samples)
        
#         # Adjust beat positions for downsampled signal
#         if beats is not None and len(beats) > 0:
#             original_length = len(ecg_signal)
#             downsampled_beats = [int(b * plot_samples / original_length) for b in beats]
#         else:
#             downsampled_beats = None
        
#         fig = go.Figure()
        
#         # Plot downsampled ECG signal
#         fig.add_trace(go.Scatter(
#             y=downsampled_signal,
#             mode='lines',
#             line=dict(color='lightgray', width=1),
#             name='ECG Signal'
#         ))
        
#         # Highlight beats if provided
#         if downsampled_beats and len(downsampled_beats) > 0:
#             if pred_class_id:
#                 class_color = CLASSES.get(pred_class_id, {}).get('color', 'red')
#                 fig.add_trace(go.Scatter(
#                     x=downsampled_beats,
#                     y=[downsampled_signal[b] for b in downsampled_beats],
#                     mode='markers',
#                     marker=dict(color=class_color, size=8),
#                     name=f'{CLASSES.get(pred_class_id, {}).get("name", "Predicted")} beats'
#                 ))
#             else:
#                 fig.add_trace(go.Scatter(
#                     x=downsampled_beats,
#                     y=[downsampled_signal[b] for b in downsampled_beats],
#                     mode='markers',
#                     marker=dict(color='blue', size=6),
#                     name='Detected beats'
#                 ))
        
#         fig.update_layout(
#             title=title,
#             xaxis_title='Samples',
#             yaxis_title='Amplitude',
#             showlegend=True,
#             template='plotly_white'
#         )
        
#         return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
#     except Exception as e:
#         print(f"Error generating ECG plot: {e}")
#         return None


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
    
# def preprocess_ecg(ecg_signal, r_peaks=None, target_length=180):
#     """Full ECG preprocessing pipeline"""
#     try:
#         # Filter and normalize
#         ecg_filtered = butterworth_filter(ecg_signal)
#         ecg_normalized = (ecg_filtered - np.mean(ecg_filtered)) / np.std(ecg_filtered)
        
#         # Align to R-peak if available
#         if r_peaks is not None and len(r_peaks) > 0:
#             ecg_aligned = align_ecg_to_r_peak(ecg_normalized, r_peaks, target_length)
#         else:
#             ecg_aligned = center_pad_ecg(ecg_normalized, target_length)
            
#         return np.expand_dims(np.expand_dims(ecg_aligned, axis=-1), axis=0)
#     except Exception as e:
#         print(f"Error in ECG preprocessing: {e}")
#         return np.zeros((1, target_length, 1))
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
def generate_ecg_plot(ecg_signal, r_peaks=None, title="ECG Signal", pred_class_id=None, fs=360, samples_to_show=2000):
    """
    Generate interactive Plotly visualization for ECG data with:
    - Pink background (#FFC0CB)
    - Black ECG line
    - R-peak markers at the absolute peak with dark pink borders
    """
    # Convert inputs to numpy arrays
    ecg_signal = np.array(ecg_signal)
    if r_peaks is not None:
        r_peaks = np.array(r_peaks)
    
    # Create figure with pink background
    fig = go.Figure()
    
    # Add ECG trace with black line
    fig.add_trace(go.Scatter(
        x=list(range(samples_to_show)),
        y=ecg_signal[:samples_to_show].tolist(),
        mode='lines',
        line=dict(color='#000000', width=1.5),
        name='ECG Signal'
    ))
    
    # Highlight R-peaks if available
    if r_peaks is not None and len(r_peaks) > 0:
        visible_r_peaks = r_peaks[r_peaks < samples_to_show]
        class_info = next((v for k, v in CLASSES.items() if v.get('id') == pred_class_id), 
                         {'name': 'Unknown', 'color': '#95a5a6'})
        
        # Find exact peak positions by looking at nearby samples
        refined_peaks = []
        for peak in visible_r_peaks:
            # Look at ±5 samples around the detected peak
            start = max(0, peak - 5)
            end = min(len(ecg_signal), peak + 5)
            window = ecg_signal[start:end]
            # Find the actual maximum in this window
            exact_peak = start + np.argmax(window)
            refined_peaks.append(exact_peak)
        
        refined_peaks = np.array(refined_peaks)
        
        fig.add_trace(go.Scatter(
            x=refined_peaks.tolist(),
            y=ecg_signal[refined_peaks].tolist(),
            mode='markers',
            marker=dict(
                color=class_info['color'],
                size=8,
                line=dict(
                    width=2, 
                    color='#C71585'  # Dark pink border
                ),
                symbol='diamond'  # Diamond shape for better peak visibility
            ),
            name=f'R-peaks ({len(refined_peaks)} beats)'
        ))
    
    # Update layout with pink background
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title='Samples',
        yaxis_title='Amplitude (mV)',
        height=450,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='#FFC0CB',
        paper_bgcolor='#FFC0CB',
        hovermode='closest',
        showlegend=True,
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="linear"
        )
    )
    
    return f"ecg_plot_{int(time.time())}.html", json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# def generate_ecg_plot(ecg_signal, r_peaks=None, title="ECG Signal", pred_class_id=None, fs=360, samples_to_show=2000):
#     """
#     Generate interactive Plotly visualization for ECG data.
    
#     Args:
#         ecg_signal: Array of ECG signal values
#         r_peaks: Array of R-peak indices
#         title: Plot title
#         pred_class_id: Class ID for rhythm prediction
#         fs: Sampling frequency (default: 360 Hz)
#         samples_to_show: Number of samples to display (default: 2000)
        
#     Returns:
#         placeholder_filename: A placeholder filename for backward compatibility
#         plot_json: JSON representation of the Plotly figure
#     """
#     # Convert inputs to numpy arrays if they aren't already
#     ecg_signal = np.array(ecg_signal)
#     if r_peaks is not None:
#         r_peaks = np.array(r_peaks)
    
#     # Create figure
#     fig = go.Figure()
    
#     # Add ECG trace - showing only the first samples_to_show samples
#     fig.add_trace(go.Scatter(
#         x=list(range(samples_to_show)),
#         y=ecg_signal[:samples_to_show].tolist(),
#         mode='lines',
#         line=dict(color='lightgray', width=1),
#         name='ECG Signal'
#     ))
    
#     # Highlight R-peaks if available
#     if r_peaks is not None and len(r_peaks) > 0:
#         # Get only peaks within the visible range
#         visible_r_peaks = r_peaks[r_peaks < samples_to_show]
        
#         # Get class information
#         class_info = next((v for k, v in CLASSES.items() if v.get('id') == pred_class_id), 
#                          {'name': 'Unknown', 'color': '#95a5a6'})
        
#         fig.add_trace(go.Scatter(
#             x=visible_r_peaks.tolist(),
#             y=ecg_signal[visible_r_peaks].tolist(),
#             mode='markers',
#             marker=dict(
#                 color=class_info['color'],
#                 size=8,
#                 line=dict(width=2, color='white')
#             ),
#             name=f'R-peaks ({len(visible_r_peaks)} beats)'
#         ))
    
#     # Update layout
#     fig.update_layout(
#         title=dict(
#             text=title,
#             x=0.5,
#             font=dict(size=18)
#         ),
#         xaxis_title='Samples',
#         yaxis_title='Amplitude (mV)',
#         height=450,
#         margin=dict(l=50, r=50, t=80, b=50),
#         plot_bgcolor='#f8f9fa',
#         paper_bgcolor='#ffffff',
#         hovermode='closest',
#         showlegend=True,
#         template='plotly_white'
#     )
    
#     # Add range slider for better navigation
#     fig.update_layout(
#         xaxis=dict(
#             rangeslider=dict(visible=True),
#             type="linear"
#         )
#     )
    
#     # Convert figure to JSON for web display
#     plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
#     # Create a placeholder filename for backward compatibility
#     placeholder_filename = f"ecg_plotly_{int(time.time())}.html"
    
#     return placeholder_filename, plot_json
# def generate_ecg_plot(ecg_signal, r_peaks, title="ECG Signal", pred_class_id=None, fs=360, show_seconds=5):
#     """Generate interactive ECG plot with Plotly but skip image saving"""
#     try:
#         print("Starting generate_ecg_plot function")
#         print(f"ECG Signal length: {len(ecg_signal)}")
#         print(f"R-peaks length: {len(r_peaks)}")
        
#         samples_to_show = fs * show_seconds
#         print(f"Samples to show: {samples_to_show}")
        
#         time_axis = np.arange(len(ecg_signal)) / fs
#         print("Created time axis")
        
#         # Get class information
#         print(f"Looking up class info for pred_class_id: {pred_class_id}")
#         class_info = CLASSES.get(pred_class_id, {'name': 'Unknown', 'color': '#95a5a6'})
#         print(f"Class info: {class_info}")
        
#         print("Creating Plotly figure...")
#         fig = go.Figure()
        
#         print("Adding ECG trace...")
#         # Add ECG trace
#         fig.add_trace(go.Scatter(
#             x=time_axis[:samples_to_show],
#             y=ecg_signal[:samples_to_show],
#             mode='lines',
#             line=dict(color='#7f8c8d', width=1.5),
#             name='ECG Signal'
#         ))
        
#         print("ECG trace added")
        
#         # Highlight beats if available
#         if len(r_peaks) > 0:
#             print("Adding R-peaks...")
#             visible_r_peaks = r_peaks[r_peaks < samples_to_show]
#             print(f"Visible R-peaks: {len(visible_r_peaks)}")
            
#             fig.add_trace(go.Scatter(
#                 x=time_axis[visible_r_peaks],
#                 y=ecg_signal[visible_r_peaks],
#                 mode='markers',
#                 marker=dict(
#                     color=class_info['color'],
#                     size=10,
#                     line=dict(width=2, color='white')
#                 ),
#                 name=f'R-peaks ({len(visible_r_peaks)} beats)'
#             ))
#             print("R-peaks added")
        
#         print("Adding annotations...")
#         # Add rhythm classification annotation
#         fig.add_annotation(
#             x=0.05,
#             y=1.1,
#             xref='paper',
#             yref='paper',
#             text=f"<b>Rhythm: {class_info['name']}</b>",
#             showarrow=False,
#             font=dict(size=14, color=class_info['color'])
#         )
        
#         print("Updating layout...")
#         fig.update_layout(
#             title=dict(
#                 text=title,
#                 x=0.5,
#                 font=dict(size=18)
#             ),
#             xaxis_title='Time (seconds)',
#             yaxis_title='Amplitude (mV)',
#             height=450
#         )
        
#         print("Layout updated")
        
#         # Create directory but skip image saving for now
#         plot_dir = os.path.join("static", "ecg_plots")
#         os.makedirs(plot_dir, exist_ok=True)
        
#         # Use a placeholder filename
#         filename = os.path.join(plot_dir, f"ecg_placeholder_{int(time.time())}.png")
        
#         # Convert to JSON for web display
#         print("Converting figure to JSON...")
#         plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
#         print("JSON conversion complete")
        
#         print("Skipping image writing for now")
#         # fig.write_image(filename)  # Comment out the problematic line
        
#         return filename, plot_json
#     except Exception as e:
#         print(f"Error generating ECG plot: {e}")
#         import traceback
#         traceback.print_exc()
#         return None, None

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

                return render_template("result.html", **result_data)

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
        # Load and process ECG data
        ecg_signal, fs, _, _ = load_ecg_sample(record_num)
        if ecg_signal is None or fs is None:
            flash("Failed to load ECG record", "danger")
            return redirect(url_for('automatic_analysis', patient_id=patient_id))
        
        # Convert to numpy array if not already
        ecg_signal = np.array(ecg_signal)
        
        # Apply Butterworth filter
        ecg_filtered = butterworth_filter(ecg_signal, fs=fs)
        if ecg_filtered is None:
            flash("Failed to filter ECG signal", "danger")
            return redirect(url_for('automatic_analysis', patient_id=patient_id))
        
        # Detect R-peaks
        r_peaks = detect_r_peaks(ecg_filtered, fs)
        r_peaks = np.array(r_peaks) if r_peaks is not None else np.array([])
        
        # Generate plot data with debug checks
        print("Generating ECG plot...")
        plot_figure, plot_json = generate_ecg_plot(
            ecg_signal=ecg_filtered,
            r_peaks=r_peaks,
            title=f"ECG Waveform - {len(r_peaks)} Beats Detected",
            pred_class_id=None,
            fs=fs,
            samples_to_show=samples_to_show
        )
        
        # Validate the plot JSON
        if not plot_json:
            flash("Failed to generate ECG plot data", "danger")
            return redirect(url_for('automatic_analysis', patient_id=patient_id))
        
        try:
            # Ensure proper JSON encoding
            if isinstance(plot_json, str):
                json.loads(plot_json)  # Validate it's proper JSON
                safe_plot_json = plot_json
            else:
                safe_plot_json = json.dumps(plot_json)
        except json.JSONDecodeError as e:
            flash("Invalid JSON data generated for plot", "danger")
            print(f"JSON decode error: {str(e)}")
            return redirect(url_for('automatic_analysis', patient_id=patient_id))
        
        # Calculate visible beats
        visible_beats = np.sum(r_peaks < samples_to_show) if len(r_peaks) > 0 else 0
        
        print("Rendering template with ECG data...")
        return render_template('ecg_waveform.html', 
                            plot_json=safe_plot_json,
                            all_beats_count=len(r_peaks),
                            visible_beats_count=visible_beats,
                            record_num=record_num,
                            patient_id=patient_id,
                            samples_to_show=samples_to_show,
                            total_samples=len(ecg_filtered))
    
    except Exception as e:
        flash(f"Error generating waveform: {str(e)}", "danger")
        print(f"Error traceback: {traceback.format_exc()}")
        return redirect(url_for('automatic_analysis', patient_id=patient_id))

        


@app.route('/generate_report/<patient_id>')
@login_required
def generate_report(patient_id):
    try:
        # Fetch patient data
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
            return redirect(url_for("input_form"))

        # Get parameters with defaults
        record_num = request.args.get('record_num', '100')
        predicted_class = request.args.get('predicted_class', 'Normal')
        confidence = float(request.args.get('confidence', 95.0))
        age = int(request.args.get('age', patient.get('Age', 30)))
        gender = patient.get('Gender', 'Unknown')
        systolic_bp = int(request.args.get('systolic_bp', 120))
        cholesterol = int(request.args.get('cholesterol', 150))
        hdl = int(request.args.get('hdl', 40))
        smoker = bool(int(request.args.get('smoker', 0)))
        diabetes = bool(int(request.args.get('diabetes', 0)))
        framingham_risk = float(request.args.get('framingham_risk', 10.0))
        grace_score = float(request.args.get('grace_score', 20.0))
        heart_rate = float(request.args.get('heart_rate', 72.0))
        qt_interval = float(request.args.get('qt_interval', 0.36))
        
        # ECG Data Processing
        ecg_signal, fs, _, _ = load_ecg_sample(record_num)
        if ecg_signal is None or fs is None:
            flash("Failed to load ECG record", "danger")
            return redirect(url_for('automatic_analysis', patient_id=patient_id))
        
        ecg_filtered = butterworth_filter(ecg_signal, fs=fs)
        r_peaks = detect_r_peaks(ecg_filtered, fs)
        
        # Generate Plotly figure
        fig = go.Figure()
        time_axis = [i/fs for i in range(len(ecg_filtered))]
        
        # Show only first 10 seconds for better visualization
        max_samples = min(10 * fs, len(ecg_filtered))
        fig.add_trace(go.Scatter(
            x=time_axis[:max_samples],
            y=ecg_filtered[:max_samples],
            mode='lines',
            line=dict(color='black', width=1),
            name='ECG Signal'
        ))
        
        if r_peaks is not None:
            r_peaks = [x for x in r_peaks if x < max_samples]
            fig.add_trace(go.Scatter(
                x=[time_axis[i] for i in r_peaks],
                y=[ecg_filtered[i] for i in r_peaks],
                mode='markers',
                marker=dict(color='red', size=8),
                name='R-peaks'
            ))
        
        fig.update_layout(
            title=f"ECG Waveform - {len(r_peaks) if r_peaks else 0} Beats Detected",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude (mV)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=True
        )
        
        # Convert plot to JSON for HTML template
        plot_json = fig.to_json()
        
        # Prepare context for the report
        context = {
            'patient': patient,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'age': age,
            'gender': gender,
            'systolic_bp': systolic_bp,
            'cholesterol': cholesterol,
            'hdl': hdl,
            'smoker': smoker,
            'diabetes': diabetes,
            'framingham_risk': framingham_risk,
            'grace_score': grace_score,
            'grace_score_percentage': min(grace_score, 100),
            'plot_json': plot_json,
            'all_beats_count': len(r_peaks) if r_peaks else 0,
            'heart_rate': heart_rate,
            'qt_interval': qt_interval,
            'report_date': datetime.now(),
            'current_year': datetime.now().year
        }
        
        # For debugging - return HTML first
        if request.args.get('debug') == 'html':
            return render_template('report.html', **context)
        
        # Generate PDF
        html = render_template('report.html', **context)
        pdf = BytesIO()
        
        # PDF generation with error handling
        pisa_status = pisa.CreatePDF(
            html,
            dest=pdf,
            encoding='UTF-8',
            link_callback=lambda uri, _: uri  # Handle static files
        )
        
        if pisa_status.err:
            error_msg = f"PDF generation failed: {pisa_status.err}"
            print(error_msg)
            flash("Failed to generate PDF report", "danger")
            return redirect(url_for('automatic_analysis', patient_id=patient_id))
        
        pdf.seek(0)
        response = make_response(pdf.read())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=ECG_Report_{patient_id}.pdf'
        
        return response
        
    except Exception as e:
        error_msg = f"Error generating report: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        flash("An error occurred while generating the report", "danger")
        return redirect(url_for('automatic_analysis', patient_id=patient_id))
# @app.route('/analyze', methods=['POST'])
# def analyze():
#     data = request.json
#     record_num = data.get('record_num', '100')
    
#     if model is None:
#         return jsonify({'error': 'Model not loaded'}), 500
    
#     try:
#         # Load ECG data
#         ecg_signal, fs, _, _ = load_ecg_sample(record_num)
#         if ecg_signal is None:
#             return jsonify({'error': 'Failed to load ECG data'}), 400
        
#         # Preprocess ECG signal
#         ecg_signal = butterworth_filter(ecg_signal, fs=fs)
#         processed_ecg = preprocess_ecg(ecg_signal)
        
#         # Detect R-peaks
#         r_peaks = detect_r_peaks(ecg_signal, fs)
        
#         # Make prediction
#         predictions = model.predict(processed_ecg, verbose=0)
#         class_probs = handle_model_output(predictions)
        
#         if class_probs is None:
#             return jsonify({'error': 'Invalid model output'}), 500
        
#         # Get results
#         pred_idx = np.argmax(class_probs)
#         confidence = class_probs[pred_idx] * 100
#         pred_class = CLASSES[pred_idx]
        
#         # Create plot
#         plot_data = generate_ecg_plot(ecg_signal, r_peaks, pred_class['id'], fs)
        
#         return jsonify({
#             'prediction': pred_class['name'],
#             'class_id': pred_class['id'],
#             'confidence': float(confidence),
#             'priority': pred_class['weight'],
#             'plot_data': plot_data,
#             'heart_rate': float(60 / np.mean(np.diff(r_peaks)/fs)) if len(r_peaks) > 1 else 0
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500




# @app.route("/automatic_analysis/<patient_id>", methods=["GET", "POST"])
# @login_required
# def automatic_analysis(patient_id):
#     # Validate patient ID
#     if not re.match(r'^PT-\d{5}-\d{4}$', patient_id):
#         flash("Invalid Patient ID format", "danger")
#         return redirect(url_for("input_form"))

#     # Get patient data
#     cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
#     try:
#         cursor.execute("SELECT * FROM patient_profile WHERE Patient_ID = %s", (patient_id,))
#         patient = cursor.fetchone()
#     except Exception as e:
#         flash(f"Database error: {str(e)}", "danger")
#         return redirect(url_for("input_form"))
#     finally:
#         cursor.close()

#     if not patient:
#         flash("Patient not found", "danger")
#         return redirect(url_for("input_form"))

#     if request.method == "POST":
#         try:
#             # Get user inputs
#             record_num = request.form.get("record_num", "100")
#             age = int(request.form.get("age", patient.get("Age", 30)))
#             cholesterol = int(request.form.get("cholesterol", 150))
#             hdl = int(request.form.get("hdl", 40))
#             systolic_bp = int(request.form.get("systolic_bp", 120))
#             smoker = 1 if "smoker" in request.form else 0
#             diabetes = 1 if "diabetes" in request.form else 0

#             # Save to database
#             cursor = mysql.connection.cursor()
#             try:
#                 cursor.execute("""
#                     INSERT INTO input 
#                     (Patient_ID, Blood_Pressure, Cholesterol, HDL, Smoker, Diabetic, Generated_AT, Doctor_ID)
#                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#                 """, (patient_id, systolic_bp, cholesterol, hdl, smoker, diabetes, 
#                      datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_user.id))
#                 mysql.connection.commit()
#             except Exception as e:
#                 mysql.connection.rollback()
#                 flash(f"Failed to save data: {str(e)}", "danger")
#                 return redirect(url_for("automatic_analysis", patient_id=patient_id))
#             finally:
#                 cursor.close()

#             # Load and preprocess ECG
#             ecg_signal, fs, _, _ = load_ecg_sample(record_num)
#             if ecg_signal is None:
#                 flash("Failed to load ECG data", "danger")
#                 return redirect(url_for("automatic_analysis", patient_id=patient_id))

#             # Check ECG quality
#             quality_ok, quality_msg = check_ecg_quality(ecg_signal, fs)
#             if not quality_ok:
#                 flash(f"Poor ECG quality: {quality_msg}", "warning")

#             # Process ECG
#             ecg_filtered = butterworth_filter(ecg_signal, fs=fs)
#             ecg_processed = (ecg_filtered - np.mean(ecg_filtered)) / np.std(ecg_filtered)

#             # Detect R-peaks and calculate heart rate
#             r_peaks = detect_r_peaks(ecg_filtered, fs)
#             if len(r_peaks) < 2:
#                 flash("Insufficient R-peaks detected", "danger")
#                 return redirect(url_for("automatic_analysis", patient_id=patient_id))

#             heart_rate, qt_interval, pr_interval, qrs_peaks = compute_intervals(ecg_filtered, r_peaks, fs)
#             heart_rate_status = "Bradycardia" if heart_rate < 60 else "Tachycardia" if heart_rate > 100 else "Normal"

#             # Prepare model input with R-peak alignment
#             model_input = preprocess_ecg(ecg_processed, r_peaks)

#             # Make prediction
#             predictions = model.predict(model_input, verbose=0)

#             if isinstance(predictions, list):
#                 # Dual output model (main classification + VFib detection)
#                 class_probs = predictions[0][0]  # Shape: (7,) for 7 classes
#                 vfib_prob = predictions[1][0][0]  # Single probability for VFib
#             else:
#                 # Single output model fallback
#                 class_probs = predictions[0]  # Shape: (7,)
#                 vfib_prob = None

#             try:
#                 # Get predicted class
#                 pred_class_idx = np.argmax(class_probs)
#                 pred_class = CLASSES[pred_class_idx]
#                 confidence = float(class_probs[pred_class_idx]) * 100

#                 # Process VFib detection if available
#                 vfib_detected = vfib_prob > 0.5 if vfib_prob is not None else None
#                 vfib_confidence = float(vfib_prob) * 100 if vfib_prob is not None else None

#                 # Create class probabilities dictionary for display
#                 class_probabilities = {
#                     class_id: {
#                         'name': CLASSES[i]['name'],
#                         'probability': float(class_probs[i]) * 100,
#                         'color': CLASSES[i]['color'],
#                         'weight': CLASSES[i]['weight']
#                     }
#                     for i, class_id in enumerate(CLASSES.keys())
#                 }

#             except Exception as e:
#                 flash("Failed to interpret model output", "danger")
#                 app.logger.error(f"Prediction error: {str(e)}\nRaw output: {predictions}")
#                 return redirect(url_for("automatic_analysis", patient_id=patient_id))

#             # Calculate risk scores
#             framingham_risk = compute_framingham_risk(age, cholesterol, hdl, systolic_bp, smoker, diabetes)
#             grace_score = compute_grace_score(age, systolic_bp, heart_rate)

#             # Generate ECG plot
#             ecg_graph_json = generate_ecg_plot(
#                 ecg_processed, 
#                 r_peaks, 
#                 f"ECG - {pred_class['name']}",
#                 pred_class['id']
#             )

#             # Generate PDF report
#             pdf_filename = generate_pdf(
#                 predicted_class=pred_class['name'],
#                 framingham_risk=framingham_risk,
#                 grace_score=grace_score,
#                 heart_rate=heart_rate,
#                 qt_interval=qt_interval,
#                 pr_interval=pr_interval,
#                 vfib_detected=vfib_detected,
#                 vfib_confidence=vfib_confidence,
#                 ecg_graph_json=ecg_graph_json
#             )

#             return render_template("result.html",
#                 patient=patient,
#                 predicted_class=pred_class['name'],
#                 predicted_class_id=pred_class['id'],
#                 confidence=confidence,
#                 age=age,
#                 cholesterol=cholesterol,
#                 hdl=hdl,
#                 systolic_bp=systolic_bp,
#                 smoker=smoker,
#                 diabetes=diabetes,
#                 heart_rate=heart_rate,
#                 heart_rate_status=heart_rate_status,
#                 qt_interval=qt_interval,
#                 pr_interval=pr_interval,
#                 ecg_graph_json=ecg_graph_json,
#                 framingham_risk=framingham_risk,
#                 grace_score=grace_score,
#                 vfib_detected=vfib_detected,
#                 vfib_confidence=vfib_confidence,
#                 class_probabilities=class_probabilities,
#                 classes=CLASSES,
#                 pdf_filename=pdf_filename
#             )

#         except Exception as e:
#             flash(f"Analysis failed: {str(e)}", "danger")
#             app.logger.error(f"Analysis error for {patient_id}:\n{traceback.format_exc()}")
#             return redirect(url_for("automatic_analysis", patient_id=patient_id))

#     return render_template("automatic_analysis.html", patient=patient)

@app.route("/download_report/<filename>")
@login_required
def download_report(filename):
    return send_from_directory(
        directory=os.path.join(app.root_path, 'static'),
        path=filename,
        as_attachment=True
    )

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

