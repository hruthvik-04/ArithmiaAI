import os
import re 
import json
import uuid
import traceback
import time
import logging 
from io import BytesIO
from datetime import datetime
from functools import wraps
import base64
import urllib.parse 
from flask import (Flask, render_template, request, jsonify, send_from_directory,
                   flash, redirect, url_for, session, make_response, abort)
from flask_login import (LoginManager, UserMixin, login_user, login_required,
                       logout_user, current_user)
from flask_bcrypt import Bcrypt
from flask_mysqldb import MySQL
import MySQLdb.cursors
import numpy as np
import pandas as pd 
import pytz 
import plotly
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from scipy.signal import butter, filtfilt, find_peaks 
import wfdb
import neurokit2 as nk 
try:
    from xhtml2pdf import pisa
    PDF_GENERATION_AVAILABLE = True
except ImportError:
    pisa = None
    PDF_GENERATION_AVAILABLE = False
    print("WARNING: xhtml2pdf library not found. PDF report downloads will be disabled.")



app = Flask(__name__, static_folder='static')

# --- Configuration ---

app.secret_key = os.environ.get("SECRET_KEY", "5010dfae019e413f06691431b2e3ba82bbb456c661b0d27332a4dbd5bbd36bd8") 
app.config["MYSQL_HOST"] = os.environ.get("MYSQL_HOST", "localhost")
app.config["MYSQL_USER"] = os.environ.get("MYSQL_USER", "root")
app.config["MYSQL_PASSWORD"] = os.environ.get("MYSQL_PASSWORD", "452003@hrX")
app.config["MYSQL_DB"] = os.environ.get("MYSQL_DB", "hospital_ecg_db")
app.config["MYSQL_CURSORCLASS"] = "DictCursor" 

# --- Paths Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['STATIC_FOLDER'] = os.path.join(BASE_DIR, 'static')
DATASET_PATH = os.path.join(BASE_DIR, "mit-bih-arrhythmia-database-1.0.0/")
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'ecg_arrhythmia_detector_20250331_165229.h5')
ECG_IMAGE_DIR = os.path.join(app.config['STATIC_FOLDER'], 'ecg_images')


try:
    os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
    os.makedirs(ECG_IMAGE_DIR, exist_ok=True)
except OSError as e:
    app.logger.error(f"Could not create static directories: {e}")
    

# --- Constants ---
MODEL_INPUT_LENGTH = 180
DEFAULT_ECG_DISPLAY_SAMPLES = 2000
TARGET_TIMEZONE = 'Asia/Kolkata'

# Arrhythmia Class Definitions 
CLASSES = {
    0: {'id': 'N', 'name': 'Normal', 'weight': 1, 'color': '#2ecc71'},
    1: {'id': 'S', 'name': 'SVT', 'weight': 100, 'color': '#e67e22'},
    2: {'id': 'AF', 'name': 'Atrial Fibrillation', 'weight': 150, 'color': '#e74c3c'},
    3: {'id': 'VF', 'name': 'Ventricular Fibrillation', 'weight': 200, 'color': '#9b59b6'},
    4: {'id': 'VT', 'name': 'Ventricular Tachycardia', 'weight': 170, 'color': '#c0392b'},
    5: {'id': 'B', 'name': 'Heart Block', 'weight': 120, 'color': '#3498db'},
    6: {'id': 'F', 'name': 'Fusion', 'weight': 80, 'color': '#a67c52'}
}


mysql = MySQL(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
# Configure Flask-Login settings
login_manager.login_view = 'staff_login'
login_manager.login_message = u"Please log in to access this page."
login_manager.login_message_category = "info"

# --- ML Model Loading ---
model = None
try:
    if not os.path.exists(MODEL_PATH):
        app.logger.error(f"Model file not found at specified path: {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    app.logger.info(f"ML Model loaded successfully from {MODEL_PATH}")
    # model.predict(np.random.rand(1, MODEL_INPUT_LENGTH, 1), verbose=0)
    # app.logger.info("Model test prediction successful.")
except Exception as e:
    
    app.logger.critical(f"FAILED TO LOAD ML MODEL: {e}", exc_info=True)
    # raise RuntimeError("Critical ML Model failed to load. Application cannot start.") from e


# --- User Class and Authentication ---
class User(UserMixin):
    
    def __init__(self, id, username=None, user_type=None):
        self.id = id
        self.username = username
        self.user_type = user_type # 'staff' or 'doctor'

    def get_type(self):
        return self.user_type

@login_manager.user_loader
def load_user(user_id):
    app.logger.debug(f"Attempting to load user with ID: {user_id}")
    is_doctor = str(user_id).startswith('DR-')
    table = "doctor" if is_doctor else "staff"
    id_column = "Doctor_ID" if is_doctor else "Staff_ID"
    username_column = "Username" if is_doctor else "StaffName"

    
    user_data = db_fetch_one(f"SELECT * FROM {table} WHERE {id_column} = %s", (user_id,))

    if user_data:
        user_obj = User(
            id=str(user_data[id_column]),
            username=user_data[username_column],
            user_type="doctor" if is_doctor else "staff"
        )
        app.logger.debug(f"User {user_id} loaded successfully as {user_obj.user_type}.")
        return user_obj
    else:
        app.logger.warning(f"User ID {user_id} not found in database during session load.")
        return None 
    
def doctor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        
        if not current_user.is_authenticated:
            flash("Please log in to access this page.", "info")
            return redirect(url_for('doctor_login', next=request.url)) 
        
        if current_user.user_type != "doctor":
            app.logger.warning(f"Unauthorized access attempt to doctor route by {current_user.user_type} user: {current_user.id}")
            flash("You do not have permission to view this page.", "danger")
            
            if current_user.user_type == 'staff':
                return redirect(url_for('patient_registration'))
            else: 
                abort(403) 
        
        return f(*args, **kwargs)
    return decorated_function



def db_execute(sql, params=(), commit=False):
    cursor = None 
    try:
        cursor = mysql.connection.cursor()
        app.logger.debug(f"Executing SQL: {cursor.mogrify(sql, params)}") 
        cursor.execute(sql, params)
        if commit:
            mysql.connection.commit()
            app.logger.debug("DB transaction committed.")
        return cursor 
    except MySQLdb.Error as db_err:
        
        app.logger.error(f"Database Error: {db_err}\nSQL: {cursor.mogrify(sql, params) if cursor else sql}", exc_info=True)
        if mysql.connection:
            mysql.connection.rollback() 
            app.logger.warning("DB transaction rolled back due to error.")
        raise
    finally:
        if cursor:
            cursor.close() 

def db_fetch_one(sql, params=()):
    
    cursor = None
    try:
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        app.logger.debug(f"Fetching one: {cursor.mogrify(sql, params)}")
        cursor.execute(sql, params)
        result = cursor.fetchone()
        return result
    except MySQLdb.Error as db_err:
        app.logger.error(f"Database Error: {db_err}\nSQL: {cursor.mogrify(sql, params) if cursor else sql}", exc_info=True)
        return None 
    finally:
        if cursor:
            cursor.close()

def db_fetch_all(sql, params=()):
    
    cursor = None
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        app.logger.debug(f"Fetching all: {cursor.mogrify(sql, params)}")
        cursor.execute(sql, params)
        results = cursor.fetchall()
        return results
    except MySQLdb.Error as db_err:
        app.logger.error(f"Database Error: {db_err}\nSQL: {cursor.mogrify(sql, params) if cursor else sql}", exc_info=True)
        return [] # Return empty list on error
    finally:
        if cursor:
            cursor.close()




def _format_datetime_helper(value, fmt, tz_name):
    
    if value is None: return "" 

    
    if not isinstance(value, datetime):
        try:
            
            value = datetime.strptime(str(value), '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            
            app.logger.warning(f"Filter received non-datetime/unparsable value: '{value}' (type: {type(value)})")
            return str(value)

    try:
        target_tz = pytz.timezone(tz_name)
    except pytz.UnknownTimeZoneError:
        app.logger.error(f"Invalid timezone '{tz_name}' specified in filter. Using UTC fallback.")
        target_tz = pytz.utc 

   
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        
        aware_value = pytz.utc.localize(value).astimezone(target_tz)
    else:
       
        aware_value = value.astimezone(target_tz)

    return aware_value.strftime(fmt)

@app.template_filter('format_reg_date')
def format_datetime_date(value, format="%d %B %Y", tz_name=TARGET_TIMEZONE):
    """ Format (e.g., 21 April 2025)"""
    return _format_datetime_helper(value, format, tz_name)

@app.template_filter('format_reg_time')
def format_datetime_time(value, format="%I:%M:%S %p", tz_name=TARGET_TIMEZONE):
    """Format  (e.g., 10:30:45 AM)"""
    return _format_datetime_helper(value, format, tz_name)


@app.context_processor
def utility_processor():
    return dict(
        pytz=pytz, 
        url_quote_plus=urllib.parse.quote_plus 
    )


# --- ECG Processing & ML Helpers ---

def load_ecg_sample(record_num_str):
    
    record_num = str(record_num_str).strip()
    if not record_num:
        flash("ECG Record number cannot be empty.", "danger")
        return None, None, None, None

    record_path = os.path.join(DATASET_PATH, record_num)
    app.logger.info(f"Attempting to load ECG record: {record_path}")

    if not os.path.exists(f"{record_path}.dat") or not os.path.exists(f"{record_path}.hea"):
        msg = f"Required ECG file (.dat or .hea) not found for record '{record_num}'. Check path."
        app.logger.error(msg)
        flash(msg, "danger")
        return None, None, None, None

    try:
        record = wfdb.rdrecord(record_path)
        if record.p_signal is None or record.p_signal.shape[1] == 0:
            raise ValueError("No signal data found in record file.")

        signal = record.p_signal[:, 0]
        fs = record.fs
        symbols, samples = [], []
        try:
            annotation = wfdb.rdann(record_path, 'atr')
            symbols = annotation.symbol
            samples = annotation.sample
        except FileNotFoundError:
            app.logger.info(f"Annotation file (.atr) optional, not found for record {record_num}.")
        except Exception as e_ann:
            app.logger.warning(f"Error reading annotations for {record_num}: {e_ann}")

        app.logger.info(f"Successfully loaded record {record_num} (fs={fs}, {len(signal)} samples).")
        return signal, fs, symbols, samples

    except ValueError as ve: 
         app.logger.error(f"Value error loading record {record_num}: {ve}")
         flash(f"Error in ECG file data for record '{record_num}'.", "danger")
         return None, None, None, None
    except Exception as e:
        app.logger.error(f"Unexpected error loading ECG record {record_num}: {e}", exc_info=True)
        flash(f"Failed to load ECG record '{record_num}'. An unexpected error occurred.", "danger")
        return None, None, None, None


def check_ecg_quality(ecg_signal, fs):
    
    if len(ecg_signal) < fs: return False, 
    if np.max(ecg_signal) - np.min(ecg_signal) < 0.1: return False, 
    return True,


def butterworth_filter(signal, cutoff=50, fs=360, order=4):
    
    try:
        nyquist = 0.5 * fs
        if cutoff >= nyquist:
            app.logger.warning(f"Filter cutoff {cutoff}Hz >= Nyquist {nyquist}Hz. Skipping filter.")
            return signal
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)
    except Exception as e:
        app.logger.error(f"Error during Butterworth filtering: {e}", exc_info=True)
        return signal 

def detect_r_peaks(ecg_signal, fs):
    try:
        
        processed_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="pantompkins1985")
        
        
        r_peaks, _ = find_peaks(
            processed_ecg,
            height=np.percentile(processed_ecg, 75),
            distance=int(0.6 * fs)
        )
        return np.array(r_peaks)
    except Exception as e:
        return np.array([])


def compute_intervals(ecg_signal, r_peaks, fs):
    """Computes HR, approximates QT/PR, returns QRS peaks."""
    
    try:
        if r_peaks is None or len(r_peaks) < 2:
            return 0.0, 0.0, 0.0, np.array([])
        rr_intervals_sec = np.diff(r_peaks) / fs
        
        if len(rr_intervals_sec) == 0 or np.mean(rr_intervals_sec) <= 0:
             heart_rate = 0.0
        else:
             heart_rate = 60.0 / np.mean(rr_intervals_sec)

        
        qrs_peaks, _ = find_peaks(ecg_signal, height=np.percentile(ecg_signal, 90), distance=int(fs*0.06))

        
        qt_interval = (r_peaks[-1] - r_peaks[0]) / fs if len(r_peaks) > 1 else 0.0
        pr_interval = (r_peaks[1] - r_peaks[0]) / fs if len(r_peaks) > 1 else 0.0

        return float(heart_rate), float(qt_interval), float(pr_interval), np.array(qrs_peaks) 
    except Exception as e:
        app.logger.error(f"Error computing intervals: {e}", exc_info=True)
        return 0.0, 0.0, 0.0, np.array([])


def compute_framingham_risk(age, cholesterol, hdl, systolic_bp, smoker, diabetes):
    
    try:
        # Original formula
        score = (0.02 * age) + (0.03 * cholesterol) - (0.05 * hdl) + (0.04 * systolic_bp) + (0.2 * smoker) + (0.15 * diabetes)
        return float(min(max(score, 0), 30)) 
    except Exception as e:
        app.logger.error(f"Error computing Framingham risk: {e}", exc_info=True)
        return 0.0

def compute_grace_score(age, systolic_bp, heart_rate):
    
    try:
        # Original formula
        score = (0.1 * age) - (0.05 * systolic_bp) + (0.2 * heart_rate)
        return float(min(max(score, 0), 20)) # Ensure float and clamp
    except Exception as e:
        app.logger.error(f"Error computing GRACE score: {e}", exc_info=True)
        return 0.0


def center_pad_ecg(ecg_signal, target_length):
    
    if len(ecg_signal) >= target_length:
        return ecg_signal[:target_length]
    
    pad_before = (target_length - len(ecg_signal)) // 2
    pad_after = target_length - len(ecg_signal) - pad_before
    return np.pad(ecg_signal, (pad_before, pad_after), mode='reflect')

def check_heart_rate(heart_rate):
    
    normal_min = 60
    normal_max = 100

    if heart_rate is None or heart_rate <= 0:
        return "Undetermined"
    elif heart_rate < normal_min:
        return "Bradycardia (slow heart rate)"
    elif heart_rate > normal_max:
        return "Tachycardia (fast heart rate)"
    else:
       
        return "Normal heart rate"
    
def preprocess_ecg(ecg_signal, target_length=180):
    try:
        
        ecg_signal = np.array(ecg_signal, dtype=np.float32)
        
        
        mean_val = np.mean(ecg_signal)
        std_dev = np.std(ecg_signal)
        if std_dev < 1e-9: std_dev = 1e-9  
        ecg_normalized = (ecg_signal - mean_val) / std_dev
        
        
        if len(ecg_normalized) > target_length:
            processed_signal = ecg_normalized[:target_length]
        else:
            pad_width = target_length - len(ecg_normalized)
            processed_signal = np.pad(ecg_normalized, (0, pad_width), 
                                    mode='constant', constant_values=0.0)
        
        return np.reshape(processed_signal, (1, target_length, 1))
    except Exception as e:
        return np.zeros((1, target_length, 1), dtype=np.float32)

def generate_ecg_waveform_plot_json(record_num, samples_to_show, patient_id, ecg_signal_full, fs, r_peaks_full=None):
    
    
    app.logger.debug(f"Generating plot for {patient_id}, record {record_num}...")
    try:
        ecg_signal_full = np.array(ecg_signal_full)
        samples_to_show = min(samples_to_show, len(ecg_signal_full))
        if samples_to_show <= 0: return json.dumps({})
        ecg_display = ecg_signal_full[:samples_to_show]
        time_axis = np.linspace(0, samples_to_show / fs, samples_to_show, endpoint=False)

        # --- Refine R-Peak Locations ---
        # (Keep the peak refinement logic exactly as in the previous version)
        refined_visible_peaks_indices = []
        if r_peaks_full is not None and len(r_peaks_full) > 0:
           
            initial_visible_peaks = np.array(r_peaks_full, dtype=int)
            initial_visible_peaks = initial_visible_peaks[initial_visible_peaks < samples_to_show]
            refined_peaks_temp = []
            window_samples = int(0.05 * fs)
            for peak_idx in initial_visible_peaks:
                start = max(0, peak_idx - window_samples)
                end = min(len(ecg_signal_full), peak_idx + window_samples)
                window_slice = ecg_signal_full[start:end]
                if len(window_slice) > 0:
                    try:
                         exact_peak_in_window = np.argmax(window_slice)
                         refined_peak_abs_idx = start + exact_peak_in_window
                         refined_peaks_temp.append(refined_peak_abs_idx)
                    except Exception as refine_err: refined_peaks_temp.append(peak_idx) 
                else: refined_peaks_temp.append(peak_idx)
            refined_visible_peaks_indices = np.array(refined_peaks_temp, dtype=int)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_axis.tolist(), y=ecg_display.tolist(), mode='lines',
            line=dict(color='#000000', width=1.5), name='ECG Signal',
            hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.2f}mV<extra></extra>'
        ))

        if len(refined_visible_peaks_indices) > 0:
            valid_indices = refined_visible_peaks_indices[refined_visible_peaks_indices < len(time_axis)]
            peak_times = time_axis[valid_indices]
            peak_values = ecg_signal_full[valid_indices]
            fig.add_trace(go.Scatter(
                x=peak_times.tolist(), y=peak_values.tolist(), mode='markers',
                marker=dict(color='#e74c3c', size=10, symbol='diamond', line=dict(width=1, color='#333333')),
                name=f'R-peaks ({len(valid_indices)} detected)',
                hovertemplate='R-peak<br>Time: %{x:.3f}s<br>Amplitude: %{y:.2f}mV<extra></extra>'
            ))

       
        fig.update_layout(
            title=dict(text=f'ECG Analysis - Patient {patient_id} (Record {record_num})', x=0.5, font=dict(size=18)),
            xaxis_title='Time (s)',
            yaxis_title='Amplitude (mV)',
            height=450,
            margin=dict(l=60, r=40, t=80, b=60), 
            plot_bgcolor='rgba(255, 235, 235, 0.5)',
            paper_bgcolor='white',
            hovermode='closest',
            showlegend=True, 

            
            legend=dict(
                orientation="h", 
                yanchor="bottom",
                y=1.02, 
                xanchor="center",
                x=0.5 
            ),
            
            xaxis=dict(
                gridcolor='rgba(255, 99, 99, 0.5)', showgrid=True, gridwidth=1, zeroline=False, dtick=0.2,
                minor=dict(showgrid=True, gridcolor='rgba(255, 99, 99, 0.2)', gridwidth=0.5, dtick=0.04),
                rangeslider=dict(visible=True), tickformat='.2f'
            ),
            yaxis=dict(
                gridcolor='rgba(255, 99, 99, 0.2)', showgrid=True, gridwidth=1, zeroline=True,
                zerolinecolor='rgba(255, 99, 99, 0.5)', dtick=0.5,
                minor=dict(showgrid=True, gridcolor='rgba(255, 99, 99, 0.1)', gridwidth=0.5, dtick=0.1)
            )
        )
        app.logger.debug("Plot layout configured with legend outside.")

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    except Exception as e:
        app.logger.error(f"Failed to generate ECG plot for {patient_id}, record {record_num}: {e}", exc_info=True)
        return json.dumps({})

def align_ecg_to_r_peak(ecg_signal, r_peaks, target_length=180):
    if len(r_peaks) == 0:
        return center_pad_ecg(ecg_signal, target_length)
    
    
    main_r_peak = r_peaks[np.argmax([ecg_signal[i] for i in r_peaks])]
    
   
    start = max(0, main_r_peak - target_length // 2)
    end = start + target_length
    
    if end > len(ecg_signal):
        end = len(ecg_signal)
        start = end - target_length
    
    segment = ecg_signal[start:end]
    
    
    if len(segment) < target_length:
        pad_before = (target_length - len(segment)) // 2
        pad_after = target_length - len(segment) - pad_before
        segment = np.pad(segment, (pad_before, pad_after), mode='reflect')
    
    return segment


# --- Route Definitions ---
# --- Authentication Routes ---
@app.route("/", methods=["GET", "POST"])
def staff_login():
    """Staff Login Page."""
    if current_user.is_authenticated and current_user.user_type == 'staff':
        app.logger.debug("Staff already logged in, redirecting to registration.")
        return redirect(url_for('patient_registration'))

    if request.method == "POST":
        staff_id = request.form.get("Staff_ID", "").strip()
        password = request.form.get("password", "")

        if not staff_id or not password:
            flash("Staff ID and password are required.", "warning")
        else:
            staff_data = db_fetch_one("SELECT * FROM staff WHERE Staff_ID = %s", (staff_id,))
            if staff_data and bcrypt.check_password_hash(staff_data["Password"], password):
                user_obj = User(id=staff_data["Staff_ID"], username=staff_data["StaffName"], user_type="staff")
                login_user(user_obj)
                app.logger.info(f"Staff login successful: {staff_id}")
                next_page = request.args.get('next')
                return redirect(next_page or url_for("patient_registration"))
            else:
                flash("Invalid Staff ID or password.", "danger")
                app.logger.warning(f"Failed staff login attempt for ID: {staff_id}")

    return render_template("staff_login.html")

@app.route("/doctor_login", methods=["GET", "POST"])
def doctor_login():
    """Doctor Login Page."""
    if current_user.is_authenticated and current_user.user_type == 'doctor':
        app.logger.debug("Doctor already logged in, redirecting to input form.")
        return redirect(url_for('input_form'))

    if request.method == "POST":
        doctor_id = request.form.get("doctor_id", "").strip()
        password = request.form.get("password", "")

        if not doctor_id or not password:
            flash("Doctor ID and password are required.", "warning")
        else:
            doctor_data = db_fetch_one("SELECT * FROM doctor WHERE Doctor_ID = %s", (doctor_id,))
            if doctor_data and bcrypt.check_password_hash(doctor_data["Password"], password):
                user_obj = User(id=str(doctor_data["Doctor_ID"]), username=doctor_data['Username'], user_type="doctor")
                login_user(user_obj)
                session['doctor_name'] = doctor_data['Username']
                app.logger.info(f"Doctor login successful: {doctor_id}")
                next_page = request.args.get('next')
                
                return redirect(next_page or url_for("input_form"))
            else:
                flash("Invalid Doctor ID or password.", "danger")
                app.logger.warning(f"Failed doctor login attempt for ID: {doctor_id}")

    return render_template("doctor_login.html")

@app.route("/logout")
@login_required
def logout():
    """Logs out the current user."""
    user_id = current_user.id
    user_type = current_user.user_type
    logout_user()
    session.pop('doctor_name', None) 
    app.logger.info(f"{user_type.capitalize()} user {user_id} logged out.")
    redirect_url = url_for('doctor_login') if user_type == 'doctor' else url_for('staff_login')
    return redirect(redirect_url)

# --- Doctor Routes ---
@app.route("/input_form", methods=["GET", "POST"])
@login_required
@doctor_required
def input_form():
    """Doctor dashboard: Search patient, view details and history."""
    doctor_name = session.get('doctor_name', getattr(current_user, 'username', 'Doctor'))
    patient = None
    medical_history = [] 

    if request.method == "POST":
        patient_id = request.form.get("Patient_ID", "").strip()
        if not re.match(r'^PT-\d{5}-\d{4}$', patient_id):
            flash("Invalid Patient ID format. Please use PT-xxxxx-YYYY.", "danger")
        else:
            app.logger.info(f"Doctor {current_user.id} searching for patient: {patient_id}")
            try:
                # --- MODIFIED SQL ---
                # Select all columns directly from patient_profile.
                # No need to join with staff table anymore for this purpose.
                sql_patient = """
                SELECT p.* 
                FROM patient_profile p
                WHERE p.Patient_ID = %s
                """
                # The result 'patient' dictionary will now contain the 'Staff_Username'
                # key directly from the patient_profile table.
                patient = db_fetch_one(sql_patient, (patient_id,))

                if patient:
                    
                    app.logger.info(f"Patient data fetched for {patient_id}: {patient}") 
                    app.logger.info(f"Value in Staff_Username column: {patient.get('Staff_Username')}") # Use .get for safety

                    # Fetch medical history entries for this patient
                    sql_history = "SELECT * FROM input WHERE Patient_ID = %s ORDER BY Generated_AT DESC"
                    medical_history = db_fetch_all(sql_history, (patient_id,))
                    app.logger.info(f"Found {len(medical_history)} medical history entries for patient {patient_id}.")
                
                else:
                    flash(f"Patient with ID '{patient_id}' not found in the system.", "warning")
                    app.logger.warning(f"Patient search failed for ID: {patient_id}")

            except Exception as e:
                app.logger.error(f"Error during patient search for {patient_id}: {e}", exc_info=True)
                flash("An error occurred while retrieving patient data. Please try again.", "danger")
 
    return render_template("input_form.html",
                           patient=patient,
                           medical_history=medical_history,
                           doctor_name=doctor_name)


@app.route("/add_medical_data/<patient_id>", methods=["POST"])
@login_required
@doctor_required 
def add_medical_data(patient_id):
    """Handles submission of basic medical data from input_form."""
    
    try:
        systolic_bp = request.form["systolic_bp"]
        cholesterol = request.form["cholesterol"]
        hdl = request.form["hdl"]
        # Basic type checks
        float(systolic_bp)
        float(cholesterol)
        float(hdl)
    except (KeyError, ValueError) as form_err:
        app.logger.error(f"Invalid form data submitted for add_medical_data {patient_id}: {form_err}")
        flash("Invalid medical data submitted. Please check values.", "danger")
        return redirect(url_for("input_form")) 

    smoker = 1 if request.form.get("smoker") == 'on' else 0 
    diabetic = 1 if request.form.get("diabetic") == 'on' else 0
    

    try:
       
        if not db_fetch_one("SELECT 1 FROM patient_profile WHERE Patient_ID = %s", (patient_id,)):
             flash(f"Patient {patient_id} not found. Cannot add medical data.", "danger")
             return redirect(url_for("input_form"))

        sql_insert = """
            INSERT INTO input
            (Patient_ID, Doctor_ID, Smoker, Alcoholic, Diabetic, Cholesterol, HDL,
             Blood_Pressure, Other_Issues, Generated_AT)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """
        params = (
            patient_id, current_user.id, smoker, 0, 
            diabetic, cholesterol, hdl, systolic_bp, "", 
        )
        db_execute(sql_insert, params, commit=True)
        app.logger.info(f"Added medical data for patient {patient_id} by doctor {current_user.id}")
    

    except Exception as e:
        app.logger.error(f"Error saving medical data for patient {patient_id}: {e}", exc_info=True)
        flash("An error occurred while saving medical data.", "danger")

    return redirect(url_for("input_form"))



@app.route("/automatic_analysis/<patient_id>", methods=["GET", "POST"])
@login_required
@doctor_required
def automatic_analysis(patient_id):
    """Complete ECG analysis with waveform generation and proper error handling"""
    
    # 1. Validate patient ID format
    if not re.match(r'^PT-\d{5}-\d{4}$', patient_id):
        flash("Invalid Patient ID format. Expected format: PT-12345-2024", "danger")
        return redirect(url_for("input_form"))

    # 2. Get patient data
    patient = db_fetch_one("""
        SELECT p.*, d.Username AS doctor_name 
        FROM patient_profile p
        LEFT JOIN doctor d ON p.Doctor_ID = d.Doctor_ID
        WHERE p.Patient_ID = %s
    """, (patient_id,))
    
    if not patient:
        flash(f"Patient {patient_id} not found in database", "danger")
        return redirect(url_for("input_form"))

    
    if request.method == "GET":
        return render_template("automatic_analysis.html", patient=patient)

    
    try:
        app.logger.info(f"try catch 1")
        record_num = request.form.get("record_num", "").strip()
        if not record_num:
            raise ValueError("ECG record number is required")
        
        age = int(request.form.get("age", patient.get("Age", 30)))
        cholesterol = float(request.form.get("cholesterol", 150))
        hdl = float(request.form.get("hdl", 40))
        systolic_bp = float(request.form.get("systolic_bp", 120))
        smoker = 1 if request.form.get("smoker") == "on" else 0
        diabetes = 1 if request.form.get("diabetes") == "on" else 0

        app.logger.info(f"try catch 2")
        if not (0 < age <= 120):
            raise ValueError("Age must be between 1-120 years")
        if not (0 < cholesterol <= 500):
            raise ValueError("Cholesterol must be between 0-500 mg/dL")
        if not (0 < hdl <= 100):
            raise ValueError("HDL must be between 0-100 mg/dL")
        if not (50 <= systolic_bp <= 250):
            raise ValueError("Systolic BP must be between 50-250 mmHg")

       
        ecg_signal, fs, _, _ = load_ecg_sample(record_num)
        if ecg_signal is None:
            raise ValueError(f"Could not load ECG record {record_num}")
        if len(ecg_signal) < 180:
            raise ValueError(f"ECG signal too short ({len(ecg_signal)} samples)")

        # Process ECG signal
        ecg_filtered = butterworth_filter(ecg_signal, fs=fs)
        r_peaks = detect_r_peaks(ecg_filtered, fs)
        
        # Calculate intervals
        heart_rate, qt_interval, pr_interval, _ = compute_intervals(ecg_filtered, r_peaks, fs)
        heart_rate_status = check_heart_rate(heart_rate)

        # Prepare model input
        model_input = preprocess_ecg(ecg_filtered)
        
        # Make prediction (handle model not loaded)
        if model is None:
            raise RuntimeError("ECG analysis model not loaded")
        app.logger.info(f"try catch 3")
        main_pred, vfib_pred = model.predict(model_input, verbose=0)
        main_pred = main_pred.flatten()
        vfib_prob = vfib_pred.flatten()[0] * 100

        # Interpret prediction
        pred_class_idx = np.argmax(main_pred)
        pred_class_name = CLASSES[pred_class_idx]['name']
        confidence = main_pred[pred_class_idx] * 100
        
        # Ventricular fibrillation override
        if vfib_prob > 50:
            pred_class_idx = 3  # VF index
            pred_class_name = CLASSES[3]['name']
            confidence = vfib_prob

        # Generate ECG waveform visualization
        waveform_data = generate_ecg_waveform_plot_json(
            record_num=record_num,
            samples_to_show=min(2000, len(ecg_filtered)),
            patient_id=patient_id,
            ecg_signal_full=ecg_filtered,
            fs=fs,
            r_peaks_full=r_peaks
        )
        
        # Verify waveform data structure
        if not isinstance(waveform_data, str):
            raise ValueError("ECG visualization generation failed")
        
        # Parse the JSON to verify it's valid
        try:
            plot_data = json.loads(waveform_data)
        except json.JSONDecodeError:
            raise ValueError("Invalid ECG plot data generated")

        # Prepare class probabilities
        class_probabilities = {}
        for i, (class_id, info) in enumerate(CLASSES.items()):
            prob = main_pred[i] * 100 if i != 3 else vfib_prob
            class_probabilities[class_id] = {
                'name': info['name'],
                'probability': float(prob),
                'color': info['color'],
                'weight': info['weight']
            }

        # Prepare result data
        result_data = {
            'patient': patient,
            'predicted_class': pred_class_name,
            'predicted_class_id': CLASSES[pred_class_idx]['id'],
            'confidence': float(confidence),
            'class_probabilities': class_probabilities,
            'heart_rate': float(heart_rate),
            'heart_rate_status': heart_rate_status,
            'qt_interval': float(qt_interval),
            'pr_interval': float(pr_interval),
            'framingham_risk': compute_framingham_risk(age, cholesterol, hdl, systolic_bp, smoker, diabetes),
            'grace_score': compute_grace_score(age, systolic_bp, heart_rate),
            'record_num': record_num,
            'all_beats_count': len(r_peaks) if r_peaks is not None else 0,
            'plot_json': waveform_data,
            'samples_to_show': min(2000, len(ecg_filtered)),
            'age': age,
            'cholesterol': cholesterol,
            'hdl': hdl,
            'systolic_bp': systolic_bp,
            'smoker': smoker,
            'diabetes': diabetes,
            'doctor_name': patient.get('doctor_name', 'N/A')
        }

        return render_template("result.html", **result_data)

    except ValueError as ve:
        flash(f"Input validation error: {str(ve)}", "warning")
        app.logger.warning(f"Input validation failed for patient {patient_id}: {str(ve)}")
    except RuntimeError as runtime_err:  # Changed from 're' to 'runtime_err'
        flash(f"System error: {str(runtime_err)}", "danger")
        app.logger.error(f"System error during analysis for {patient_id}: {str(runtime_err)}", exc_info=True)
    except Exception as e:
        flash("An unexpected error occurred during ECG analysis", "danger")
        app.logger.error(f"Unexpected error analyzing {patient_id}: {str(e)}", exc_info=True)
    
    return redirect(url_for("automatic_analysis", patient_id=patient_id))

# --- Report Generation Route ---
# Applying previous fixes for JSON parameter encoding
@app.route('/save_ecg_image', methods=['POST'])
@login_required
def save_ecg_image():
    """Saves the ECG plot image sent from the frontend (Original Logic)."""
    
    try:
        
        os.makedirs(ECG_IMAGE_DIR, exist_ok=True) 

        if not request.is_json:
            app.logger.error("Save ECG image request received without JSON data.")
            return jsonify({'success': False, 'message': 'Missing JSON in request'}), 400

        data = request.get_json()
        app.logger.debug(f"Received data for image save: {data.keys()}") 

        
        required_fields = ['image_data', 'patient_id', 'record_num']
        if not all(field in data for field in required_fields):
            app.logger.error(f"Missing required fields in image save request. Needed: {required_fields}, Got: {list(data.keys())}")
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400

        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
       
        filename = f"ecg_{data['patient_id']}_{data['record_num']}_{timestamp}.png"
        filepath = os.path.join(ECG_IMAGE_DIR, filename) 

        
        try:
            # Handle data URI format if present
            img_data_str = data['image_data']
            img_data = img_data_str.split('base64,')[1] if 'base64,' in img_data_str else img_data_str

            with open(filepath, 'wb') as f:
                f.write(base64.b64decode(img_data))

            app.logger.info(f"ECG Image saved: {filepath}")
            # Return the URL path using url_for for correctness
            relative_path = url_for('static', filename=f'ecg_images/{filename}')
            return jsonify({
                'success': True,
                'path': relative_path # Return the URL path
            })

        except (TypeError, ValueError) as decode_err:
             app.logger.error(f"Error decoding image data for save: {decode_err}")
             return jsonify({'success': False, 'message': 'Invalid image data received.'}), 400
        except Exception as e:
            app.logger.error(f"Error writing image file {filepath}: {e}", exc_info=True)
            return jsonify({'success': False, 'message': f"Failed to save image file: {str(e)}"}), 500

    except Exception as e:
        # Catch errors in request handling/validation
        app.logger.error(f"Server error in /save_ecg_image route: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f"Server error: {str(e)}"}), 500

# --- Report Generation Route ---
@app.route('/generate_report/<patient_id>')
@login_required
def generate_report(patient_id):
    try:
        verify_directories()
        
        required_params = ['predicted_class', 'record_num', 'heart_rate']
        for param in required_params:
            if not request.args.get(param):
                flash(f"Missing required parameter: {param}", "danger")
                return redirect(url_for('automatic_analysis', patient_id=patient_id))

      
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        try:
            cursor.execute("""
                SELECT p.*, d.Username AS doctor_name, d.Doctor_ID
                FROM patient_profile p
                LEFT JOIN doctor d ON p.Doctor_ID = d.Doctor_ID
                WHERE p.Patient_ID = %s
            """, (patient_id,))
            patient = cursor.fetchone()
            
            if not patient:
                flash("Patient not found", "danger")
                return redirect(url_for("input_form"))
            
            
            doctor_id = patient.get('Doctor_ID')
            doctor_name = patient.get('doctor_name', "Not assigned")
            
        except Exception as db_error:
            print(f"Database error: {db_error}")
            doctor_name = "Not available"
            doctor_id = None
        finally:
            cursor.close()

        
        report_date = datetime.now()
        report_id = f"ECG-{report_date.strftime('%Y%m%d')}-{patient_id}"
        
        data = {
            'record_num': request.args.get('record_num'),
            'predicted_class': request.args.get('predicted_class'),
            'confidence': float(request.args.get('confidence', 0)),
            'heart_rate': float(request.args.get('heart_rate', 0)),
            'qt_interval': float(request.args.get('qt_interval', 0)),
            'pr_interval': float(request.args.get('pr_interval', 0)),
            'framingham_risk': float(request.args.get('framingham_risk', 0)),
            'grace_score': float(request.args.get('grace_score', 0)),
            'systolic_bp': float(request.args.get('systolic_bp', 0)),
            'cholesterol': float(request.args.get('cholesterol', 0)),
            'hdl': float(request.args.get('hdl', 0)),
            'smoker': request.args.get('smoker', '0') == '1',
            'diabetes': request.args.get('diabetes', '0') == '1',
            'all_beats_count': int(request.args.get('all_beats_count', 0)),
            'ecg_image': request.args.get('ecg_image', ''),
            'doctor_name': doctor_name
        }

   
        class_probabilities = {}
        try:
            class_probs = request.args.get('class_probabilities')
            if class_probs:
                class_probabilities = json.loads(class_probs)
        except json.JSONDecodeError as e:
            print(f"Error parsing class probabilities: {e}")

        # Find ECG image path
        ecg_image_path = None
        static_dir = os.path.join(app.root_path, 'static', 'ecg_images')
        
        if data['ecg_image'] and os.path.exists(os.path.join(app.root_path, data['ecg_image'].lstrip('/'))):
            ecg_image_path = data['ecg_image']
        elif os.path.exists(static_dir):
            ecg_files = [f for f in os.listdir(static_dir) if f.startswith(f"ecg_{patient_id}_{data['record_num']}_")]
            if not ecg_files:
                ecg_files = [f for f in os.listdir(static_dir) if f.startswith(f"ecg_{patient_id}_")]
            if ecg_files:
                ecg_files.sort(reverse=True)
                ecg_image_path = f"/static/ecg_images/{ecg_files[0]}"

        
        try:
            cursor = mysql.connection.cursor()
            
         
            smoker = 1 if data['smoker'] else 0
            diabetes = 1 if data['diabetes'] else 0
            
            cursor.execute("""
                INSERT INTO ecg_reports (
                    report_id, patient_id, doctor_id, report_date, record_num,
                    predicted_class, confidence, heart_rate, qt_interval, pr_interval,
                    framingham_risk, grace_score, systolic_bp, cholesterol, hdl,
                    smoker, diabetes, all_beats_count, class_probabilities, ecg_image_path
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
            """, (
                report_id, patient_id, doctor_id, report_date, data['record_num'],
                data['predicted_class'], data['confidence'], data['heart_rate'], 
                data['qt_interval'], data['pr_interval'],
                data['framingham_risk'], data['grace_score'], data['systolic_bp'], 
                data['cholesterol'], data['hdl'],
                smoker, diabetes, data['all_beats_count'], 
                json.dumps(class_probabilities) if class_probabilities else None,
                ecg_image_path
            ))
            mysql.connection.commit()
            
        except Exception as e:
            mysql.connection.rollback()
            print(f"Database save error: {str(e)}")
            

        
        context = {
            'patient': patient,
            'doctor_name': doctor_name,
            'age': patient.get('Age', 'N/A'),
            'gender': patient.get('Gender', 'N/A'),
            'report_date': report_date,
            'current_year': report_date.year,
            'report_id': report_id,
            'ecg_image_path': ecg_image_path,
            'classes': CLASSES,
            **data,
            'class_probabilities': class_probabilities,
            'pdf_mode': True  
        }

        
        if request.args.get('download') == 'pdf':
           
            report_content = render_template('report.html', **context)
            
            
            if ecg_image_path:
                try:
                    import base64
                    image_path = os.path.join(app.root_path, ecg_image_path.lstrip('/'))
                    if os.path.exists(image_path):
                        with open(image_path, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        report_content = report_content.replace(
                            f'src="{ecg_image_path}"',
                            f'src="data:image/png;base64,{encoded_string}"'
                        )
                except Exception as e:
                    app.logger.error(f"Error encoding ECG image: {str(e)}")

           
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{context['report_id']}</title>
                <style>
                    @page {{
                        margin: 1.5cm;
                        size: A4;
                    }}
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.5;
                        color: #333;
                        margin: 0;
                        padding: 20px;
                        font-size: 12px;
                    }}
                    .report-container {{
                        max-width: 800px;
                        margin: 0 auto;
                    }}
                    img {{
                        max-width: 100%;
                        height: auto;
                    }}
                </style>
            </head>
            <body>
                {report_content}
            </body>
            </html>
            """

            pdf = BytesIO()
            
            
            try:
                from weasyprint import HTML
                HTML(string=html).write_pdf(pdf)
            except ImportError:
                # Fallback to xhtml2pdf
                pisa_status = pisa.CreatePDF(
                    html,
                    dest=pdf,
                    encoding='UTF-8'
                )
                if pisa_status.err:
                    raise Exception(f"PDF generation error: {pisa_status.err}")
            
            pdf.seek(0)
            response = make_response(pdf.getvalue())
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = (
                f'attachment; filename={context["report_id"]}.pdf'
            )
            return response

        context['pdf_mode'] = False
        return render_template('report.html', **context)
        
    except Exception as e:
        app.logger.error(f"Report generation error: {str(e)}\n{traceback.format_exc()}")
        
        return redirect(url_for('automatic_analysis', patient_id=patient_id))



def verify_directories():
    required_dirs = [
        os.path.join(app.static_folder, 'ecg_images')
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Verified directory: {directory}")

def _generate_new_patient_id():
    """Generates next patient ID."""
    
    current_year = datetime.now().year
    try:
        sql = """
            SELECT Patient_ID FROM patient_profile
            WHERE Patient_ID LIKE %s
            ORDER BY CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(Patient_ID, '-', 2), '-', -1) AS UNSIGNED) DESC
            LIMIT 1
        """
        param = f'PT-%-{current_year}'
        last_patient = db_fetch_one(sql, (param,))
        if last_patient and last_patient.get('Patient_ID'):
            try:
                last_num = int(last_patient['Patient_ID'].split('-')[1])
                new_num = last_num + 1
            except (IndexError, ValueError):
                 app.logger.warning(f"Could not parse last patient ID: {last_patient['Patient_ID']}. Starting sequence.")
                 new_num = 10001
        else:
            new_num = 10001
        return f"PT-{new_num}-{current_year}"
    except Exception as e:
        app.logger.error(f"Failed to generate new Patient ID: {e}", exc_info=True)
        raise RuntimeError("Could not generate Patient ID") from e

# --- Validation Function ---
def _validate_patient_form(form, editing_id=None):
    """
    email duplication check:
    """
    errors = {}
    name = form.get('Patient_Name', '').strip()
    age_str = form.get('Age', '').strip()
    gender = form.get('Gender', '')
    address = form.get('Address', '').strip()
    email = form.get('Email_ID', '').strip().lower()
    phone = form.get('Personal_Contact', '').strip()
    emergency_phone = form.get('Emergency_Contact', '').strip()
    doctor_id = form.get('Doctor_ID', '')


    if not name: errors['Patient_Name'] = "Patient name needed."
    if not address: errors['Address'] = "Address needed."
    if not gender: errors['Gender'] = "Select a gender."
    if not doctor_id: errors['Doctor_ID'] = "Assign a doctor."
    if not age_str: errors['Age'] = "Age is required."
    if not email: errors['Email_ID'] = "Email is required."
    if not phone: errors['Personal_Contact'] = "Phone contact needed."
    if not emergency_phone: errors['Emergency_Contact'] = "Emergency contact needed."


    if age_str:
        try:
            age = int(age_str)
            if not (0 < age <= 150):
                errors['Age'] = "Age should be between 1 and 150."
        except (ValueError, TypeError): 
            errors['Age'] = "Age must be a number."

  
    if email:
 
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            if 'Email_ID' not in errors: 
                 errors['Email_ID'] = "Looks like an invalid email format."
        else:
            
            query = "SELECT Patient_ID FROM patient_profile WHERE LOWER(Email_ID) = %s"
            params = (email,)

            
            if editing_id:
                query += " AND Patient_ID != %s"
                params += (editing_id,)

            try:
               
                duplicate = db_fetch_one(query, params)
                if duplicate:
                    errors['Email_ID'] = "This email is already used by another patient."
                    # app.logger.warning(f"Duplicate email found: '{email}' conflicts with PID {duplicate.get('Patient_ID', '?')}")
            except Exception as e:
                app.logger.error(f"DB error checking email duplicate: {e}")
               
                pass 

    if phone:
        if not re.match(r'^\d{10}$', phone): 
            if 'Personal_Contact' not in errors:
                 errors['Personal_Contact'] = "Phone should be 10 digits."

    if emergency_phone:
        if not re.match(r'^\d{10}$', emergency_phone):
             if 'Emergency_Contact' not in errors:
                  errors['Emergency_Contact'] = "Emergency phone should be 10 digits."
   
        elif phone and phone == emergency_phone:
            if 'Personal_Contact' not in errors and 'Emergency_Contact' not in errors:
                errors['Emergency_Contact'] = "Emergency contact must be different."


    if doctor_id:
        try:
   
            doc = db_fetch_one("SELECT 1 FROM doctor WHERE Doctor_ID = %s", (doctor_id,))
            if not doc:
                 if 'Doctor_ID' not in errors:
                     errors['Doctor_ID'] = "Selected Doctor ID seems invalid."
        except Exception as e:
             app.logger.error(f"DB error checking doctor ID: {e}")
             # errors['Doctor_ID'] = "Server error checking doctor."
             pass


    if errors:
        app.logger.debug(f"Validation failed (editing ID: {editing_id}): {errors}")
    else:
         app.logger.debug(f"Validation OK (editing ID: {editing_id}).")

    return errors


# --- Patient Registration Route ---

@app.route("/patient_registration", methods=["GET", "POST"])
@login_required 
def patient_registration():
    
    
    if current_user.user_type != 'staff':
        flash("Not allowed! Staff only.", "warning")
        return redirect(url_for('staff_login'))

    patient_details = None 
    errors = {}
    form_data = request.form if request.method == 'POST' else {} 
    doctors_list = [] 
    try:
        
        doctors_list = db_fetch_all("SELECT Doctor_ID, Username FROM doctor ORDER BY Username")
    except Exception as e:
         app.logger.error(f"Error getting doctors list: {e}")
         flash("Couldn't load doctor list.", "danger")

 
    if request.method == "POST":
   
        errors = _validate_patient_form(request.form, editing_id=None)

        if not errors: #
            try:
               
                new_id = _generate_new_patient_id()
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sql = """
                    INSERT INTO patient_profile
                           (Patient_ID, Patient_Name, Age, Gender, Address, Email_ID, Personal_Contact,
                            Emergency_Contact, Doctor_ID, Created_At, Staff_Username)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
             
                data_tuple = (
                    new_id, request.form['Patient_Name'].strip(), int(request.form['Age']),
                    request.form['Gender'], request.form['Address'].strip(),
                    request.form['Email_ID'].strip().lower(), # Store lowercase
                    request.form['Personal_Contact'].strip(), request.form['Emergency_Contact'].strip(),
                    request.form['Doctor_ID'], ts,
                    current_user.username 
                )
                
                db_execute(sql, data_tuple, commit=True)

                flash(f"Patient '{request.form['Patient_Name']}' registered OK! ID: {new_id}", "success")
                app.logger.info(f"Patient {new_id} added by staff {current_user.id} ({current_user.username})")

               
                patient_details = db_fetch_one("SELECT * FROM patient_profile WHERE Patient_ID = %s", (new_id,))
                form_data = {}

            except Exception as e:
                app.logger.error(f"DB error saving new patient: {e}", exc_info=True)
                flash("Error saving patient to database!", "danger")
                form_data = request.form # Keep submitted data on DB error

        else: 
            flash("Please fix the errors shown below.", "warning")
            form_data = request.form # Keep submitted data in form

    
    # Needs patient (if just added), errors, form data, doctors list
    return render_template("patient_registration.html",
                           patient=patient_details,
                           errors=errors,
                           form_data=form_data,
                           doctors=doctors_list)


# --- Edit Patient Route ---

@app.route("/edit_patient/<patient_id>", methods=["GET", "POST"])
@login_required 
def edit_patient(patient_id):
    """Page for staff to edit existing patient details."""
    
    if current_user.user_type != 'staff':
         flash("Not allowed! Staff only.", "warning")
         return redirect(url_for('staff_login'))

    errors = {}
    doctors_list = [] 
    try:
       
        doctors_list = db_fetch_all("SELECT Doctor_ID, Username FROM doctor ORDER BY Username")
    except Exception as e:
         app.logger.error(f"Error getting doctors list for edit page: {e}")
         flash("Couldn't load doctor list.", "danger")


    data_for_form = None

    if request.method == "POST":
      
        errors = _validate_patient_form(request.form, editing_id=patient_id)

        if not errors: 
            try:
               
                sql = """
                    UPDATE patient_profile SET
                        Patient_Name = %s, Age = %s, Gender = %s, Address = %s, Email_ID = %s,
                        Personal_Contact = %s, Emergency_Contact = %s, Doctor_ID = %s
                    WHERE Patient_ID = %s
                """
               
                data_tuple = (
                    request.form['Patient_Name'].strip(), int(request.form['Age']),
                    request.form['Gender'], request.form['Address'].strip(),
                    request.form['Email_ID'].strip().lower(), # Store lowercase
                    request.form['Personal_Contact'].strip(), request.form['Emergency_Contact'].strip(),
                    request.form['Doctor_ID'],
                    patient_id 
                )
             
                db_execute(sql, data_tuple, commit=True)

                app.logger.info(f"Patient {patient_id} updated by staff {current_user.id} ({current_user.username})")
                flash("Patient info updated.", "success")

        
                # NOTE: Redirecting back to edit page is better UX:
                # return redirect(url_for('edit_patient', patient_id=patient_id))
                return redirect(url_for('patient_registration'))

            except Exception as e:
                 app.logger.error(f"DB error updating patient {patient_id}: {e}", exc_info=True)
                 flash("Error saving changes to database!", "danger")
                
                 data_for_form = request.form
        else:
            flash("Please fix the errors shown below.", "warning")
           
            data_for_form = request.form

   
    if data_for_form is None: 
        try:
          
            db_data = db_fetch_one("SELECT * FROM patient_profile WHERE Patient_ID = %s", (patient_id,))
            if not db_data:
                flash(f"Patient {patient_id} not found.", "danger")
                return redirect(url_for('patient_registration')) 
            data_for_form = db_data 
        except Exception as e:
            app.logger.error(f"Error fetching patient {patient_id} for edit: {e}")
            flash("Error loading patient data.", "danger")
            return redirect(url_for('patient_registration')) 

    
    patient_data_for_template = dict(data_for_form)
    patient_data_for_template['Patient_ID'] = patient_id 

    
    return render_template("edit_patient.html",
                           patient=patient_data_for_template, 
                           patient_id=patient_id,
                           errors=errors,
                           doctors=doctors_list) 

# --- Utility Routes ---
@app.route("/debug")
def debug():
    """Debug route showing session info (REMOVE IN PRODUCTION)."""
    app.logger.info("Accessing debug route.")
    return jsonify({
        "user_id": session.get("_user_id"),
        "is_authenticated": current_user.is_authenticated,
        "user_type": getattr(current_user, 'user_type', None),
        "username": getattr(current_user, 'username', None),
        "session_contents": dict(session)
    })

@app.route('/debug_static_dir')
def debug_static_dir_check():
    """Check static dir status (REMOVE IN PRODUCTION)."""
    static_dir = ECG_IMAGE_DIR
    exists = os.path.exists(static_dir)
    is_writable = False
    error_msg = None
    if exists:
        test_file = os.path.join(static_dir, f'write_test_{uuid.uuid4()}.tmp')
        try:
            with open(test_file, 'w') as f: f.write("test")
            os.remove(test_file)
            is_writable = True
            app.logger.debug(f"Static image directory {static_dir} is writable.")
        except Exception as e:
            error_msg = str(e)
            app.logger.warning(f"Static image directory {static_dir} is NOT writable: {e}")

    return jsonify({
        'static_dir': static_dir,
        'exists': exists,
        'writable': is_writable,
        'write_error': error_msg,
    })


# --- Main Application Runner ---
if __name__ == "__main__":
    
    log_format = '%(asctime)s [%(levelname)s] %(name)s (%(module)s:%(lineno)d) - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    
    app.logger.setLevel(logging.INFO)

    
    if not PDF_GENERATION_AVAILABLE:
        app.logger.warning("PDF generation disabled (xhtml2pdf not found).")
    if model is None:
        app.logger.warning("ML Model was not loaded. Analysis features will fail.")

    app.logger.info("Starting Flask ECG Application...")
    
    app.run(debug=True)  