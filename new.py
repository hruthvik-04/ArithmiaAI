import plotly.graph_objects as go
import uuid
import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_mysqldb import MySQL
import MySQLdb.cursors
import numpy as np
import wfdb
from tensorflow.keras.models import load_model  # type: ignore
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from scipy.signal import butter, filtfilt, resample
import neurokit2 as nk
from scipy.signal import find_peaks
import pandas as pd
from reportlab.lib.utils import ImageReader

app = Flask(__name__)

app.secret_key = "5010dfae019e413f06691431b2e3ba82bbb456c661b0d27332a4dbd5bbd36bd8"
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "452003@hrX"
app.config["MYSQL_DB"] = "hospital_ecg_db"
app.config["MYSQL_CURSORCLASS"] = "DictCursor"

# Initialize extensions
mysql = MySQL(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "doctor_login"

# Constants
DATASET_PATH = "mit-bih-arrhythmia-database-1.0.0/"
MODEL_PATH = os.path.join('model', 'model.hdf5')
CLASSES = ["Normal", "Atrial Fibrillation", "Ventricular Tachycardia", "Heart Block", "Other1", "Other2"]

# Load model
model = load_model(MODEL_PATH)

# Check and compile if needed
if not hasattr(model, 'optimizer') or model.optimizer is None:
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# User Model
class User(UserMixin):
    def __init__(self, id, username=None, user_type=None):
        self.id = id
        self.username = username
        self.user_type = user_type
    
    def get_type(self):
        return self.user_type

@login_manager.user_loader
def load_user(user_id):
    print(f"Loading user with ID: {user_id}")
    
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    try:
        if user_id.startswith('DR-'):
            cursor.execute("SELECT * FROM doctor WHERE Doctor_ID = %s", (user_id,))
            doctor = cursor.fetchone()
            if doctor:
                print(f"Loaded doctor user: {doctor['Doctor_ID']}")
                return User(
                    id=str(doctor["Doctor_ID"]),
                    username=doctor['Username'],
                    user_type="doctor"
                )
        else:
            cursor.execute("SELECT * FROM staff WHERE Staff_Username = %s", (user_id,))
            staff = cursor.fetchone()
            if staff:
                print(f"Loaded staff user: {staff['Staff_Username']}")
                return User(
                    id=staff["Staff_Username"],
                    username=staff["Staff_Username"],
                    user_type="staff"
                )
    except MySQLdb.Error as e:
        print(f"Database error: {e}")
    finally:
        cursor.close()
    
    print("No user found")
    return None

# ECG Processing Functions
def load_ecg_sample(record_num="100"):
    """
    Load an ECG sample from the MIT-BIH Arrhythmia Database.
    
    Args:
        record_num (str): The record number to load (e.g., "100").
    
    Returns:
        dict: A dictionary containing the ECG signal, sampling rate, and duration.
    """
    record_path = f"{DATASET_PATH}{record_num}"
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal
    fs = record.fs  # Original sampling rate
    
    # Handle multi-channel signals by selecting the first channel
    if len(signal.shape) > 1:
        original_signal = signal[:, 0]
    else:
        original_signal = signal
    
    # Downsample to a target rate of 200 Hz
    target_rate = 200
    if fs != target_rate:
        num_samples = int(len(original_signal) * target_rate / fs)
        downsampled_signal = resample(original_signal, num_samples)
    else:
        downsampled_signal = original_signal
    
    # Extract a 10-second segment
    segment_duration = 10  # 10 seconds
    segment_samples = target_rate * segment_duration
    
    # Ensure the signal is long enough
    if len(downsampled_signal) < segment_samples:
        # Pad with zeros if the signal is too short
        padded_signal = np.zeros(segment_samples)
        padded_signal[:len(downsampled_signal)] = downsampled_signal
        segment = padded_signal
    else:
        segment = downsampled_signal[:segment_samples]
    
    return {
        "sampling_rate": target_rate,
        "duration": segment_duration,
        "signal": segment.tolist()
    }

def generate_ecg_plot(signal, peaks=None, title="ECG Signal"):
    unique_id = str(uuid.uuid4())[:8]
    filename = f"ecg_plot_{unique_id}.png"
    path = os.path.join("static", filename).replace("\\", "/")
    
    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=signal, 
        mode='lines', 
        name='ECG Signal', 
        line=dict(color='black', width=1)  # Black line for ECG signal
    ))

    if peaks is not None:
        fig.add_trace(go.Scatter(
            x=peaks, 
            y=signal[peaks], 
            mode='markers', 
            name='R-peaks', 
            marker=dict(color='red', symbol='x', size=10)  # Red markers for R-peaks
        ))

    # Update layout for pink background, longer graph, and scrollbar
    fig.update_layout(
        title=dict(text=title, font=dict(color='black')),  # Black title
        xaxis_title="Time (samples)",
        yaxis_title="Amplitude (mV)",
        plot_bgcolor='pink',  # Pink background for the plot area
        paper_bgcolor='pink',  # Pink background for the entire figure
        showlegend=True,  # Show legend for R-peaks
        xaxis=dict(
            showgrid=True, 
            gridcolor='lightgray', 
            rangeslider=dict(visible=True),  # Add scrollbar
            title_font=dict(color='black'),  # Black x-axis title
            tickfont=dict(color='black')  # Black x-axis ticks
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='lightgray',
            title_font=dict(color='black'),  # Black y-axis title
            tickfont=dict(color='black')  # Black y-axis ticks
        ),
        width=2000,  # Make the graph longer by increasing width
        height=500   # Set a fixed height for the graph
    )

    # Save the figure as an image
    fig.write_image(path)
    return filename

def butterworth_filter(signal, cutoff=50, fs=360, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def detect_r_peaks(ecg_signal, fs):
    processed_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="pantompkins1985")
    _, r_peaks = nk.ecg_peaks(processed_ecg, sampling_rate=fs)
    return r_peaks["ECG_R_Peaks"]

def compute_intervals(ecg_signal, r_peaks, fs):
    rr_intervals = np.diff(r_peaks) / fs
    heart_rate = 60 / rr_intervals.mean()
    qrs_peaks, _ = find_peaks(ecg_signal, height=np.percentile(ecg_signal, 90), distance=fs*0.06)
    qt_interval = (r_peaks[-1] - r_peaks[0]) / fs
    pr_interval = (r_peaks[1] - r_peaks[0]) / fs
    return heart_rate, qt_interval, pr_interval, qrs_peaks

def preprocess_ecg(ecg_signal):
    # Normalize the signal
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
    
    # Ensure the signal has the correct number of samples
    if len(ecg_signal) < 4096:
        # Pad with zeros if shorter than 4096
        padded = np.zeros(4096)
        padded[:len(ecg_signal)] = ecg_signal
        ecg_signal = padded
    elif len(ecg_signal) > 4096:
        # Truncate if longer than 4096
        ecg_signal = ecg_signal[:4096]
    
    # For automatic analysis (12-lead), ensure shape is (4096, 12)
    if ecg_signal.ndim == 2 and ecg_signal.shape[1] == 12:
        # Already 12-lead data
        ecg_resized = ecg_signal
    else:
        # For manual analysis (1-lead), duplicate across all 12 channels
        ecg_resized = np.repeat(ecg_signal[:, np.newaxis], 12, axis=1)
    
    # Add batch dimension
    return np.expand_dims(ecg_resized, axis=0)

def compute_framingham_risk(age, cholesterol, hdl, systolic_bp, smoker, diabetes):
    score = (0.02 * age) + (0.03 * cholesterol) - (0.05 * hdl) + (0.04 * systolic_bp) + (0.2 * smoker) + (0.15 * diabetes)
    return min(max(score, 0), 30)

def compute_grace_score(age, systolic_bp, heart_rate):
    score = (0.1 * age) - (0.05 * systolic_bp) + (0.2 * heart_rate)
    return min(max(score, 0), 20)

def generate_pdf(predicted_class, framingham_risk, grace_score, heart_rate, qt_interval, pr_interval, ecg_filename):
    unique_id = str(uuid.uuid4())[:8]
    pdf_filename = f"ECG_Report_{unique_id}.pdf"
    pdf_path = os.path.join("static", pdf_filename).replace("\\", "/")
    
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "ECG Analysis Report")
    c.setFont("Helvetica", 12)
    
    # Add report content
    y_position = 700
    c.drawString(100, y_position, f"Prediction: {predicted_class}")
    y_position -= 30
    c.drawString(100, y_position, f"Framingham Risk: {framingham_risk:.2f}%")
    y_position -= 30
    c.drawString(100, y_position, f"GRACE Score: {grace_score:.2f}%")
    y_position -= 30
    c.drawString(100, y_position, f"Heart Rate: {heart_rate:.2f} BPM")
    
    # Add ECG plot
    plot_path = os.path.join("static", ecg_filename)
    if os.path.exists(plot_path):
        c.drawImage(plot_path, 100, 400, width=400, height=200)
    
    c.save()
    return pdf_filename

def simulate_ecg_signal(p_peak, qrs_interval, qt_interval, pr_interval, heart_rate, fs=360):
    # Generate a signal with exactly 4096 samples
    duration = 4096 / fs  # Calculate duration based on fs and desired samples
    t = np.linspace(0, duration, 4096)
    
    # Simulate basic ECG components
    heartbeat = np.zeros_like(t)
    for i in range(len(t)):
        if i % int(fs * 60 / heart_rate) < int(fs * 0.2):  # QRS complex
            heartbeat[i] = 1.0
        elif i % int(fs * 60 / heart_rate) < int(fs * 0.4):  # T wave
            heartbeat[i] = 0.5
    
    ecg_signal = np.zeros((4096, 12))
    ecg_signal[:, 0] = heartbeat * p_peak
    return ecg_signal

@app.route("/")
def home():
    return redirect(url_for("doctor_login"))

@app.route("/doctor_login", methods=["GET", "POST"])
def doctor_login():
    if request.method == "POST":
        doctor_id = request.form.get("doctor_id")
        password = request.form.get("password")
        
        if not doctor_id or not password:
            flash("Please enter both doctor ID and password", "danger")
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
                flash("Login successful!", "success")
                return redirect(url_for("automatic_analysis"))
            else:
                flash("Invalid credentials", "danger")
        except MySQLdb.Error as e:
            flash(f"Database error: {str(e)}", "danger")
        finally:
            cursor.close()
    
    return render_template("doctor_login.html")
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        if not all([username, email, password]):
            return jsonify({"status": "error", "message": "All fields required"}), 400

        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
        
        try:
            cursor = mysql.connection.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE username = %s OR email = %s",
                (username, email)
            )
            if cursor.fetchone():
                return jsonify({"status": "error", "message": "Username or email already exists"}), 400
            
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                (username, email, hashed_password)
            )
            mysql.connection.commit()
            cursor.close()
            return jsonify({"status": "success", "message": "Registration successful"}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("doctor_login"))

@app.route("/automatic_analysis", methods=["GET", "POST"])
@login_required
def automatic_analysis():
    if request.method == "POST":
        try:
            # Process form data
            record_num = request.form["record_num"]
            age = int(request.form["age"])
            cholesterol = int(request.form["cholesterol"])
            hdl = int(request.form["hdl"])
            systolic_bp = int(request.form["systolic_bp"])
            smoker = "smoker" in request.form
            diabetes = "diabetes" in request.form

            # Load ECG sample
            ecg_data = load_ecg_sample(record_num)
            ecg_signal = np.array(ecg_data["signal"])
            fs = ecg_data["sampling_rate"]

            # Continue with filtering, peak detection, etc.
            ecg_signal = butterworth_filter(ecg_signal, fs=fs)
            r_peaks = detect_r_peaks(ecg_signal, fs)
            heart_rate, qt_interval, pr_interval, qrs_peaks = compute_intervals(ecg_signal, r_peaks, fs)

            # Generate plot
            ecg_filename = generate_ecg_plot(ecg_signal, r_peaks, "ECG with Anomalies")
            
            # Get model prediction
            prediction = model.predict(preprocess_ecg(ecg_signal))
            predicted_class = CLASSES[np.argmax(prediction)]
            
            # Generate PDF
            pdf_filename = generate_pdf(
                predicted_class,
                compute_framingham_risk(age, cholesterol, hdl, systolic_bp, smoker, diabetes),
                compute_grace_score(age, systolic_bp, heart_rate),
                heart_rate,
                qt_interval,
                pr_interval,
                ecg_filename
            )

            return render_template("result.html",
                predicted_class=predicted_class,
                framingham_risk=compute_framingham_risk(age, cholesterol, hdl, systolic_bp, smoker, diabetes),
                grace_score=compute_grace_score(age, systolic_bp, heart_rate),
                heart_rate=heart_rate,
                qt_interval=qt_interval,
                pr_interval=pr_interval,
                ecg_filename=ecg_filename,
                pdf_filename=pdf_filename
            )

        except Exception as e:
            app.logger.error(f"Automatic analysis error: {str(e)}")
            flash(f"Analysis failed: {str(e)}", "danger")
            return redirect(url_for("automatic_analysis"))

    return render_template("automatic_analysis.html")

@app.route("/manual_analysis", methods=["GET", "POST"])
@login_required
def manual_analysis():
    if request.method == "POST":
        try:
            # Process form data
            params = {k: float(request.form[k]) for k in ['p_peak', 'qrs_interval', 'qt_interval', 'pr_interval', 'heart_rate']}
            demographics = {k: int(request.form[k]) for k in ['age', 'cholesterol', 'hdl', 'systolic_bp']}
            
            # Generate ECG signal
            ecg_signal = simulate_ecg_signal(**params, fs=360)
            
            # Generate plot
            ecg_filename = generate_ecg_plot(ecg_signal[:, 0], title="Simulated ECG")
            
            # Get model prediction
            prediction = model.predict(preprocess_ecg(ecg_signal))
            predicted_class = CLASSES[np.argmax(prediction)]
            
            # Generate PDF
            pdf_filename = generate_pdf(
                predicted_class,
                compute_framingham_risk(**demographics, smoker="smoker" in request.form, diabetes="diabetes" in request.form),
                compute_grace_score(demographics['age'], demographics['systolic_bp'], params['heart_rate']),
                params['heart_rate'],
                params['qt_interval'],
                params['pr_interval'],
                ecg_filename
            )

            return render_template("result.html",
                predicted_class=predicted_class,
                framingham_risk=compute_framingham_risk(**demographics, smoker="smoker" in request.form, diabetes="diabetes" in request.form),
                grace_score=compute_grace_score(demographics['age'], demographics['systolic_bp'], params['heart_rate']),
                heart_rate=params['heart_rate'],
                qt_interval=params['qt_interval'],
                pr_interval=params['pr_interval'],
                ecg_filename=ecg_filename,
                pdf_filename=pdf_filename
            )

        except Exception as e:
            app.logger.error(f"Manual analysis error: {str(e)}")
            flash(f"Analysis failed: {str(e)}", "danger")
            return redirect(url_for("manual_analysis"))

    return render_template("manual_analysis.html")

@app.route("/download_report/<filename>")
@login_required
def download_report(filename):
    return send_from_directory(
        directory=os.path.join(app.root_path, 'static'),
        path=filename,
        as_attachment=True
    )

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=current_user.username)

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)

@app.context_processor
def inject_pytz():
   return dict(pytz=pytz)


# if __name__ == "__main__":
#     os.makedirs("static", exist_ok=True)
    
#     # Hardcoded doctor details
#     doctor_id = "DR-002-2024"
#     username = "HRUTHVIK"
#     password = "hru@123MIND"

#     # Ensure the application context is active
#     with app.app_context():
#         # Register the doctor implicitly
#         register_doctor_implicitly(doctor_id, username, password)

#     app.run(debug=True)
{% extends "base.html" %}
{% block title %}Analysis Result - ECG Analysis{% endblock %}

{% block content %}
<style>
    body {
        font-family: Arial, sans-serif;
        background-color: #889ba3;
        text-align: center;
        z-index: -1;
    }
    .container {
        width: 80%;
        margin: auto;
        padding: 20px;
        background-color: #eceff07b;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
    }
    h1 {
        font-size: 24px;
        text-decoration: underline;
    }
    .info-section {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
        padding: 10px;
        background-color: #4cc6efc8;
        border-radius: 5px;
    }
    .info-box {
        flex: 1;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        background-color: #fff;
        border-radius: 5px;
        margin: 0 10px;
    }
    .result-section {
        background-color: #6bd3f9ba;
        padding: 15px;
        margin: 15px 0;
        font-size: 20px;
        font-weight: bold;
        border-radius: 5px;
    }
    .risk-scores, .metrics {
        display: inline-block;
        width: 45%;
        vertical-align: top;
        text-align: left;
    }
    .risk-scores div, .metrics div {
        margin: 10px 0;
        padding: 10px;
        background-color: #fff;
        border-radius: 5px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .ecg-waveform {
        margin-top: 20px;
        background-color: #fff;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
    }
    .ecg-image {
        max-width: 100%;
        height: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    .button-container {
        margin-top: 20px;
    }
    .button {
        padding: 10px 20px;
        border-radius: 5px;
        text-decoration: none;
        font-size: 18px;
        font-weight: bold;
        display: inline-block;
        margin: 10px;
    }
    .btn-download {
        background-color: #28a745;
        color: white;
    }
    .btn-back {
        background-color: #6c757d;
        color: white;
    }
</style>

<div class="container">
    <h1>ECG ANALYSIS RESULTS</h1>

    <div class="info-section">
        <div class="info-box">
            <strong>NAME:</strong> 
            <span id="patientName">{{ patient.Patient_Name if patient else "Not Provided" }}</span>
        </div>
        <div class="info-box">
            <strong>Patient ID:</strong> 
            <span id="patientId">{{ patient.Patient_ID if patient else "Not Provided" }}</span>
        </div>
    </div>

    <div class="result-section">
        <strong>RESULT: {{ predicted_class }}</strong>
    </div>

    <div class="risk-scores">
        <strong>RISK SCORES</strong>
        <div>
            Framingham's score: 
            <span>{{ framingham_risk|round(2) }}%</span>
        </div>
        <div>
            GRACE score: 
            <span>{{ grace_score|round(2) }}%</span>
        </div>
    </div>

    <div class="metrics">
        <div>
            Heart Rate: 
            <span>{{ heart_rate|round(2) }} BPM</span>
        </div>
        <div>
            QT Interval: 
            <span>{{ qt_interval|round(3) }} s</span>
        </div>
        <div>
            PR Interval: 
            <span>{{ pr_interval|round(3) }} s</span>
        </div>
    </div>

    <div class="ecg-waveform">
        <strong>ECG WAVEFORM</strong>
        <div class="ecg-image-container">
            <img src="{{ url_for('static', filename=ecg_filename) }}" 
                 class="ecg-image" 
                 alt="ECG Waveform - {{ predicted_class }}">
        </div>
    </div>
    <div class="button-container">
        <a href="{{ url_for('download_report', filename=pdf_filename) }}" class="button btn-download">
            <i class="fas fa-file-pdf"></i> Download Report
        </a>
        <a href="{{ url_for('dashboard') }}" class="button btn-back">
            <i class="fas fa-arrow-left"></i> Back to Dashboard
        </a>
    </div>
</div>

<script>
    document.addEventListener("DOMContentLoaded", function () {
        let patientName = document.getElementById("patientName");
        let patientId = document.getElementById("patientId");

        // Ensure name and ID have consistent spacing
        if (patientName.innerText.trim() === "") {
            patientName.innerText = "Not Provided";
        }
        if (patientId.innerText.trim() === "") {
            patientId.innerText = "Not Provided";
        }
    });
</script>

{% endblock %}

{% extends "base.html" %}
{% block title %}Analysis Result - ECG Analysis{% endblock %}

{% block content %}
<style>
    body {
        font-family: Arial, sans-serif;
        background-color: #33BBCF;
        text-align: center;
        margin: 0;
        padding: 0;
    }
    .header {
        background-color: black;
        color: white;
        padding: 10px;
        font-size: 24px;
        font-weight: bold;
    }
    .result {
        background-color: lightgray;
        padding: 20px;
        font-size: 18px;
        font-weight: bold;
        height: 40px;
        text-align: left;
        padding-left: 20px;
    }
    .waveform-container {
        background-color: lightgray;
        height: 120px;
        margin: 10px;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        position: relative;
    }
    .waveform-text {
        position: absolute;
        top: 5px;
        font-size: 14px;
        font-weight: bold;
    }
    .container {
        display: flex;
        justify-content: space-around;
        padding: 20px;
    }
    .box {
        background-color: lightgray;
        padding: 20px;
        width: 250px;
        border-radius: 10px;
        text-align: left;
        font-size: 16px;
    }
    .title {
        font-weight: bold;
        text-decoration: underline;
        margin-bottom: 10px;
    }
</style>

<div class="header">ECG ANALYSIS RESULTS</div>
<div class="result">RESULT: <span id="analysisResult">{{ predicted_class }}</span></div>
<div class="waveform-container">
    <div class="waveform-text">ECG WAVEFORM</div>
</div>
<div class="title">PATIENT DETAILS</div>
<div class="container">
    <div class="box">
        <p>NAME: <span id="patientName">{{ patient_name }}</span></p>
        <p>ID: <span id="patientId">{{ patient_id }}</span></p>
        <p>BP: <span id="patientBP">{{ patient_bp }}</span></p>
        <p>CHOLESTEROL: <span id="patientCholesterol">{{ patient_cholesterol }}</span></p>
        <p>SMOKER: <span id="patientSmoker">{{ patient_smoker }}</span></p>
        <p>DIABETIC: <span id="patientDiabetic">{{ patient_diabetic }}</span></p>
    </div>
    <div class="box">
        <p>FRAMINGHAM’S SCORE: <span id="framinghamScore">{{ framingham_risk|round(2) }}%</span></p>
        <p>GRACE SCORE: <span id="graceScore">{{ grace_score|round(2) }}%</span></p>
        <p>HEART RATE: <span id="heartRate">{{ heart_rate|round(2) }} BPM</span></p>
        <p>PR INTERVAL: <span id="prInterval">{{ pr_interval|round(3) }} s</span></p>
        <p>QT INTERVAL: <span id="qtInterval">{{ qt_interval|round(3) }} s</span></p>
    </div>
</div>

<div class="button-container">
    <a href="{{ url_for('download_report', filename=pdf_filename) }}" class="button btn-download">
        <i class="fas fa-file-pdf"></i> Download Report
    </a>
    <a href="{{ url_for('dashboard') }}" class="button btn-back">
        <i class="fas fa-arrow-left"></i> Back to Dashboard
    </a>
</div>

{% endblock %}


<!-- Flash Messages -->
<div class="container">
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            {% for category, message in messages %}
                <div class="alert alert-{{ category }} alert-dismissible fade show" style="max-width: 400px; margin: 0 auto;">
                    <i class="fas fa-info-circle"></i> {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            {% endfor %}
        {% endif %}
    {% endwith %}
    
    {% block content %}{% endblock %}
</div>








{% extends "base.html" %}
{% block title %}Automatic Analysis - ECG Analysis{% endblock %}

{% block content %}
<style>
    .patient-details {
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
<div class="row justify-content-center">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header bg-white">
                <h3 class="mb-0"><i class="fas fa-microchip"></i> AI ECG Analysis</h3>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-5">
                        <div class="patient-details mb-4">
                            <h5>Patient Details</h5>
                            <hr>
                            <p><strong>Patient ID:</strong> {{ patient.Patient_ID }}</p>
                            <p><strong>Name:</strong> {{ patient.Patient_Name }}</p>
                            <p><strong>Age:</strong> {{ patient.Age }}</p>
                            <p><strong>Gender:</strong> {{ patient.Gender }}</p>
                            
                            <p><strong>Contact Info:</strong> {{ patient.Personal_Contact }}</p>
                            
                            <p><strong>Doctor ID:</strong> {{ patient.Doctor_ID }}</p>
                            <p><strong>Date:</strong> {{ patient.Created_At.astimezone(pytz.timezone('Asia/Kolkata')).strftime("%d %B %Y") }}</p>
                        </div>
                        
                    </div>
                    <div class="col-md-7">
                        <form method="POST">
                            <div class="mb-3">
                                <label class="form-label">Select ECG Record</label>
                                <select name="record_num" class="form-select" required>
                                    <option value="">Select a record...</option>
                                    {% for record in ['100', '101', '102', '103', '104', '105', '106', '107'] %}
                                    <option value="{{ record }}">Record {{ record }}</option>
                                    {% endfor %}
                                </select>
                            </div>
                            
                            <h5 class="mt-4">Patient Information</h5>
                            <hr>
                            
                            <div class="row g-3">
                                <div class="col-md-6">
                                    <label class="form-label">Age (years)</label>
                                    <input type="number" class="form-control" name="age" min="18" max="120" value="{{ patient.Age }}" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Systolic BP (mmHg)</label>
                                    <input type="number" class="form-control" name="systolic_bp" min="50" max="250" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Cholesterol (mg/dL)</label>
                                    <input type="number" class="form-control" name="cholesterol" min="50" max="500" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">HDL (mg/dL)</label>
                                    <input type="number" class="form-control" name="hdl" min="20" max="100" required>
                                </div>
                                <div class="col-md-6">
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" name="smoker">
                                        <label class="form-check-label">
                                            <i class="fas fa-smoking"></i> Current Smoker
                                        </label>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" name="diabetes">
                                        <label class="form-check-label">
                                            <i class="fas fa-diabetes"></i> Diabetes Diagnosis
                                        </label>
                                    </div>
                                </div>
                            </div>
                            <button type="submit" class="btn btn-primary w-100 mt-4">
                                <span class="spinner-border spinner-border-sm d-none"></span>
                                <i class="fas fa-arrow-right"></i> Analyze ECG
                            </button>
                            
                        </form>

                        <!-- ECG Waveform Display -->
                        {% if ecg_filename %}
                        <div class="mt-4">
                            <h5>ECG Waveform</h5>
                            <hr>
                            <div id="ecg-plot">
                                <iframe src="{{ url_for('static', filename=ecg_filename) }}" 
                                        width="100%" 
                                        height="400" 
                                        frameborder="0" 
                                        scrolling="no"></iframe>
                            </div>
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}



{% extends "base.html" %}

{% block title %}ECG Analysis Results{% endblock %}

{% block content %}
<style>
    :root {
        --primary-color: #3498db;
        --secondary-color: #2c3e50;
        --white: #ffffff;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    body {
        background-color: #f5f7fa;
        color: var(--secondary-color);
        line-height: 1.6;
    }

    /* Patient Details Card */
    .patient-card {
        background: var(--white);
        border-radius: 10px;
        padding: 20px;
        box-shadow: var(--shadow);
        width: 300px;
        float: left;
        margin-right: 20px;
    }

    .patient-header {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
        padding-bottom: 5px;
   
    }

    .patient-header h2 {
        margin-left: 10px;
        color: var(--primary-color);
        font-size: 18px;
    }

    .info-group {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        gap: 10px;
    }

    .info-label {
        font-weight: 600;
        color: #7f8c8d;
        font-size: 14px;
        min-width: 80px;
    }

    .info-value {
        font-size: 15px;
        color: var(--secondary-color);
        flex: 1;
    }

    /* Result Card */
    .result-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: var(--shadow);
        margin-bottom: 20px;
        height: 400px;
    }

    h2 {
        color: var(--primary-color);
        font-size: 18px;
        margin-top: 0;
        padding-top: 8px;
        border-top: 2px solid var(--primary-color);
        display: inline-block;
    }

    .result-display {
        font-size: 22px;
        font-weight: bold;
        color: #27ae60;
        text-align: center;
        padding: 15px;
        background-color: #f1f9f5;
        border-radius: 8px;
        margin: 15px 0;
    }

    .risk-metrics {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin-top: 15px;
    }

    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .metric-label {
        font-weight: 600;
        color: #7f8c8d;
        font-size: 14px;
    }

    .metric-value {
        font-size: 16px;
        color: var(--secondary-color);
    }

    /* Waveform Card */
    .waveform-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: var(--shadow);
        clear: both;
        margin-top: 20px;
    }

    .waveform-container {
        width: 100%;
        height: 250px;
    }

    .waveform-image {
        width: 100%;
        height: 250px;
        border-radius: 8px;
        box-shadow: var(--shadow);
    }

    /* Button Styles */
    .button-group {
        display: flex;
        justify-content: center;
        gap: 10px;
        align-items: center;
        margin-top: 15px;
    }

    .button {
        padding: 8px 16px;
        border-radius: 5px;
        align-items: center;
        text-decoration: none;
        font-size: 14px;
        width: 200px;
        font-weight: normal;
        display: inline-block;
        color: white;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .btn-primary {
        background-color: var(--primary-color);
        padding: 8px 12px;
        font-size: 13px;
    }

    .btn-secondary {
        background-color: #7f8c8d;
    }

    .button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Responsive Design */
    @media (max-width: 992px) {
        .patient-card {
            width: 100%;
            float: none;
            margin-right: 0;
            margin-bottom: 20px;
        }
        
        .risk-metrics {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 576px) {
        .risk-metrics {
            grid-template-columns: 1fr;
        }
        
        .info-label {
            font-size: 12px;
            min-width: 60px;
        }
        
        .info-value {
            font-size: 14px;
        }
    }
</style>

<div class="container clearfix">
    <!-- Patient Details Card -->
    <div class="patient-card">
        <div class="patient-header">
            <h2>Patient Information</h2>
        </div>
        
        <div class="info-group">
            <span class="info-label">Name:</span>
            <span class="info-value">{{ patient.Patient_Name if patient else "Not Provided" }}</span>
        </div>
        
        <div class="info-group">
            <span class="info-label">Patient ID:</span>
            <span class="info-value">{{ patient.Patient_ID if patient else "Not Provided" }}</span>
        </div>
        
        <div class="info-group">
            <span class="info-label">Age:</span>
            <span class="info-value">{{ age }} years</span>
        </div>
        
        <div class="info-group">
            <span class="info-label">Systolic BP:</span>
            <span class="info-value">{{ systolic_bp }} mmHg</span>
        </div>
        
        <div class="info-group">
            <span class="info-label">Cholesterol:</span>
            <span class="info-value">{{ cholesterol }} mg/dL</span>
        </div>
        
        <div class="info-group">
            <span class="info-label">HDL:</span>
            <span class="info-value">{{ hdl }} mg/dL</span>
        </div>
        
        <div class="info-group">
            <span class="info-label">Smoker:</span>
            <span class="info-value">{{ 'Yes' if smoker else 'No' }}</span>
        </div>
        
        <div class="info-group">
            <span class="info-label">Diabetic:</span>
            <span class="info-value">{{ 'Yes' if diabetes else 'No' }}</span>
        </div>
    </div>
    
    <!-- Result Card -->
    <div class="result-card">
        <h2>Analysis Results</h2>
        
        <div class="result-display">
            {{ predicted_class }}
        </div>
        
        <div class="risk-metrics">
            <!-- Risk Scores -->
            <div class="metric-card">
                <span class="metric-label">Framingham's Score</span>
                <span class="metric-value">
                    {% if framingham_risk is not none %}
                        {{ framingham_risk|round(2) }}%
                    {% else %}
                        Not Available
                    {% endif %}
                </span>
            </div>
            
            <div class="metric-card">
                <span class="metric-label">GRACE Score</span>
                <span class="metric-value">
                    {% if grace_score is not none %}
                        {{ grace_score|round(2) }}%
                    {% else %}
                        Not Available
                    {% endif %}
                </span>
            </div>
            
            <!-- Metrics -->
            <div class="metric-card">
                <span class="metric-label">Heart Rate</span>
                <span class="metric-value">
                    {% if heart_rate is not none %}
                        {{ heart_rate|round(2) }} BPM
                    {% else %}
                        Not Available
                    {% endif %}
                </span>
            </div>
            
            <div class="metric-card">
                <span class="metric-label">QT Interval</span>
                <span class="metric-value">
                    {% if qt_interval is not none %}
                        {{ qt_interval|round(3) }} s
                    {% else %}
                        Not Available
                    {% endif %}
                </span>
            </div>
            
            <div class="metric-card">
                <span class="metric-label">PR Interval</span>
                <span class="metric-value">
                    {% if pr_interval is not none %}
                        {{ pr_interval|round(3) }} s
                    {% else %}
                        Not Available
                    {% endif %}
                </span>
            </div>
        </div>
    </div>
    
    <!-- Waveform Card -->
    <div class="waveform-card">
        <h2>ECG Waveform</h2>
        <div class="waveform-container">
            <img src="{{ url_for('static', filename=ecg_filename) }}" 
                 class="waveform-image" 
                 alt="ECG Waveform - {{ predicted_class }}">
        </div>
    </div>
    
    <!-- Buttons -->
    <div class="button-group">
        <a href="{{ url_for('download_report', filename=pdf_filename) }}" class="button btn-primary">
            Download Report
        </a>
        <a href="{{ url_for('dashboard') }}" class="button btn-secondary">
            Back to Dashboard
        </a>
    </div>
</div>

<script>
    document.addEventListener("DOMContentLoaded", function () {
        // Any JavaScript functionality can be added here
        console.log("ECG Analysis Results Page Loaded");
    });
</script>

{% endblock %}