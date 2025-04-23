# ArithmiaAI


A deep learning-powered analysis for automated ECG interpretation and cardiovascular risk prediction.

---

##  About the Project

Cardiovascular diseases (CVDs) are a leading global cause of death. This system automates ECG analysis using AI to:
- Detect arrhythmias, heart attacks, and other abnormalities
- Predict 10-year heart disease risk (Framingham Score)
- Generate PDF reports for doctors/patients
- Enable real-time monitoring via web dashboard

---

##  Key Features

### 1. **ECG Signal Processing**

#### Example: Noise removal
from neurokit2 import ecg_clean
cleaned_ecg = ecg_clean(raw_signal, sampling_rate=100)

- Butterworth & Wavelet filters for noise removal  
- Pan-Tompkins algorithm for R-peak/QRS detection  
- HRV, QT/PR interval calculation  

### 2. **AI Classification (CNN Model)**
- Classifies ECGs into 
  - Normal
  - SVT
  - Atrial Fibrillation
  - Ventricular Fibrillation
  - Ventricular Tachycardia
  - Heart Block  
- Trained on **MIT-BIH** and **PTB-XL** datasets

### 3. **Risk Assessment**
- Framingham Risk Score (10-year CVD risk)  
- GRACE Score (post-heart attack mortality)  

### 4. **Web Dashboard**
- Real-time ECG visualization (Plotly)  
- Secure patient portal (Flask + MySQL)  

---

## Installation


## Technology Stack

| Category          | Technologies Used                 |
| :---------------- | :-------------------------------- |
| **Backend**       | Python, Flask                     |
| **Database**      | MySQL                             |
| **Authentication**| Flask-Login, Flask-Bcrypt         |
| **ML/AI**         | TensorFlow/Keras                  |
| **ECG Processing**| WFDB, NeuroKit2, SciPy, NumPy     |
| **Data Handling** | Pandas                            |
| **Visualization** | Plotly                            |
| **PDF Generation**| xhtml2pdf / WeasyPrint (Optional) |


---

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Python:** Version 3.8 or higher.
*   **MySQL Server:** Community Server recommended (See Installation steps below).
*   **Git:** For cloning the repository.
*   **System Dependencies:** Required for certain Python packages (especially `mysqlclient` and PDF generators).
    *   **For Ubuntu/Debian:**
        ```bash
        sudo apt-get update && sudo apt-get install -y python3-dev default-libmysqlclient-dev build-essential libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 libffi-dev shared-mime-info pkg-config
        ```
    *   **For Fedora/CentOS/RHEL:**
        ```bash
        sudo yum update && sudo yum install -y python3-devel mysql-devel gcc-c++ cairo pango gdk-pixbuf2 libffi-devel redhat-rpm-config
        ```
    *   **For macOS:** Use Homebrew:
        ```bash
        brew install mysql pkg-config cairo pango gdk-pixbuf libffi
        ```
    *   **For Windows:** Installers for Python and MySQL usually handle dependencies. Ensure `mysqlclient` can be installed (it might require Visual C++ Build Tools). PDF generation might require extra steps depending on the library.

---

### Setting Up MySQL Community Server

Step 1: Download MySQL Community Server

  1.	Visit the MySQL Community Downloads Page:
      [MySQL Community Downloads](https://dev.mysql.com/downloads/workbench/)
  2.	Download MySQL Community Server:
      Download MySQL Community Server using MSI installer according to your operating system.
      If prompted, you can skip the login/signup by clicking on "No thanks, just start my download".
    	
Step 2: Install MySQL Community Server

  1.	Run the Installer:
      Once the download is complete, run the installer file.
  2.	Choose Setup Type:
      In the MySQL Installer window, select the Developer Default setup type.
      Click Next and proceed with the complete installation. The installer will download and install the selected MySQL products.

Step 3: Configuration

  1.	Server Configuration:
      After the installation, the MySQL Installer will prompt you to configure the server.
    	Select Standalone MySQL Server. Click Next.
    	Choose the default port (3306) and ensure that it is open and available. Click Next.

Step 4: Authentication Method

  1.	Set Authentication:
      Use the default authentication method (recommended). Click Next.
    	Set the root password for your MySQL server. Remember this password as you will need it to connect.
    	Optionally, add additional MySQL user accounts. Click Next.

Step 5: Apply Configuration

  1.	Apply Settings:
      Review the configuration settings and click Execute to apply them.
      Once the configuration is complete, click Finish.


### Database Setup
Create a New SQL Tab for Executing Queries
Open SQL Tab:
    Click on File > New Query Tab.

### 1. Create Database
```sql
CREATE DATABASE hospital_ecg_db;
USE hospital_ecg_db;
```

### 2. Staff Table
```sql
CREATE TABLE `staff` (
  `Staff_ID` varchar(80) NOT NULL,
  `Password` varchar(80) NOT NULL,
  `StaffName` varchar(80) DEFAULT NULL,
  PRIMARY KEY (`Staff_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Sample staff data
INSERT INTO `staff` VALUES 
('1AH21CS043','$2b$12$pVtihF8xr8znJqCByaSOVuJ.m7fw..iiEzD2GnbUTDvPvBW2ZUmfK','Hruthvik');
```

### 3. Doctor Table
```sql
CREATE TABLE `doctor` (
  `Doctor_ID` varchar(255) NOT NULL,
  `Username` varchar(255) NOT NULL,
  `Password` varchar(255) NOT NULL,
  PRIMARY KEY (`Doctor_ID`),
  UNIQUE KEY `Username` (`Username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Sample doctor data
INSERT INTO `doctor` VALUES 
('DR-002-2024','HRUTHVIK','$2b$12$Ei7rVxrjAW32IJujwDtPQuUs21e26iIcQqhNHqx46UlTNOG5E2ffW'),
('DR-009-2024','Hruthik','$2b$12$9YzlKL6HEYwXJ0QMX3zP7.HYc3iEa938jzUqvyf/v3FfMSQu8fnxm');
```

### 4. Patient Profile Table
```sql
CREATE TABLE `patient_profile` (
  `Patient_ID` varchar(20) NOT NULL,
  `Patient_Name` varchar(45) NOT NULL,
  `Age` int NOT NULL,
  `Gender` varchar(45) NOT NULL,
  `Address` varchar(60) NOT NULL,
  `Email_ID` varchar(45) NOT NULL,
  `Personal_Contact` bigint NOT NULL,
  `Emergency_Contact` bigint NOT NULL,
  `Doctor_ID` varchar(255) DEFAULT NULL,
  `Created_At` datetime NOT NULL,
  `Staff_Username` varchar(80) NOT NULL,
  PRIMARY KEY (`Patient_ID`),
  UNIQUE KEY `Patient_ID_UNIQUE` (`Patient_ID`),
  KEY `fk_patient_doctor` (`Doctor_ID`),
  CONSTRAINT `fk_patient_doctor` FOREIGN KEY (`Doctor_ID`) REFERENCES `doctor` (`Doctor_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
```

### 5. ECG Input Data Table
```sql
CREATE TABLE `input` (
  `Record_ID` int NOT NULL AUTO_INCREMENT,
  `Patient_ID` varchar(20) DEFAULT NULL,
  `Smoker` tinyint NOT NULL,
  `Diabetic` tinyint NOT NULL,
  `Cholesterol` float NOT NULL,
  `HDL` int NOT NULL,
  `Blood_Pressure` float NOT NULL,
  `Other_Issues` varchar(200) NOT NULL DEFAULT '',
  `Generated_AT` varchar(45) NOT NULL DEFAULT 'Timestamp(Now)',
  `Doctor_ID` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`Record_ID`),
  UNIQUE KEY `Record_ID_UNIQUE` (`Record_ID`),
  KEY `_idx` (`Patient_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=690 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
```

### 6. ECG Reports Table 
```sql
CREATE TABLE `ecg_reports` (
  `report_id` varchar(50) NOT NULL,
  `patient_id` varchar(20) NOT NULL,
  `doctor_id` varchar(20) DEFAULT NULL,
  `report_date` datetime NOT NULL,
  `record_num` varchar(20) DEFAULT NULL,
  `predicted_class` varchar(50) NOT NULL,
  `confidence` float NOT NULL DEFAULT '0',
  `heart_rate` float NOT NULL DEFAULT '0',
  `qt_interval` float NOT NULL DEFAULT '0',
  `pr_interval` float NOT NULL DEFAULT '0',
  `framingham_risk` float NOT NULL DEFAULT '0',
  `grace_score` float NOT NULL DEFAULT '0',
  `systolic_bp` float NOT NULL DEFAULT '0',
  `cholesterol` float NOT NULL DEFAULT '0',
  `hdl` float NOT NULL DEFAULT '0',
  `smoker` tinyint(1) NOT NULL DEFAULT '0',
  `diabetes` tinyint(1) NOT NULL DEFAULT '0',
  `all_beats_count` int NOT NULL DEFAULT '0',
  `class_probabilities` json DEFAULT NULL,
  `ecg_image_path` varchar(255) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`report_id`),
  KEY `idx_patient` (`patient_id`),
  KEY `idx_doctor` (`doctor_id`),
  KEY `idx_report_date` (`report_date`),
  CONSTRAINT `ecg_reports_ibfk_1` FOREIGN KEY (`patient_id`) REFERENCES `patient_profile` (`Patient_ID`),
  CONSTRAINT `ecg_reports_ibfk_2` FOREIGN KEY (`doctor_id`) REFERENCES `doctor` (`Doctor_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
```

**Execution Tip:** Run the command in terminal/powershell:
```bash
mysql -u root -p < schema_setup.sql
```


## 🛠 Installation & Setup

### Prerequisites
- Python 3.8+
- MySQL Server
- System dependencies:
  ```bash
  # For Ubuntu/Debian
  sudo apt-get install python3-dev default-libmysqlclient-dev build-essential libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 libffi-dev shared-mime-info


---

---
## 🌟 Features
- **Multi-user authentication** (Doctors/Staff)
- **MIT-BIH Arrhythmia Database** integration
- **Interactive Plotly charts** for ECG visualization
- **Risk score calculation** (Framingham/GRACE)
- **Responsive web interface**
---


---
### 3. Application Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/hruthvik-04/AI-ECG-ANALYSIS-PROJECT.git
    cd AI-ECG-ANALYSIS-PROJECT
    ```

2.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate Environment:**
    *   Linux/macOS: `source venv/bin/activate`
    *   Windows (cmd/powershell): `.\venv\Scripts\activate`

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure you have created the `requirements.txt` file provided in the previous response)*

5.  **Download ECG Datasets:**
    *   Place the **MIT-BIH Arrhythmia Database** files (e.g., `100.dat`, `100.hea`, `100.atr`, etc.) inside the `mit-bih-arrhythmia-database-1.0.0/` directory within the project root. You can download it from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/).
    *   *(If using PTB-XL, specify its location and update relevant code paths)*

6.  **Place ML Model:**
    *   Ensure your trained model file (e.g., `ecg_arrhythmia_detector_....h5`) is placed in the `model/` directory within the project root.

7.  **Configure Application:**
    *   Open the main Flask application file (e.g., `app.py`).
    *   **Crucially, update the MySQL connection details:**
        ```python
        app.config["MYSQL_HOST"] = "localhost" # Or your MySQL host
        app.config["MYSQL_USER"] = "your_mysql_user" # Replace with your user
        app.config["MYSQL_PASSWORD"] = "your_mysql_password" # Replace with your password
        app.config["MYSQL_DB"] = "hospital_ecg_db"
        ```
    *   **Set a strong `app.secret_key`:** This is vital for session security. Replace the default value. You can generate one using `python -c 'import os; print(os.urandom(24).hex())'`.
    *   Verify `DATASET_PATH`, `MODEL_PATH`, and other paths if you've placed files differently. *(Consider using environment variables for configuration in production)*.

---

## Running the Application

1.  Ensure your virtual environment is activated.
2.  Make sure your MySQL server is running.
3.  Run the Flask development server:
    ```bash
    flask run
    # Or if your main file is app.py: python app.py
    ```
4.  Open your web browser and navigate to `http://127.0.0.1:5000` (or the address shown in the terminal).

---

## Usage

1.  **Login:** Access the application via your browser.
    *   **Staff:** Use the main login page (`/`) with Staff credentials.
    *   **Doctors:** Use the `/doctor_login` page with Doctor credentials.
2.  **Staff:**
    *   Register new patients via the `/patient_registration` page.
    *   Edit existing patient details via `/edit_patient/<patient_id>`.
3.  **Doctors:**
    *   Access the main dashboard (`/input_form`) after login.
    *   Search for patients by ID.
    *   View patient details and medical history.
    *   Enter ECG record numbers and clinical data to perform automatic analysis (`/automatic_analysis/<patient_id>`).
    *   View analysis results, risk scores, and interactive ECG plots on the `/result` page (rendered automatically).
    *   View and optionally download PDF reports (`/generate_report/<patient_id>`).

---

## PDF Report Generation

*   The application can generate PDF reports of the analysis results.
*   This feature requires either `xhtml2pdf` (fallback) or `WeasyPrint` (preferred, better results but more complex installation with system dependencies) to be installed.
*   If neither library is found, PDF download buttons may be disabled, and a warning will be logged at startup.

---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## License

Distributed under the MIT License. See `LICENSE` file for more information.

---

## Contact

Hruthvik - [hruthvik2K3@gmail.com](mailto:hruthvik2K3@gmail.com)


Project Link: [https://github.com/hruthvik-04/AI-ECG-ANALYSIS-MINDTECK.git](https://github.com/hruthvik-04/AI-ECG-ANALYSIS-MINDTECK.git)
