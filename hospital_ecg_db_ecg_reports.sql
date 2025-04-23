CREATE DATABASE  IF NOT EXISTS `hospital_ecg_db` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;
USE `hospital_ecg_db`;
-- MySQL dump 10.13  Distrib 8.0.41, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: hospital_ecg_db
-- ------------------------------------------------------
-- Server version	8.3.0

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `ecg_reports`
--

DROP TABLE IF EXISTS `ecg_reports`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
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
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ecg_reports`
--

LOCK TABLES `ecg_reports` WRITE;
/*!40000 ALTER TABLE `ecg_reports` DISABLE KEYS */;
INSERT INTO `ecg_reports` VALUES ('ECG-20250421-PT-10002-2025','PT-10002-2025','DR-002-2024','2025-04-21 20:13:27','100','Heart Block',56.09,75.51,1805.32,0.814,2.12,13.2,80,100,90,0,0,2273,NULL,'/static/ecg_images/ecg_PT-10002-2025_100_20250421_201327.png','2025-04-21 14:43:27','2025-04-21 14:43:27'),('ECG-20250421-PT-10064-2025','PT-10064-2025','DR-002-2024','2025-04-21 21:01:17','205','Fusion',58.23,87.25,1804.53,0.658,8.84,17.95,86,278,85,1,1,2625,NULL,'/static/ecg_images/ecg_PT-10064-2025_205_20250421_210116.png','2025-04-21 15:31:16','2025-04-21 15:31:16'),('ECG-20250421-PT-10065-2025','PT-10065-2025','DR-002-2024','2025-04-21 21:06:27','207','Ventricular Fibrillation',87.47,65.65,1804.91,0.683,7.27,13.08,89,221,83,1,1,1976,NULL,'/static/ecg_images/ecg_PT-10065-2025_207_20250421_210627.png','2025-04-21 15:36:27','2025-04-21 15:36:27'),('ECG-20250421-PT-10066-2025','PT-10066-2025','DR-002-2024','2025-04-21 21:28:32','124','Normal',37.11,53.9,1804.54,1.25,7.67,10.73,89,256,89,0,0,1622,NULL,'/static/ecg_images/ecg_PT-10066-2025_124_20250421_212832.png','2025-04-21 15:58:32','2025-04-21 15:58:32'),('ECG-20250422-PT-10002-2025','PT-10002-2025','DR-002-2024','2025-04-22 00:36:30','207','Ventricular Fibrillation',87.47,65.65,1804.91,0.683,5.49,10.83,88,200,89,0,0,1976,NULL,'/static/ecg_images/ecg_PT-10002-2025_207_20250422_003629.png','2025-04-21 19:06:29','2025-04-21 19:06:29'),('ECG-20250422-PT-10064-2025','PT-10064-2025','DR-002-2024','2025-04-22 01:00:20','105','SVT',51.21,83.6,1804.29,0.728,3.02,17.07,89,100,90,0,0,2515,NULL,'/static/ecg_images/ecg_PT-10064-2025_105_20250422_010019.png','2025-04-21 19:30:19','2025-04-21 19:30:19'),('ECG-20250422-PT-10068-2025','PT-10068-2025','DR-002-2024','2025-04-22 18:41:23','205','Fusion',58.23,87.25,1804.53,0.658,6.71,19.5,89,200,87,1,0,2625,NULL,'/static/ecg_images/ecg_PT-10068-2025_205_20250422_184123.png','2025-04-22 13:11:23','2025-04-22 13:11:23'),('ECG-20250422011459-PT-10002-2025','PT-10002-2025','DR-002-2024','2025-04-22 01:14:59','205','Fusion',58.23,87.25,1804.53,0.658,2.57,15.05,90,100,89,0,0,2625,NULL,'/static/ecg_images/ecg_PT-10002-2025_205_20250422_011459.png','2025-04-21 19:44:59','2025-04-21 19:44:59'),('ECG-20250422011526-PT-10002-2025','PT-10002-2025','DR-002-2024','2025-04-22 01:15:27','205','Fusion',58.23,87.25,1804.53,0.658,2.57,15.05,90,100,89,0,0,2625,NULL,'/static/ecg_images/ecg_PT-10002-2025_205_20250422_011459.png','2025-04-21 19:45:26','2025-04-21 19:45:26');
/*!40000 ALTER TABLE `ecg_reports` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-04-23 11:16:50
