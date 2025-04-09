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
-- Table structure for table `doctors`
--

DROP TABLE IF EXISTS `doctors`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `doctors` (
  `Doctor_ID` int NOT NULL,
  `Doctor_Name` varchar(45) NOT NULL,
  `Contact_Info` varchar(45) NOT NULL,
  `Patient_ID` int NOT NULL,
  PRIMARY KEY (`Doctor_ID`),
  UNIQUE KEY `DoctorsID_UNIQUE` (`Doctor_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `doctors`
--

LOCK TABLES `doctors` WRITE;
/*!40000 ALTER TABLE `doctors` DISABLE KEYS */;
INSERT INTO `doctors` VALUES (1,'Dr.Steve','steve@gmail.com',0),(2,'Dr.Nisha','nisha@gmail.com',0),(3,'Dr.Nandini','nandini@gmail.com',0),(4,'Dr.Bharath','bharath@gmail.com',0),(5,'Dr. Alice','alice@gmail.com',0),(6,'Dr. Bob','bob@gmail.com',0),(7,'Dr. Charlie','charlie@gmail.com',0),(8,'Dr. Diana','diana@gmail.com',0),(9,'Dr. Eve','eve@gmail.com',0),(10,'Dr. Frank','frank@gmail.com',0),(11,'Dr. Grace','grace@gmail.com',0),(12,'Dr. Hank','hank@gmail.com',0),(13,'Dr. Ivy','ivy@gmail.com',0),(14,'Dr. Jack','jack@gmail.com',0),(15,'Dr. Karen','karen@gmail.com',0),(16,'Dr. Leo','leo@gmail.com',0),(17,'Dr. Mona','mona@gmail.com',0),(18,'Dr. Nina','nina@gmail.com',0),(19,'Dr. Oscar','oscar@gmail.com',0),(20,'Dr. Paul','paul@gmail.com',0),(21,'Dr. Quincy','quincy@gmail.com',0),(22,'Dr. Rachel','rachel@gmail.com',0),(23,'Dr. Steve','steve2@gmail.com',0),(24,'Dr. Tina','tina@gmail.com',0),(25,'Dr. Uma','uma@gmail.com',0),(26,'Dr. Victor','victor@gmail.com',0),(27,'Dr. Wendy','wendy@gmail.com',0),(28,'Dr. Xander','xander@gmail.com',0),(29,'Dr. Yara','yara@gmail.com',0),(30,'Dr. Zack','zack@gmail.com',0),(31,'Dr. Amy','amy@gmail.com',0),(32,'Dr. Ben','ben@gmail.com',0),(33,'Dr. Cara','cara@gmail.com',0),(34,'Dr. Dave','dave@gmail.com',0),(35,'Dr. Ella','ella@gmail.com',0),(36,'Dr. Finn','finn@gmail.com',0),(37,'Dr. Gina','gina@gmail.com',0),(38,'Dr. Harry','harry@gmail.com',0),(39,'Dr. Isla','isla@gmail.com',0),(40,'Dr. Jake','jake@gmail.com',0),(41,'Dr. Kara','kara@gmail.com',0),(42,'Dr. Liam','liam@gmail.com',0),(43,'Dr. Mia','mia@gmail.com',0),(44,'Dr. Noah','noah@gmail.com',0),(45,'Dr. Olive','olive@gmail.com',0),(46,'Dr. Pete','pete@gmail.com',0),(47,'Dr. Quinn','quinn@gmail.com',0),(48,'Dr. Ryan','ryan@gmail.com',0),(49,'Dr. Sara','sara@gmail.com',0),(50,'Dr. Tom','tom@gmail.com',0),(51,'Dr. Uma','uma2@gmail.com',0),(52,'Dr. Vince','vince@gmail.com',0),(53,'Dr. Willa','willa@gmail.com',0),(54,'Dr. Xena','xena@gmail.com',0),(55,'Dr. Yvonne','yvonne@gmail.com',0),(56,'Dr. Zane','zane@gmail.com',0),(57,'Dr. Aaron','aaron@gmail.com',0),(58,'Dr. Bella','bella@gmail.com',0),(59,'Dr. Chris','chris@gmail.com',0),(60,'Dr. Dana','dana@gmail.com',0),(61,'Dr. Eric','eric@gmail.com',0),(62,'Dr. Fiona','fiona@gmail.com',0),(63,'Dr. George','george@gmail.com',0),(64,'Dr. Hannah','hannah@gmail.com',0),(65,'Dr. Ian','ian@gmail.com',0),(66,'Dr. Julia','julia@gmail.com',0),(67,'Dr. Kevin','kevin@gmail.com',0),(68,'Dr. Laura','laura@gmail.com',0),(69,'Dr. Mike','mike@gmail.com',0),(70,'Dr. Nancy','nancy@gmail.com',0),(71,'Dr. Oliver','oliver@gmail.com',0),(72,'Dr. Paige','paige@gmail.com',0),(73,'Dr. Quinn','quinn2@gmail.com',0),(74,'Dr. Ryan','ryan2@gmail.com',0),(75,'Dr. Sophia','sophia@gmail.com',0),(76,'Dr. Tyler','tyler@gmail.com',0),(77,'Dr. Uma','uma3@gmail.com',0),(78,'Dr. Victor','victor2@gmail.com',0),(79,'Dr. Wendy','wendy2@gmail.com',0),(80,'Dr. Xavier','xavier@gmail.com',0),(81,'Dr. Yara','yara2@gmail.com',0),(82,'Dr. Zack','zack2@gmail.com',0),(83,'Dr. Amy','amy2@gmail.com',0),(84,'Dr. Ben','ben2@gmail.com',0),(85,'Dr. Cara','cara2@gmail.com',0),(86,'Dr. Dave','dave2@gmail.com',0),(87,'Dr. Ella','ella2@gmail.com',0),(88,'Dr. Finn','finn2@gmail.com',0),(89,'Dr. Gina','gina2@gmail.com',0),(90,'Dr. Harry','harry2@gmail.com',0),(91,'Dr. Isla','isla2@gmail.com',0),(92,'Dr. Jake','jake2@gmail.com',0),(93,'Dr. Kara','kara2@gmail.com',0),(94,'Dr. Liam','liam2@gmail.com',0),(95,'Dr. Mia','mia2@gmail.com',0),(96,'Dr. Noah','noah2@gmail.com',0),(97,'Dr. Olive','olive2@gmail.com',0),(98,'Dr. Pete','pete2@gmail.com',0),(99,'Dr. Quinn','quinn3@gmail.com',0),(100,'Dr. Ryan','ryan3@gmail.com',0),(101,'Dr. Sophia','sophia2@gmail.com',0),(102,'Dr. Tyler','tyler2@gmail.com',0),(103,'Dr. Uma','uma4@gmail.com',0),(104,'Dr. Victor','victor3@gmail.com',0);
/*!40000 ALTER TABLE `doctors` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-04-05 18:50:56
