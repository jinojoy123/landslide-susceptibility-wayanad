# GIS-based Landslide Susceptibility Mapping of Wayanad District, India

### Using Frequency Ratio (FR), Logistic Regression (LR), and Random Forest (RF) Models

## 📌 Overview

This repository contains the code and workflow developed for the study:

**“GIS-based Landslide Susceptibility Mapping of Wayanad District, India Using FR, LR and RF Methods: A Case Study”**

The study integrates geospatial analysis and machine learning techniques to generate landslide susceptibility maps and evaluate model performance using statistical and data-driven approaches.

---

## 🎯 Objectives

* To develop landslide susceptibility maps using **FR, LR, and RF models**
* To compare model performance using **ROC–AUC metrics**
* To identify the most suitable model for landslide prediction in the Wayanad region

---

## 🗺 Study Area

* **Location:** Wayanad District, Kerala, India
* **Region:** Western Ghats
* Characterized by:

  * Steep terrain
  * High monsoonal rainfall
  * Complex lithology
  * Dense vegetation

---

## ⚙️ Methodology

The workflow consists of three major stages:

### 1. Data Preparation

* DEM-based factor extraction:

  * Slope
  * Curvature
  * Relative Relief (RR)
  * Topographic Wetness Index (TWI)
* Distance-based factors:

  * Distance from streams
* Landslide inventory preparation
* Generation of non-landslide points (balanced dataset)

---

### 2. Landslide Susceptibility Modeling

#### 📊 Frequency Ratio (FR)

* Bivariate statistical approach
* Calculates relationship between landslide occurrence and factor classes

#### 📈 Logistic Regression (LR)

* Multivariate statistical model
* Estimates probability of landslide occurrence

#### 🌲 Random Forest (RF)

* Machine learning approach
* Handles nonlinear relationships and interactions
* Hyperparameter tuning performed using **GridSearchCV**

---

### 3. Model Validation

* Performance evaluated using:

  * **Receiver Operating Characteristic (ROC) curve**
  * **Area Under the Curve (AUC)**
* AUC computed using trapezoidal rule
* Comparison of model predictive accuracy

---

## 💻 Software and Tools

* Python (scikit-learn, NumPy, Pandas, Matplotlib)
* GIS Software:

  * ArcGIS / QGIS
* Raster processing:

  * GDAL / Rasterio

---

## 📊 Key Results

* RF model achieved the highest predictive performance
* ROC–AUC values indicate strong agreement between predicted and observed landslides
* Susceptibility maps highlight high-risk zones in steep and high rainfall regions

---

## 📥 Data Availability

* DEM data: USGS Earth Explorer (SRTM 30 m resolution)
* Due to data size and restrictions:

  * Full datasets are not included
  * Sample data / instructions can be provided upon request

## 📌 Reproducibility

This repository provides all necessary scripts to reproduce:

* Landslide susceptibility modeling
* Model comparison
* ROC–AUC evaluation

---

## ⚠️ Notes

* Ensure proper GIS preprocessing before running models
* Coordinate systems and raster alignment are critical
* Hyperparameters may be adjusted based on dataset

---

## 👤 Author

**Jino Joy**
Agricultural and Biosystems Engineering
North Dakota State University

---

## 🔗 License

This project is intended for academic and research purposes.

---

## 🤝 Acknowledgements

* USGS Earth Explorer
* Open-source Python and GIS communities
