# HRV-Based Aerobic/Anaerobic Threshold Estimation Using Machine Learning

## Overview
This project investigates the use of **heart rate variability (HRV)** to identify exercise intensity zones using **machine learning**. The central aim is to determine whether short RR-interval sequences (approximately 1 minute in the current dataset) can reliably indicate exercise intensity zones. Such an approach could enable the identification of **lactate and ventilatory thresholds** using HRV alone during graded exercise tests.

## Objectives
- Develop a data preprocessing pipeline to ensure high-quality input  
- Extract features from RR-intervals, including standard time-domain metrics and measures such as **DFA-alpha1**  
- Develop a **Random Forest** model to predict exercise intensity zones based on features calculated from RR-interval sequences  
- Create an API to enable practical testing and integration with wearable devices  

## Data
The data currently used for development were collected from **incremental cycle ergometer tests**, during which ventilatory thresholds (VT1 and VT2) were identified. The dataset includes:  
- **Power output**  
- **RR-intervals**  
- **VO₂ measurements**  
- **Ventilatory thresholds**  

The dataset comprises **18 subjects** and is sourced from the [PhysioNet ACTES dataset](https://physionet.org/content/actes-cycloergometer-exercise/1.0.0/).

## Future Work
- Extend classification methods to include lactate thresholds  
- Validate the model across larger populations and datasets with improved graded testing protocols  
