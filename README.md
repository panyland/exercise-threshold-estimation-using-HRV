# HRV-Based Intensity Zone Classification Using Machine Learning

## Overview
This project investigates the use of **heart rate variability (HRV)** to classify exercise intensity zones. The central aim is to determine whether short RR-interval sequences (~1 minute in the current dataset) can reliably indicate the intensity zone of exercise. Such an approach could enable the identification of **lactate and ventilatory thresholds** using HRV alone during graded exercise tests.

## Objectives
- **Data preprocessing and cleaning** to ensure quality input  
- **Feature extraction** from RR-intervals, including time-domain and frequency-domain metrics  
- **Classification using Random Forest models**, tested across both 2-class and 3-class setups  
- **Hyperparameter tuning and feature selection** to optimize model performance

### Practical use
1. Graded test with HRV measurement (e.g., on a Wattbike) to identify thresholds  
2. Follow-up test with **finer workload increments** to improve resolution of threshold detection  
3. Repeat this process to enhance precision in threshold detection ?  

## Data
Data were collected from **incremental cycle ergometer tests** where ventilatory thresholds (VT1 and VT2) were identified between workload steps. The dataset includes:  
- **Power output**  
- **RR-intervals**  
- **VO₂ measurements**  
- **Ventilatory thresholds**  

The study involved **18 subjects**, with data sourced from the [PhysioNet ACTES dataset](https://physionet.org/content/actes-cycloergometer-exercise/1.0.0/).

Model development is currently ongoing.

## Future Work
- Extend classification methods to include **lactate thresholds**  
- Validate the model across **broader populations** and better testing protocols  
- Explore integration with **wearable sensor systems** for real-world applications  
