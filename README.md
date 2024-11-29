# LSTM-autoencoder

Kick Detection Using LSTM Autoencoder Networks

Kick detection is a critical safety concern in oil and gas drilling operations. Undetected kicks can lead to dangerous explosions, environmental damage, property loss, and harm to human life. This repository contains the implementation of a novel approach for kick detection using Long Short-Term Memory Autoencoder (LSTM-AE) networks—a type of recurrent neural network designed for sequential and time-dependent data analysis.

Features

Key Highlights

	•	LSTM-AE Model: Captures long-term dependencies in sequential data, identifying subtle patterns that could indicate impending kicks.
	•	Preprocessing Pipeline: Includes feature selection, normalization, and handling of missing values for effective data preparation.
	•	Baseline Comparison: Compares the LSTM-AE model with conventional kick detection methods to demonstrate improvements in accuracy, precision, and recall.

Dashboard

A Streamlit-based dashboard is developed for real-time monitoring of drilling operations:
	•	Visualize summary statistics and preview sensor data.
	•	Run the LSTM-AE model to detect kicks.
	•	Plot actual vs. predicted values for critical parameters like pressure and flow rate.

Dataset

The dataset used in this research is proprietary and cannot be shared publicly due to confidentiality agreements. To use this repository, you will need to provide your own dataset that aligns with the format and requirements described in the data/README.md file.

Installation

Clone the repository and install the required dependencies:

git clone https://github.com/your-repo/kick-detection.git
cd kick-detection
pip install -r requirements.txt

Usage

	1.	Prepare the Data: Ensure your data follows the format specified in the data/README.md file.
	2.	Train the Model: Use historical drilling data to train the LSTM-AE model.
	3.	Run the Dashboard: Launch the Streamlit dashboard for real-time monitoring.

streamlit run dashboard.py

Results

The LSTM-AE model demonstrates:
	•	High accuracy and recall for kick detection.
	•	Superior performance compared to baseline methods.
	•	Effective handling of noisy and large-volume sensor data.

Limitations & Future Work

	•	Limitations: The model may require further tuning for different drilling environments and cannot address all edge cases.
	•	Future Scope: Incorporation of additional models and more robust testing on diverse datasets.

This repository is a step towards improving safety and operational efficiency in drilling operations. Contributions and suggestions are welcome to enhance the project further.
