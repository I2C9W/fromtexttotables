# fromtexttotables
## Confusion Matrix Analysis Script

This Python script, `confusionmatrix.py`, generates confusion matrices for machine learning model predictions. It compares predictions against a ground truth dataset to visualize the performance of a classification model.

### Setup and Run

1. Ensure you have Python installed.
2. Install required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`.
3. Place your ground truth data and prediction results in accessible paths.

### Usage

Run the script from the command line by specifying the path to your ground truth data and predictions:

```bash
python confusionmatrix.py path/to/ground_truth.csv path/to/predictions.jsonl

```
The script will generate confusion matrices for each classification label, helping you assess your model's performance.


## Accuracy Comparison Script

`accuracy_comparison.py` is a Python script designed to compare the accuracy of different machine learning models. It calculates and visualizes the accuracy of each model for various symptoms.

### Setup and Run

1. Ensure Python is installed on your system.
2. Install necessary Python packages: `pandas`, `numpy`, `matplotlib`, `sklearn`.
3. Place your ground truth dataset in an accessible location.

### Usage

To use the script, run it from the command line with the path to your ground truth data:

```bash
python accuracy_comparison.py path/to/ground_truth.csv
```
The script will calculate the accuracies of specified models for different symptoms and plot the results, aiding in the comparative analysis of model performance.

## MIMIC Features Extraction Script

`extract_mimic_features_from_report.py` is a Python script designed to extract and analyze specific medical features from patient reports using a predefined grammar and prompt.

### Setup and Run

1. Ensure Python is installed on your system.
2. Install necessary Python packages: `pandas`, `requests`, `tqdm`.
3. Place your MIMIC ground truth dataset in an accessible location.

### Usage

Run the script from the command line by specifying the path to your MIMIC ground truth data:

```bash
python extract_mimic_features_from_report.py path/to/MIMIC_groundtruth.csv
```

The script processes each report in the dataset, extracting specific medical features using a specialized grammar and saves the results in a JSONL file, facilitating the analysis of medical data.