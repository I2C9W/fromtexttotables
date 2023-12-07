import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import argparse

def parse(x):
    return "True" if x == 1 else "False"

def result_json_to_df(json_path, symptoms):
    with open(json_path, "r") as json_file:
        records = []
        for line in json_file:
            try:
                llama_response = json.loads(line)
                extracted_info = json.loads(llama_response["content"])
                records.append(
                    (
                        llama_response["report"],
                        *[extracted_info[label]["present"] for label in symptoms]
                    )
                )
            except json.JSONDecodeError:
                continue
    pred_df = pd.DataFrame(records, columns=["report", *symptoms])
    pred_df[symptoms] = pred_df[symptoms].applymap(parse)
    return pred_df

def compute_accuracies(df, symptoms, models):
    model_accuracies = {model: [] for model in models}
    for symptom in symptoms:
        for model in models:
            pred_df = result_json_to_df(f"results-{model}_binary_p1.jsonl", symptoms)
            merged_df = df.merge(pred_df, on="report", suffixes=[None, " pred"])
            accuracies = []
            for _ in range(100):
                sample_df = merged_df.sample(frac=1, replace=True)
                y_true = sample_df[symptom]
                y_pred = sample_df[f"{symptom} pred"]
                accuracies.append(accuracy_score(y_true, y_pred))
            model_accuracies[model].append(np.mean(accuracies))
    return model_accuracies

def plot_model_accuracies(model_accuracies, symptoms, models):
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        accuracies = [np.mean(model_accuracies[model][symptom]) for symptom in symptoms]
        plt.plot(symptoms, accuracies, label=model)
    plt.title("Model Accuracies for Each Symptom")
    plt.ylabel("Accuracy")
    plt.xlabel("Symptoms")
    plt.legend()
    plt.show()

def main(ground_truth_path):
    symptoms = ["ascites", "abdominal pain", "shortness of breath", "confusion", "liver cirrhosis"]
    models = ["7b", "13b", "70b"]
    gt_df = pd.read_csv(ground_truth_path)
    gt_df[symptoms] = gt_df[symptoms].applymap(parse)
    model_accuracies = compute_accuracies(gt_df, symptoms, models)
    plot_model_accuracies(model_accuracies, symptoms, models)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Accuracy Comparison of Models.')
    parser.add_argument('ground_truth_path', type=str, help='Path to the ground truth CSV file')
    args = parser.parse_args()
    main(args.ground_truth_path)
