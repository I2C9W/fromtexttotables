import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
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

def plot_confusion_matrix(df, symptoms):
    for symptom in symptoms:
        y_true = df[symptom]
        y_pred = df[f"{symptom} pred"]
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        cm_df = pd.DataFrame(cm, index=["False", "True"], columns=["False", "True"])
        plt.figure(figsize=(8,6))
        ax = sns.heatmap(cm_df, annot=True, fmt=".2f", cmap='Blues', vmin=0, vmax=1, annot_kws={"size": 28})
        plt.title(f'{symptom.capitalize()}', fontsize = 28)
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 28)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 28)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=28)
        plt.show()

def main(ground_truth_path, predictions_path):
    symptoms = ["ascites", "abdominal pain", "shortness of breath", "confusion", "liver cirrhosis"]
    gt_df = pd.read_csv(ground_truth_path)
    gt_df[symptoms] = gt_df[symptoms].applymap(parse)
    pred_df = result_json_to_df(predictions_path, symptoms)
    df = gt_df.merge(pred_df, on="report", suffixes=[None, " pred"])
    plot_confusion_matrix(df, symptoms)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('ground_truth_path', type=str, help='Path to ground truth CSV file')
    parser.add_argument('predictions_path', type=str, help='Path to predictions JSONL file')
    args = parser.parse_args()
    main(args.ground_truth_path, args.predictions_path)
