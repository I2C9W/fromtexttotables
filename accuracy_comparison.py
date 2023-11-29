# %%
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# %%
df = pd.read_csv("/mnt/bulk/isabella/llamaproj/MIMIC_groundtruth_TF.csv")
df.head()

# %%
def parse(x):
    if x == 0:
        return "False"
    elif x == 1:
        return "True"
    else:
        return "False"


def result_json_to_df(json_path):
    symptoms = [
        "ascites",
        "abdominal pain",
        "shortness of breath",
        "confusion",
        "liver cirrhosis",
    ]
    with open(json_path, "r") as json_file:
        records = []
        for line in json_file:
            try:
                llama_response = json.loads(line)
                extracted_info = json.loads(llama_response["content"])
                records.append(
                    (
                        llama_response["report"],
                        *[
                            extracted_info[label] and extracted_info[label]["present"]
                            for label in symptoms
                        ],
                    )
                )
            except json.JSONDecodeError:
                continue

    pred_df = pd.DataFrame(records, columns=["report", *symptoms])
    pred_df[symptoms] = pred_df[symptoms].applymap(parse)
    return pred_df


#%%
gt_df = pd.read_csv("/mnt/bulk/isabella/llamaproj/MIMIC_groundtruth_TF.csv")
symptoms = [
    "ascites",
    "abdominal pain",
    "shortness of breath",
    "confusion",
    "liver cirrhosis",
]
# gt_df[symptoms] = gt_df[symptoms].map(parse)
gt_df[symptoms] = gt_df[symptoms].applymap(parse)

#%%
# pred_df = result_json_to_df(f"results-70b_binary_SYS_rq_cotgrammar_defimpl.jsonl")
# df = gt_df.merge(pred_df, on="report", suffixes=[None, " pred"])

#%%
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

for symptom in symptoms:
    models = []
    accuracy_lowers, accuracy_medians, accuracy_uppers = [], [], []

    for model in ["7b", "13b", "70b"]:
        pred_df = result_json_to_df(f"results-{model}_binary_p1.jsonl")
        df = gt_df.merge(pred_df, on="report", suffixes=[None, " pred"])

        # sensitivities, specificities, accuracies, ppvs, npvs = [], [], [], [], []
        accuracies = []
        for _ in range(100):
            sample_df = df.sample(frac=1, replace=True)
            y_true = sample_df[symptom]
            y_pred = sample_df[f"{symptom} pred"]

            accuracies.append(accuracy_score(y_true, y_pred))

        models.append(model)

        accuracy_lowers.append(np.quantile(accuracies, 0.025))
        accuracy_medians.append(np.quantile(accuracies, 0.5))
        accuracy_uppers.append(np.quantile(accuracies, 0.975))

        # ... [Rest of the code for confusion matrices]

    plt.figure()

    plt.errorbar(
        range(len(models)),
        accuracy_medians,
        yerr=np.abs(np.stack([accuracy_lowers, accuracy_uppers]) - accuracy_medians),
        capsize=3,
        label="accuracy",
    )
    plt.xticks(np.arange(len(models)), labels=models)
    plt.title(f"{symptom}")
    plt.ylim(0, 1)
    plt.show()

    # ... [Rest of the code for plotting]

# %%
##############################################################
# Try boxplots instead
#############################################################
#%%
for symptom in symptoms:
    all_accuracies = []  # List to store accuracies for all models

    for model in ["7b", "13b", "70b"]:
        pred_df = result_json_to_df(f"results-{model}_binary_p1.jsonl")
        df = gt_df.merge(pred_df, on="report", suffixes=[None, " pred"])

        accuracies = []
        for _ in range(100):
            sample_df = df.sample(frac=1, replace=True)
            y_true = sample_df[symptom]
            y_pred = sample_df[f"{symptom} pred"]
            accuracies.append(accuracy_score(y_true, y_pred))

        all_accuracies.append(accuracies)  # Append model accuracies to the list

    plt.figure()
    plt.boxplot(all_accuracies, labels=["7b", "13b", "70b"])
    plt.title(f"{symptom}")
    plt.ylim(0, 1)
    plt.show()

# %%
##############################################################
# Accuracy across all symptoms
#############################################################
model_accuracies = {model: [] for model in ["7b", "13b", "70b"]}  # Dictionary to store accuracies

for symptom in symptoms:
    for model in ["7b", "13b", "70b"]:
        pred_df = result_json_to_df(f"results-{model}_binary_p1.jsonl")
        df = gt_df.merge(pred_df, on="report", suffixes=[None, " pred"])

        accuracies = []
        for _ in range(100):
            sample_df = df.sample(frac=1, replace=True)
            y_true = sample_df[symptom]
            y_pred = sample_df[f"{symptom} pred"]
            accuracies.append(accuracy_score(y_true, y_pred))

        model_accuracies[model].append(accuracies)  # Append accuracies for each symptom

# Preparing data for boxplot
boxplot_data = []
for model in ["7b", "13b", "70b"]:
    # Flatten accuracies for each model across all symptoms
    flattened_accuracies = [acc for symptom_acc in model_accuracies[model] for acc in symptom_acc]
    boxplot_data.append(flattened_accuracies)

plt.figure(figsize=(10, 6))
plt.boxplot(boxplot_data, labels=["7b", "13b", "70b"])
plt.title("Accuracies Across All Symptoms")
plt.ylim(0, 1)
plt.show()

# %%
##############################################################
# Accuracy across all symptoms
#############################################################

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'symptoms' is a list of five symptoms
model_accuracies = {model: {symptom: [] for symptom in symptoms} for model in ["7b", "13b", "70b"]}

for symptom in symptoms:
    for model in ["7b", "13b", "70b"]:
        pred_df = result_json_to_df(f"results-{model}_binary_p1.jsonl")
        df = gt_df.merge(pred_df, on="report", suffixes=[None, " pred"])

        accuracies = []
        for _ in range(100):
            sample_df = df.sample(frac=1, replace=True)
            y_true = sample_df[symptom]
            y_pred = sample_df[f"{symptom} pred"]
            accuracies.append(accuracy_score(y_true, y_pred))

        model_accuracies[model][symptom] = accuracies

# Plotting
plt.figure(figsize=(15, 6))

# Number of models and symptoms
n_models = len(model_accuracies)
n_symptoms = len(symptoms)
model_names = list(model_accuracies.keys())

# Preparing data for boxplot
data = []
for model in model_names:
    for symptom in symptoms:
        data.append(model_accuracies[model][symptom])

# Plotting the boxplots
positions = np.arange(1, n_models * n_symptoms + 1)
plt.boxplot(data, positions=positions, widths=0.6)

# Customizing the x-axis
xtick_positions = np.arange(1.5, n_models * n_symptoms + 1, n_symptoms)
plt.xticks(xtick_positions, model_names)

plt.title("Model Accuracies for Each Symptom")
plt.ylim(0, 1)
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'symptoms' is a list of five symptoms
model_accuracies = {model: {symptom: [] for symptom in symptoms} for model in ["7b", "13b", "70b"]}

for symptom in symptoms:
    for model in ["7b", "13b", "70b"]:
        pred_df = result_json_to_df(f"results-{model}_binary_p1.jsonl")
        df = gt_df.merge(pred_df, on="report", suffixes=[None, " pred"])

        accuracies = []
        for _ in range(10000):
            sample_df = df.sample(frac=1, replace=True)
            y_true = sample_df[symptom]
            y_pred = sample_df[f"{symptom} pred"]
            accuracies.append(accuracy_score(y_true, y_pred))

        model_accuracies[model][symptom] = accuracies

# Plotting
plt.figure(figsize=(11, 6))

n_models = len(model_accuracies)
n_symptoms = len(symptoms)
model_names = list(model_accuracies.keys())
#colors = ['blue', 'green', 'red', 'cyan', 'magenta']  # Assuming 5 symptoms
colors = ['#ffff33', '#4DBBD5FF', '#984ea3', '#ff7f00', '#a65628']

# Adjusting positions for closer grouping
group_width = 0.8  # Width of each group of boxplots
spacing = 1.0     # Spacing between each group

positions = []
current_position = 1
for _ in range(n_models):
    for _ in range(n_symptoms):
        positions.append(current_position)
        current_position += group_width / n_symptoms
    current_position += spacing

# Preparing data and plotting each boxplot
for i, model in enumerate(model_names):
    for j, symptom in enumerate(symptoms):
        data = model_accuracies[model][symptom]
        plt.boxplot(data, positions=[positions[i * n_symptoms + j]], widths=group_width / n_symptoms, 
                    patch_artist=True, boxprops=dict(facecolor=colors[j]))

# Customizing the x-axis
xtick_positions = np.arange(1 + group_width / 2, n_models * (group_width + spacing) + 1, group_width + spacing)
# # Customizing the x-axis
# xtick_positions = np.arange(1.5, n_models * n_symptoms + 1, n_symptoms)
plt.xticks(xtick_positions, model_names)

# %%
##############################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Assuming 'symptoms' is a list of five symptoms
model_accuracies = {model: {symptom: [] for symptom in symptoms} for model in ["7b", "13b", "70b"]}

for symptom in symptoms:
    for model in ["7b", "13b", "70b"]:
        pred_df = result_json_to_df(f"results-{model}_binary_p1.jsonl")
        df = gt_df.merge(pred_df, on="report", suffixes=[None, " pred"])

        accuracies = []
        for _ in range(100):
            sample_df = df.sample(frac=1, replace=True)
            y_true = sample_df[symptom]
            y_pred = sample_df[f"{symptom} pred"]
            accuracies.append(accuracy_score(y_true, y_pred))

        model_accuracies[model][symptom] = accuracies

# Plotting
plt.figure(figsize=(15, 6))

n_models = len(model_accuracies)
n_symptoms = len(symptoms)
model_names = list(model_accuracies.keys())
colors = ['blue', 'green', 'red', 'cyan', 'magenta']  # Assuming 5 symptoms

# Adjusting positions for closer grouping
group_width = 0.8  # Width of each group of boxplots
spacing = 1.0     # Spacing between each group

positions = []
current_position = 1
for _ in range(n_models):
    for _ in range(n_symptoms):
        positions.append(current_position)
        current_position += group_width / n_symptoms
    current_position += spacing

# Preparing data and plotting each boxplot
for i, model in enumerate(model_names):
    for j, symptom in enumerate(symptoms):
        data = model_accuracies[model][symptom]
        plt.boxplot(data, positions=[positions[i * n_symptoms + j]], widths=group_width / n_symptoms, 
                    patch_artist=True, boxprops=dict(facecolor=colors[j]))

# Customizing the x-axis
xtick_positions = np.arange(1 + (group_width / 2) * n_symptoms - (group_width / 2), 
                            n_models * (group_width + spacing), 
                            group_width + spacing)
plt.xticks(xtick_positions, model_names, fontsize=14)

plt.title("Model Accuracies for Each Symptom by Model", fontsize=16)
plt.ylim(0, 1)

# Adding legend
legend_elements = [Patch(facecolor=color, label=symptom) for color, symptom in zip(colors, symptoms)]
plt.legend(handles=legend_elements, title="Symptoms", title_fontsize='13', fontsize='12')

plt.show()
