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

#%%
# # Rename column names in the dataframe
# df = df.rename(columns={
#     'ascites': 'Ascites',
#     'abdominal pain': 'Abdominal pain',
#     'shortness of breath': 'Shortness of breath',
#     'confusion': 'Confusion',
#     'liver cirrhosis': 'Liver cirrhosis',
# })
# df.head()

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
# gt_df = pd.read_csv("/mnt/bulk/isabella/llamaproj/MIMIC_groundtruth_TF.csv")
gt_df = df
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
pred_df = result_json_to_df(f"results-70b_binary_SYS_rq_cotgrammar.jsonl")
pred_df.head()
#%%
df = gt_df.merge(pred_df, on="report", suffixes=[None, " pred"])


###########################################################################
# Use seaborn
##########################################################################
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

#%%
# Assuming 'df' and 'symptoms' are defined as in your context
for symptom in symptoms:
    y_true = df[symptom]
    y_pred = df[f"{symptom} pred"]

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')

     # Normalize the confusion matrix manually
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Convert to DataFrame for easier plotting
    cm_df = pd.DataFrame(cm, index=["False", "True"], columns=["False", "True"])

    # Plotting the confusion matrix using Seaborn
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(cm_df, annot=True, fmt=".2f", cmap='Blues', vmin=0, vmax=1,
                     annot_kws={"size": 28})  # Increase font size for annotations)
    plt.title(f'{symptom}', fontsize = 28)
    #plt.ylabel('Actual Values', fontsize=18)
    #plt.xlabel('Predicted Values', fontsize=18)

    # Set the font size for the tick labels (both axes)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 28)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 28)


     # Increase font size of the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=28)  # Adjust to preferred size

    plt.show()


# %%
#########################################################################
# Combine absolute values and fractions
#########################################################################
for symptom in symptoms:
    y_true = df[symptom]
    y_pred = df[f"{symptom} pred"]

    # Compute the confusion matrix (non-normalized for absolute numbers)
    cm_absolute = confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix for fractions
    cm_normalized = cm_absolute.astype('float') / cm_absolute.sum(axis=1)[:, np.newaxis]

    # Convert to DataFrame for easier plotting
    cm_df = pd.DataFrame(cm_normalized, index=["False", "True"], columns=["False", "True"])

    # Create annotations combining absolute numbers and fractions
    annotations = [["{0:d}\n({1:.2f})".format(abs_num, frac) for abs_num, frac in zip(row_abs, row_frac)] 
                    for row_abs, row_frac in zip(cm_absolute, cm_normalized)]

    # Plotting the confusion matrix using Seaborn with increased font sizes
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(cm_df, annot=annotations, fmt="", cmap='Blues', vmin=0, vmax=1, annot_kws={"size": 28})

    plt.title(f'{symptom.capitalize()}', fontsize=28)
    #plt.ylabel('Actual Values', fontsize=18)
    #plt.xlabel('Predicted Values', fontsize=18)

    # Set the font size for the tick labels (both axes)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 28)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 28)

    # Increase font size of the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=28)

    plt.show()

# %%
