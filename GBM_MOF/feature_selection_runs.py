from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from data_preprocessing import transform_data, load_data, scale_features, load_data_cv
from training_procedures import make_folds, run_GBM, cross_validation_GBM
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
import tqdm


import time
start_time = time.time()

"""
The plots will need to be modified with respect to target for ylabel. 
"""

def set_font_size_figures(fontsize):
    params = {'legend.fontsize': fontsize,
         'axes.labelsize': fontsize,
         'axes.titlesize': fontsize,
         'xtick.labelsize': fontsize,
         'ytick.labelsize': fontsize}
    plt.rcParams.update(params)

set_font_size_figures("medium")

# differenciate the trainings
prop = 'nest1000_dtree5_dac2_vol'
n_estimators = 1000
max_depth = 5
target = "Volumetric_Uptake_g_L"               #"Gravimetric_Uptake_wt%_g_g" 
depth_ac = 2
type_GBM = 'MS'  ### Both MS, Multiplication M, Substraction S, Linkers, L, SBU SBU, M_geom multiplicity with geomtry
# MS_geom both multiplicity and substraction with geometry

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")

"""
Loading figure and data path.
"""
path_to_here = os.getcwd()

data_path = "/data/"

fig_path = path_to_here + f"/results_gravimetric/feature_importance/figures/{prop}/"
data_saving_path = path_to_here + f"/results_gravimetric/feature_importance/data/{prop}/"
path_saving_features = path_to_here + f"/results_gravimetric/reduced_autocorrelation_vectors/{prop}/"
os.makedirs(fig_path, exist_ok=True)
os.makedirs(data_saving_path, exist_ok=True)
os.makedirs(path_saving_features, exist_ok=True)

# Optionally, use this to create directories
create_directory_if_not_exists(fig_path)
create_directory_if_not_exists(data_saving_path)
create_directory_if_not_exists(path_saving_features)

"""
Names of autocorrelation vectors
"""
# Names of the new data AABBA_MOF
ac_MOF = "core8268_ddmof3090_H2mgpg_80_20_bar_geo_aabba.csv"
target_path = data_path + ac_MOF

"""
Change periodic to whatever you want to run here, and target can be changed to
'target_distance' if you want to test on the prediction of the H---H distances
instead.
"""
df  = load_data_cv(data_path + ac_MOF, target_path, target, depth_ac, type_GBM)

print('New training...')
params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": 0.05,
        "loss": "squared_error",
        #"min_samples_split": 6, 
        #"min_samples_leaf": 6 #"min_weight_fraction_leaf": 0.03,
        }

# Number of folds in cross-validation
k = 5

"""
k-fold cross-validation
"""
fis, mses_tr, mses, maes, maes_tr, r2s_tr, r2s_te, best_data = cross_validation_GBM(k, df, "target", params)
test_size = len(best_data[0]["true"])
preds_and_truths = {}
for run, best_data_run in enumerate(best_data):
    data_preds = best_data_run["pred"]
    data_truth = best_data_run["true"].to_numpy()
    #print(type(data_truth))
    if len(data_truth) != test_size:
        data_preds = np.concatenate((data_preds, np.array([10])))
        data_truth = np.concatenate((data_truth, np.array([10])))
    #print("preds length: ", len(data_preds))
    #print("truth length: ", len(data_truth))
    preds_and_truths[f"preds_{str(run+1)}"] = data_preds
    preds_and_truths[f"truths_{str(run+1)}"] = data_truth

"""
Save predictions and truths data
"""
#print(preds_and_truths)
df_preds_and_truths = pd.DataFrame(data=preds_and_truths)
df_preds_and_truths.to_csv(data_saving_path + f"GP_predsVtruths_{target}_{n_estimators}_dt{max_depth}_dac{depth_ac}.csv")

"""
Scatter Preds vs Truth
"""
for i in range(k):
    fig = plt.figure()
    plt.scatter(preds_and_truths[f"truths_{str(i+1)}"],preds_and_truths[f"preds_{str(i+1)}"], color="tab:blue", alpha=0.3)
    plt.xlabel("DFT calculated absorption")
    plt.ylabel("Predicted absorption")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(fig_path + f"GP_GBM_pairplot_{target}_{n_estimators}_{i:.0f}_dt{max_depth}_dac{depth_ac}.pdf", format="pdf", bbox_inches="tight")

"""
Saving data about runs
"""

argmins = np.zeros(k)
for i in range(k):
    argmins[i] = int(np.argmin(maes[i]))
data_dict_info = {"run": [i+1 for i in range(k)],
                      "best_mae": [maes[i][int(argmins[i])] for i in range(k)],
                      "best_mae_train": [maes_tr[i][int(argmins[i])] for i in range(k)],
                      "best_mse": [mses[i][int(argmins[i])] for i in range(k)],
                      "best_mse_train": [mses_tr[i][int(argmins[i])] for i in range(k)],
                      "best_r2": [r2s_te[i][int(argmins[i])] for i in range(k)],
                      "best_r2_train": [r2s_tr[i][int(argmins[i])] for i in range(k)]
                     }

df_run_specifics = pd.DataFrame(data=data_dict_info)
df_run_specifics.to_csv(data_saving_path + f"GP_{target}_{n_estimators}_dt{max_depth}__dac{depth_ac}_run_specifications.csv")

data_dict_run_info = {"boosting_iteration": [i+1 for i in range(params["n_estimators"])]}

for i in range(k):
    data_dict_run_info[f"mae_run_{i+1}"] = maes[i]
    data_dict_run_info[f"mae_train_run_{i+1}"] = maes_tr[i]
    data_dict_run_info[f"r2_run_{i+1}"] = r2s_te[i]
    data_dict_run_info[f"r2_train_run_{i+1}"] = r2s_tr[i]
    data_dict_run_info[f"mse_run_{i+1}"] = mses[i]
    data_dict_run_info[f"mse_train_run_{i+1}"] = mses_tr[i]

df_run_data = pd.DataFrame(data=data_dict_run_info)
df_run_data.to_csv(data_saving_path + f"GP_{target}_{n_estimators}_dt{max_depth}__dac{depth_ac}_runs.csv")

"""
Calculating standard errors of the means
"""
r2_test_seotm = np.std(r2s_te, axis=0)/np.sqrt(k)
r2_train_seotm = np.std(r2s_tr, axis=0)/np.sqrt(k)
mae_test_seotm = np.std(maes, axis=0)/np.sqrt(k)
mae_train_seotm = np.std(maes_tr, axis=0)/np.sqrt(k)
mse_test_seotm = np.std(mses, axis=0)/np.sqrt(k)
mse_train_seotm = np.std(mses_tr, axis=0)/np.sqrt(k)

"""
Feature importances
"""
feature_importances = np.mean(fis, axis=0)
feature_importances_seotm = np.std(fis, axis=0)/np.sqrt(k)

"""
R2 plot
"""
fig = plt.figure()
ax = plt.gca()
plt.plot([i+1 for i in range(params["n_estimators"])],
         np.mean(r2s_tr, axis=0),
         color="tab:blue",
         label="R2 training")
ax.fill_between([i+1 for i in range(params["n_estimators"])],
                np.mean(r2s_tr, axis=0) - 1.96*r2_train_seotm,
                np.mean(r2s_tr, axis=0) + 1.96*r2_train_seotm,
                color="tab:blue",
                alpha=0.3,
                label="95% CI training")
plt.plot([i+1 for i in range(params["n_estimators"])],
         np.mean(r2s_te, axis=0),
         color="tab:orange",
         label="R2 testing")
ax.fill_between([i+1 for i in range(params["n_estimators"])],
                np.mean(r2s_te, axis=0) - 1.96*r2_test_seotm,
                np.mean(r2s_te, axis=0) + 1.96*r2_test_seotm,
                color="tab:orange",
                alpha=0.3,
                label="95% CI testing")
plt.legend(loc="lower right")
plt.xlabel("Boosting iterations")
plt.ylabel(r"R$^2$ Gavimetric uptake ")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(fig_path + f"GP_R2_GBR_{target}_{n_estimators}_dt{max_depth}_dac{depth_ac}.pdf", format="pdf", bbox_inches="tight")
plt.show()
"""
MSE plot
"""
fig = plt.figure()
ax = plt.gca()
plt.plot([i+1 for i in range(params["n_estimators"])],
         np.mean(mses_tr, axis=0),
         color="tab:blue",
         label="MSE train")
ax.fill_between([i+1 for i in range(params["n_estimators"])],
                np.mean(mses_tr, axis=0) - 1.96*mse_train_seotm,
                np.mean(mses_tr, axis=0) + 1.96*mse_train_seotm,
                color="tab:blue",
                alpha=0.3,
                label="95% CI training")
plt.plot([i+1 for i in range(params["n_estimators"])],
         np.mean(mses_tr, axis=0),
         color="tab:orange",
         label="MSE test")
ax.fill_between([i+1 for i in range(params["n_estimators"])],
                np.mean(mses, axis=0) - 1.96*mse_test_seotm,
                np.mean(mses, axis=0) + 1.96*mse_test_seotm,
                color="tab:orange",
                alpha=0.3,
                label="95% CI testing")
plt.legend(loc="upper right")
plt.xlabel("Boosting iterations")
plt.ylabel(r"MSE Gavimetric uptake [wt%]")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(fig_path + f"GP_MSE_GBR_{target}_{n_estimators}_dt{max_depth}_dac{depth_ac}.pdf", format="pdf", bbox_inches="tight")
plt.show()
"""
MAE plot
"""
fig = plt.figure()
ax = plt.gca()
plt.plot([i+1 for i in range(params["n_estimators"])],
         np.mean(maes_tr, axis=0),
         color="tab:blue",
         label="MAE train")
ax.fill_between([i+1 for i in range(params["n_estimators"])],
                np.mean(maes_tr, axis=0) - 1.96*mae_train_seotm,
                np.mean(maes_tr, axis=0) + 1.96*mae_train_seotm,
                color="tab:blue",
                alpha=0.3,
                label="95% CI training")
plt.plot([i+1 for i in range(params["n_estimators"])],
         np.mean(maes, axis=0),
         color="tab:orange",
         label="MAE test")
ax.fill_between([i+1 for i in range(params["n_estimators"])],
                np.mean(maes, axis=0) - 1.96*mae_test_seotm,
                np.mean(maes, axis=0) + 1.96*mae_test_seotm,
                color="tab:orange",
                alpha=0.3,
                label="95% CI testing")
plt.legend(loc="upper right")
plt.xlabel("Boosting iterations")
plt.ylabel(r"MAE Gavimetric uptake [wt%]")    
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(fig_path + f"GP_MAE_GBR_{target}_{n_estimators}_dt{max_depth}_dac{depth_ac}.pdf", format="pdf", bbox_inches="tight")
plt.show()
"""
Feature importance
"""
sorted_idx = feature_importances.argsort()
sorted_idx = np.flip(sorted_idx)
pos = np.arange(sorted_idx[0:20].shape[0]) + 0.5

relevance_dictionary = {"feature": df.columns[sorted_idx],
                        "relevance": feature_importances[sorted_idx],
                        "relevance_seotm": feature_importances_seotm[sorted_idx]}
relevance_data = pd.DataFrame(data=relevance_dictionary)
relevance_data.to_csv(path_or_buf=path_saving_features + f"/gp_relevance_{target}_{n_estimators}_dt{max_depth}_dac{depth_ac}.csv")

"""
Bar plot
"""
set_font_size_figures("medium")
fig = plt.figure()
plt.bar(pos, 100*feature_importances[sorted_idx[0:20]],
        align="center",
        color="tab:blue",
        yerr=100*feature_importances_seotm[sorted_idx[0:20]],
        linewidth=1,
        edgecolor="white",
        alpha=1.0)
plt.xticks(pos, np.array(df.columns[sorted_idx[0:20]]))
plt.ylabel("AC descriptor importance in GBM for Gavimetric uptake [%]", fontsize="medium")
plt.xlabel("AC descriptor", fontsize="medium")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
plt.xticks(rotation="vertical")
plt.savefig(fig_path + f"GP_feature_importance_{target}_{n_estimators}_dt{max_depth}_dac{depth_ac}.pdf", format="pdf", bbox_inches="tight")
plt.show()
best_idx = np.argmin(np.mean(maes, axis=0))
print("-"*50)
print(f"Best MAE Gavimetric uptake periodic set: {np.mean(maes, axis=0)[best_idx]}+-{1.96*mae_test_seotm[best_idx]}")
print(f"Best MAE Gavimetric uptake training periodic set: {np.mean(maes_tr, axis=0)[best_idx]}+-{1.96*mae_train_seotm[best_idx]}")
print(f"Best R2 Gavimetric uptake periodic set: {np.mean(r2s_te, axis=0)[best_idx]}+-{1.96*r2_test_seotm[best_idx]}")
print(f"Best R2 Gavimetric uptake training periodic set: {np.mean(r2s_tr, axis=0)[best_idx]}+-{1.96*r2_train_seotm[best_idx]}")
print("-"*50)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
