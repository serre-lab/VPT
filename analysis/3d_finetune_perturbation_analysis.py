import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from glob import glob
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib import patches
import utils 


# Hyperparams
plot_data = False
use_filters = False
CI = 34  # SEM

# Folders/colors
data_folder = "perspective_data"
vpt_ps_folder = "prolific_ps_vpt"
depth_ps_folder = "prolific_ps_depth"
vpt_color = "#1C75BC"
depth_color = "#92278F"

# Get participant files
vpt_files = glob(os.path.join(data_folder, vpt_ps_folder, "*.csv"))
depth_files = glob(os.path.join(data_folder, depth_ps_folder, "*.csv"))

# Get Peisen's stim list
peisen_list = pd.read_csv(os.path.join(data_folder, "human_balanced.csv"))

# Get model data
model_performance_vpt = pd.read_csv(os.path.join(data_folder, "Perspective_Results_perspective_finetune_full.csv"))
model_performance_depth = pd.read_csv(os.path.join(data_folder, "Perspective_Results_depth_finetune_full.csv"))
raytracing_performance = pd.read_csv(os.path.join(data_folder, "Perspective_Results_view_exp_finetune_preds.csv"))

color_key = {
    "True_CNN": "#ff6961",
    "False_CNN": "#ffb480",
    "False_Diffusion": "#ada6a0",
    "False_LLM": "#ad631d",
    "False_Transformer": "#08cad1",
    "True_Transformer": "#59adf6",
}
vpt_df, vpt_floor, vpt_sub_res, vpt_sub_lab, vpt_sub_file = utils.process_human_data(vpt_files, flip=True)
depth_df, depth_floor, depth_sub_res, depth_sub_lab, depth_sub_file = utils.process_human_data(depth_files)
# vpt_025, vpt_975, vpt_500 = np.percentile(vpt_floor, 2.5), np.percentile(vpt_floor, 97.5), np.percentile(vpt_floor, 50)
# depth_025, depth_975, depth_500 = np.percentile(depth_floor, 2.5), np.percentile(depth_floor, 97.5), np.percentile(depth_floor, 50)

# Filter subjects if using mturk
trim_vpt_df, trim_depth_df = utils.filter_data(vpt_df, depth_df, use_filters)

# Filter subjects to match peisen's splits
filtered_trim_vpt_df = utils.filter_ps(trim_vpt_df, peisen_list)
filtered_trim_depth_df = utils.filter_ps(trim_depth_df, peisen_list)

# Get human moments
vpt_true_mean, vpt_high_ci, vpt_low_ci = utils.get_human_moments(filtered_trim_vpt_df, ci=CI)
depth_true_mean, depth_high_ci, depth_low_ci = utils.get_human_moments(filtered_trim_depth_df, ci=CI)

# Make a combined human df
vpt_sub_df = pd.DataFrame(np.stack((filtered_trim_vpt_df.test_perf.values.astype(np.float32), filtered_trim_vpt_df.subject.values.astype(np.float32)), 1), columns=["Performance", "Subject"])
vpt_means = vpt_sub_df.groupby("Subject").mean().values
depth_sub_df = pd.DataFrame(np.stack((filtered_trim_depth_df.test_perf.values.astype(np.float32), filtered_trim_depth_df.subject.values.astype(np.float32)), 1), columns=["Performance", "Subject"])
depth_means = depth_sub_df.groupby("Subject").mean().values
X = np.concatenate((depth_means, vpt_means))
y = np.concatenate((["Depth perception"] * len(depth_means), ["VPT"] * len(vpt_means)))
y = y.reshape(-1, 1)
human_combined_df = pd.DataFrame(np.concatenate((X, y), 1), columns=["Human accuracy", "Task"])
human_combined_df["Human accuracy"] = pd.to_numeric(human_combined_df["Human accuracy"])

# Convert models from wide to long
model_melted_depth = pd.melt(
    model_performance_depth,
    id_vars=['img_name', 'depth_label'],
    value_vars=model_performance_depth.columns[2:],
    value_name='score',
    var_name='model_name')
model_melted_depth["correct"] = (model_melted_depth.depth_label == (model_melted_depth.score > 0)).astype(np.float32)
model_depth_correct = model_melted_depth.groupby("model_name").mean("correct").reset_index()
model_depth_err = model_melted_depth[["model_name", "correct"]].groupby("model_name").agg(lambda x: x.std() / np.sqrt(float(len(x)))).reset_index()
model_melted_vpt = pd.melt(
    model_performance_vpt,
    id_vars=['img_name', 'vpt_label'],
    value_vars=model_performance_vpt.columns[2:],
    value_name='score',
    var_name='model_name')
model_melted_vpt["correct"] = (model_melted_vpt.vpt_label == (model_melted_vpt.score > 0)).astype(np.float32)
model_vpt_correct = model_melted_vpt.groupby("model_name").mean("correct").reset_index()
model_vpt_err = model_melted_vpt[["model_name", "correct"]].groupby("model_name").agg(lambda x: x.std() / np.sqrt(float(len(x)))).reset_index()

#### Plot ray tracing vs. test set perf
rt_labs = raytracing_performance.values[:, 1]
rt_humans = raytracing_performance.values[:, 2].astype(np.float32)
rt_decs = (raytracing_performance.values[:, 3:].astype(np.float32) > 0).astype(np.float32)
rt_files = np.asarray([x.split("_")[0] for x in raytracing_performance.values[:, 0]]).reshape(-1, 1)
rt_human_correct = (rt_humans == rt_labs).astype(np.float32).reshape(-1, 1)
rt_dnn_correct = (rt_decs == rt_labs.reshape(-1, 1)).astype(np.float32)
rt_df = pd.DataFrame(np.concatenate((rt_files, rt_labs.reshape(-1, 1), rt_human_correct, rt_dnn_correct), 1), columns=raytracing_performance.columns)
rt_df_summary = rt_df.groupby("img_name").mean().reset_index()
human_perf = rt_df_summary.values[:, 2].mean()
human_se = rt_df_summary.values[:, 2].std() / np.sqrt(float(len(rt_df_summary.values[:, 2])))

rt_melted = pd.melt(
    rt_df_summary,
    id_vars=['img_name'],
    value_vars=rt_df_summary.columns[3:],
    value_name='rt_score',
    var_name='model_name')
rt_melted = rt_melted.groupby("model_name")["rt_score"].mean().reset_index()
models = rt_melted.model_name.values
model_label, aligned_ft = [], []
for m in models:
    if "in2" in m:
        data = True
    else:
        data = False
    
    if "vgg" in m or "resnet" in m or "hrnet" in m:
        mtp = "CNN"
    else:
        mtp = "Transformer"
    model_label.append("{}_{}".format(data, mtp))

    # Also align model_vpt_correct with rt_exp
    ft_score = model_vpt_correct.correct[model_vpt_correct.model_name == m].values.squeeze().astype(np.float32)
    aligned_ft.append(ft_score)
rt_melted["model_label"] = np.asarray(model_label)
rt_melted["ft_score"] = np.asarray(aligned_ft)
# rt_melted.rt_score = pd.to_numeric(rt_melted.rt_score)
# rt_melted.ft_score = pd.to_numeric(rt_melted.ft_score)

xlim, ylim = [0.4, 1], [0.4, 1]

f = plt.figure()
ax = sns.scatterplot(data=rt_melted, x="ft_score", y="rt_score", hue="model_label", palette=color_key, legend=False, s=40)
# vpt_rect = utils.plot_human_perf(vpt_true_mean, vpt_high_ci, vpt_low_ci, ylim, facecolor=vpt_color, transpose=True)
# rt_rect = utils.plot_human_perf(human_perf, human_perf + human_se, human_perf - human_se, xlim, facecolor=vpt_color, transpose=False)
vpt_kappa_ci = [vpt_low_ci, vpt_high_ci]
vpt_rect = patches.Rectangle(
    (vpt_kappa_ci[0], ylim[0]),
    vpt_kappa_ci[1] - vpt_kappa_ci[0],
    vpt_high_ci - ylim[0],
    linewidth=1,
    facecolor=vpt_color,
    alpha=0.25)
ax.add_patch(vpt_rect)
old_vpt_kappa_ci = np.copy(vpt_kappa_ci)
vpt_kappa_ci = [human_perf - human_se, human_perf + human_se]
rt_rect = patches.Rectangle(
    (xlim[0], vpt_low_ci),
    old_vpt_kappa_ci[0] - xlim[0],
    vpt_high_ci - vpt_low_ci,
    linewidth=1,
    facecolor=vpt_color,
    alpha=0.25)
ax.add_patch(rt_rect)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("VPT")
ax.set_ylabel("VPT-perturbations")
plt.show()

#### Get corrs
vpt_corrs, vpt_kappa_errs = utils.compute_ck(
    model_melted_vpt,
    vpt_sub_res,
    vpt_sub_file,
    vpt_sub_lab)
depth_corrs, depth_kappa_errs = utils.compute_ck(
    model_melted_depth,
    depth_sub_res,
    depth_sub_file,
    depth_sub_lab)

#### Get human ceil/floor
depth_ceil, depth_floor = utils.human_kappas(depth_sub_res, depth_sub_lab, depth_sub_file)
vpt_ceil, vpt_floor = utils.human_kappas(vpt_sub_res, vpt_sub_lab, vpt_sub_file)

# Plot model acc vs. correlation
xlim = [-0.2, 0.5]
ylim = [0.4, 1]
f, axs = plt.subplots(1, 1)

# plt.vlines(
#     np.mean(depth_ceil),
#     xlim[0],
#     np.mean(depth_true_mean),
#     linestyles="dotted",
#     colors=depth_color,
#     alpha=0.5)
# plt.vlines(
#     np.mean(vpt_ceil),
#     xlim[0],
#     np.mean(vpt_true_mean),
#     linestyles="dotted",
#     colors=vpt_color,
#     alpha=0.5)
# plt.hlines(
#     np.mean(depth_true_mean),
#     xlim[0],
#     np.mean(depth_ceil),
#     linestyles="dotted",
#     colors=depth_color,
#     alpha=0.5)
# plt.hlines(
#     np.mean(vpt_true_mean),
#     xlim[0],
#     np.mean(vpt_ceil),
#     linestyles="dotted",
#     colors=vpt_color,
#     alpha=0.5)
# Add human patches to axes
margin = (100 - float(CI)) / 2
high_ci = 100 - margin
low_ci = margin
vpt_kappa_ci = [np.percentile(vpt_ceil, low_ci), np.percentile(vpt_ceil, high_ci)]  # +/- 1SE
rect = patches.Rectangle(
    (vpt_kappa_ci[0], ylim[0]),
    vpt_kappa_ci[1] - vpt_kappa_ci[0],
    vpt_high_ci - ylim[0],
    linewidth=1,
    facecolor=vpt_color,
    alpha=0.25)
axs.add_patch(rect)
rect = patches.Rectangle(
    (xlim[0], vpt_low_ci),
    vpt_kappa_ci[0] - xlim[0],
    vpt_high_ci - vpt_low_ci,
    linewidth=1,
    facecolor=vpt_color,
    alpha=0.25)
axs.add_patch(rect)

depth_kappa_ci = [np.percentile(depth_ceil, low_ci), np.percentile(depth_ceil, high_ci)]  # +/- 1SE
rect = patches.Rectangle(
    (depth_kappa_ci[0], ylim[0]),
    depth_kappa_ci[1] - depth_kappa_ci[0],
    depth_high_ci - ylim[0],
    linewidth=1,
    facecolor=depth_color,
    alpha=0.25)
axs.add_patch(rect)
rect = patches.Rectangle(
    (xlim[0], depth_low_ci),
    depth_kappa_ci[0] - xlim[0],
    depth_high_ci - depth_low_ci,
    linewidth=1,
    facecolor=depth_color,
    alpha=0.25)
axs.add_patch(rect)

# Plot data
plt.scatter(depth_corrs, model_depth_correct.correct.values, color=depth_color, label="Depth perception", s=20)
plt.scatter(vpt_corrs, model_vpt_correct.correct.values, color=vpt_color, label="VPT", s=20)
axs.errorbar(depth_corrs, model_depth_correct.correct.values, yerr=model_depth_err.correct.values, xerr=depth_kappa_errs, elinewidth=1, alpha=0.1, ecolor=depth_color, ls='none')
axs.errorbar(vpt_corrs, model_vpt_correct.correct.values, yerr=model_vpt_err.correct.values, xerr=vpt_kappa_errs, elinewidth=1, alpha=0.1, ecolor=vpt_color, ls='none')

# Print corrs
print(np.corrcoef(depth_corrs, model_depth_correct.correct.values)[0, 1])
print(np.corrcoef(vpt_corrs, model_vpt_correct.correct.values)[0, 1])

# Adjust fig
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel("Cohen's Kappa")
plt.ylabel("Task accuracy")
# plt.legend(loc="upper left")
# plt.legend.remove()
plt.show()

# # Do the same for spearman
# # Analysis. Spearman corrs between average human and model
# depth_corrs = utils.spearman_alignment(model_performance_depth, depth_sub_res, depth_sub_file)
# vpt_corrs = utils.spearman_alignment(model_performance_vpt, vpt_sub_res, vpt_sub_file)

# #### Get human ceil/floor
# depth_ceil = utils.human_spearmans(depth_sub_res, depth_sub_file)
# vpt_ceil = utils.human_spearmans(vpt_sub_res, vpt_sub_file)
# depth_floor = utils.human_spearmans(depth_sub_res, depth_sub_file, shuffle=True)
# vpt_floor = utils.human_spearmans(vpt_sub_res, vpt_sub_file, shuffle=True)

# # Plot model acc vs. correlation
# ylim = [0.4, 1]
# f, axs = plt.subplots(1, 1)

# plt.vlines(
#     np.mean(depth_ceil),
#     xlim[0],
#     np.mean(depth_true_mean),
#     linestyles="dotted",
#     colors=depth_color,
#     alpha=0.5)
# plt.vlines(
#     np.mean(vpt_ceil),
#     xlim[0],
#     np.mean(vpt_true_mean),
#     linestyles="dotted",
#     colors=vpt_color,
#     alpha=0.5)
# plt.hlines(
#     np.mean(depth_true_mean),
#     xlim[0],
#     np.mean(depth_ceil),
#     linestyles="dotted",
#     colors=depth_color,
#     alpha=0.5)
# plt.hlines(
#     np.mean(vpt_true_mean),
#     xlim[0],
#     np.mean(vpt_ceil),
#     linestyles="dotted",
#     colors=vpt_color,
#     alpha=0.5)

# # Plot data
# plt.scatter(depth_corrs, model_depth_correct.correct.values, color=depth_color, label="Depth perception")
# plt.scatter(vpt_corrs, model_vpt_correct.correct.values, color=vpt_color, label="VPT")

# # Print corrs
# print(np.corrcoef(depth_corrs, model_depth_correct.correct.values)[0, 1])
# print(np.corrcoef(vpt_corrs, model_vpt_correct.correct.values)[0, 1])

# # Adjust fig
# plt.xlim(xlim)
# plt.ylim(ylim)
# plt.xlabel("Spearman's Rho")
# plt.ylabel("Task accuracy")
# plt.legend(loc="upper left")
# plt.show()

