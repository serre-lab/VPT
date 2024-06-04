import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from glob import glob
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib import patches
import utils 
from sklearn.metrics import cohen_kappa_score as cks
from scipy.stats import pearsonr


# Hyperparams
plot_data = True
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
model_performance = pd.read_csv(os.path.join(data_folder, "Perspective_Results_finetune_results_full.csv"))

color_key = {
    "True_CNN": "#ff6961",
    "False_CNN": "#ffb480",
    "False_Diffusion": "#ada6a0",
    "False_LLM": "#ad631d",
    "False_Transformer": "#08cad1",
    "True_Transformer": "#59adf6",
}

vpt_df, vpt_floor, vpt_sub_res, vpt_sub_lab, vpt_sub_files = utils.process_human_data(vpt_files, flip=True)
depth_df, depth_floor, depth_sub_res, depth_sub_lab, depth_sub_files = utils.process_human_data(depth_files)
margin = (100 - float(CI)) / 2
high_ci = 100 - margin
low_ci = margin
vpt_025, vpt_975, vpt_500 = np.percentile(vpt_floor, low_ci), np.percentile(vpt_floor, high_ci), np.percentile(vpt_floor, 50)
depth_025, depth_975, depth_500 = np.percentile(depth_floor, low_ci), np.percentile(depth_floor, high_ci), np.percentile(depth_floor, 50)

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

# Prepare DF
model_performance["ImageNet Accuracy"] = model_performance.imagenet_accuracy / 100.
model_performance["VPT Accuracy"] = model_performance.perspective_accuracy / 100.
model_performance["Depth Perception Accuracy"] = model_performance.depth_accuracy / 100.

# Copy model_performance and create one that is just tasks
task_performance = model_performance.copy()
task_performance = task_performance[task_performance.imagenet_accuracy > 0]

print("Depth pearson test: {}".format(pearsonr(task_performance["ImageNet Accuracy"], task_performance["Depth Perception Accuracy"])))
print("VPT pearson test: {}".format(pearsonr(task_performance["ImageNet Accuracy"], task_performance["VPT Accuracy"])))
import pdb;pdb.set_trace()

# 0. Create color key
training = model_performance.training_data == "ImageNet 1k"
models = model_performance.model_type.values
models[models == "MetaFormer"] = "Transformer"
models[models == "Hybrid"] = "Transformer"
training_models = np.asarray(["{}_{}".format(x, y) for x, y in zip(training, models)])
model_performance["color_key"] = training_models
training = task_performance.training_data == "ImageNet 1k"
models = task_performance.model_type.values
models[models == "MetaFormer"] = "Transformer"
models[models == "Hybrid"] = "Transformer"
training_models = np.asarray(["{}_{}".format(x, y) for x, y in zip(training, models)])
task_performance["color_key"] = training_models

if plot_data:
    ##### 0. human depth vs. perspective taking acc
    xlim = [0.6, 1]
    f, axs = plt.subplots(1, 1)

    depth_rect, vpt_rect = utils.get_noise_floors(
        depth_025,
        depth_975,
        vpt_025,
        vpt_975,
        depth_color=depth_color,
        vpt_color=vpt_color
    )
    axs.add_patch(depth_rect)
    axs.add_patch(vpt_rect)
    # plt.hlines(0., xlim[0], xlim[1], linestyles="dashed", colors="black", alpha=0.5)

    # Plot
    spec = dict(
        x="Task",
        y="Human accuracy",
        data=human_combined_df,
            palette={
            "Depth perception": depth_color,
            "VPT": vpt_color
        })
    sns.stripplot(**spec, alpha=0.5)
    sns.pointplot(
        **spec,
        linestyle='none',
        ci=0,
        scale=.5,
        capsize=.7)
    plt.ylim([0.4, 1])
    plt.title("Human performance")
    plt.show()
    plt.close(f)

    ##### 1. human vs. model acc on perspective taking
    xlim = [0.6, 1]
    f, axs = plt.subplots(1, 1)

    # Add human patches to the Axes
    vpt_rect = utils.plot_human_perf(vpt_true_mean, vpt_high_ci, vpt_low_ci, xlim, facecolor=vpt_color)
    axs.add_patch(vpt_rect)

    # Plot chance
    plt.hlines(vpt_500, xlim[0], xlim[1], linestyles="dashed", colors="black", alpha=0.5)

    # Plot
    ax = sns.scatterplot(
        data=task_performance,
        hue="color_key",
        x="ImageNet Accuracy",
        y="VPT Accuracy",
        palette=color_key,
        legend=False)  # , ax=axs)
    plt.ylim([0.4, 1])
    plt.xlim(xlim)
    plt.title("VPT")
    plt.show()
    plt.close(f)

    ##### 2. human vs. model acc on depth decoding
    xlim = [0.6, 1]
    f, axs = plt.subplots(1, 1)

    # Add human patches to the Axes
    depth_rect = utils.plot_human_perf(depth_true_mean, depth_high_ci, depth_low_ci, xlim, facecolor=depth_color)
    axs.add_patch(depth_rect)

    # Plot chance
    plt.hlines(depth_500, xlim[0], xlim[1], linestyles="dashed", colors="black", alpha=0.5)

    # Plot
    ax = sns.scatterplot(
        data=task_performance,
        hue="color_key",
        x="ImageNet Accuracy",
        y="Depth Perception Accuracy",
        palette=color_key,
        legend=False)  # , ax=axs)
    plt.ylim([0.4, 1])
    plt.xlim(xlim)
    plt.title("Depth perception")
    plt.show()
    plt.close(f)

    ##### 3. Both tasks
    xlim = [0.6, 1]
    f, axs = plt.subplots(1, 1)

    # Add human patches to axes
    vpt_rect = utils.plot_human_perf(vpt_true_mean, vpt_high_ci, vpt_low_ci, xlim, facecolor=vpt_color)
    depth_rect = utils.plot_human_perf(depth_true_mean, depth_high_ci, depth_low_ci, xlim, facecolor=depth_color)
    axs.add_patch(vpt_rect)
    axs.add_patch(depth_rect)

    # Plot chance
    plt.hlines(0.5 * (vpt_500 + depth_500), xlim[0], xlim[1], linestyles="dashed", colors="black", alpha=0.5)

    # Plot
    ax = sns.regplot(data=task_performance, truncate=False, x="ImageNet Accuracy", y="VPT Accuracy", color="#1C75BC", order=1, scatter_kws=dict(alpha=0.5, s=12))  # , ax=axs)
    ax = sns.regplot(data=task_performance, truncate=False, x="ImageNet Accuracy", y="Depth Perception Accuracy", color="#92278F", order=1, scatter_kws=dict(alpha=0.5, s=12))  # , ax=axs)
    plt.ylim([0.4, 1])
    plt.xlim(xlim)
    plt.title("VPT and depth perception")
    plt.show()
    plt.close(f)

    ##### 4. Plot tasks against each other
    xlim = [0.45, 1]
    f, axs = plt.subplots(1, 1)
    # plt.plot([0.45, 1], [0.45, 1], "k--", alpha=0.5)

    # Add human patches to axes
    vpt_rect = utils.plot_human_perf(vpt_true_mean, vpt_high_ci, vpt_low_ci, xlim, facecolor=vpt_color)
    depth_rect = utils.plot_human_perf(depth_true_mean, depth_high_ci, depth_low_ci, xlim, facecolor=depth_color, transpose=True)
    axs.add_patch(vpt_rect)
    axs.add_patch(depth_rect)

    # Plot
    ax = sns.scatterplot(
        data=model_performance,
        x="Depth Perception Accuracy",
        y="VPT Accuracy",
        palette=color_key,
        hue="color_key",
        legend=False)  # , ax=axs)
    plt.ylim([0.45, 1])
    plt.xlim(xlim)
    plt.title("VPT and depth perception")
    plt.show()
    plt.close(f)


# ANALYSES TO DO:
# 1. Stat test of perspective taking vs. chance on the individual model level
# 2. Stat test of depth vs. chance on the individual model level. CORRECT FOR MULTIPLE COMPARISONS
# 3. Per-model-class scaling laws for Depth perception and perspective taking (latter will be n.s.)
# 4. Stat test of Depth vs. Perspective. Randomization test.

# Analysis 4. Find images that are easy or hard for humans
sorted_by_acc_depth = filtered_trim_depth_df.groupby("stimuli").mean().sort_values("test_perf").reset_index()
sorted_by_acc_vpt = filtered_trim_vpt_df.groupby("stimuli").mean().sort_values("test_perf").reset_index()
sorted_by_acc_depth.to_csv("sorted_by_acc_depth.csv")
sorted_by_acc_vpt.to_csv("sorted_by_acc_vpt.csv")

# Analysis 5. Spearman corrs between average human and model
model_scores_depth = pd.read_csv(os.path.join(data_folder, "Perspective_Results_depth_finetune.csv"))
model_scores_vpt = pd.read_csv(os.path.join(data_folder, "Perspective_Results_perspective_finetune.csv"))
# depth_rhos = utils.spearman_alignment(model_scores_depth, depth_sub_res, depth_sub_files)
# vpt_rhos = utils.spearman_alignment(model_scores_vpt, vpt_sub_res, vpt_sub_files)

# #### Get human ceil/floor
# rho_depth_ceil = utils.human_spearmans(depth_sub_res, depth_sub_files)
# rho_vpt_ceil = utils.human_spearmans(vpt_sub_res, vpt_sub_files)
# rho_depth_floor = utils.human_spearmans(depth_sub_res, depth_sub_files, shuffle=True)
# rho_vpt_floor = utils.human_spearmans(vpt_sub_res, vpt_sub_files, shuffle=True)

# Analysis 6. Cohen's kappa corrs between average human and model
model_melted_depth = pd.melt(
    model_scores_depth,
    id_vars=['img_name', 'depth_label'],
    value_vars=model_scores_depth.columns[2:],
    value_name='score',
    var_name='model_name')
model_melted_depth["correct"] = (model_melted_depth.depth_label == (model_melted_depth.score > 0)).astype(np.float32)
model_depth_correct = model_melted_depth.groupby("model_name").mean("correct").reset_index()
model_depth_err = model_melted_depth[["model_name", "correct"]].groupby("model_name").agg(lambda x: x.std() / np.sqrt(float(len(x)))).reset_index()
model_melted_vpt = pd.melt(
    model_scores_vpt,
    id_vars=['img_name', 'vpt_label'],
    value_vars=model_scores_vpt.columns[2:],
    value_name='score',
    var_name='model_name')
model_melted_vpt["correct"] = (model_melted_vpt.vpt_label == (model_melted_vpt.score > 0)).astype(np.float32)
model_vpt_correct = model_melted_vpt.groupby("model_name").mean("correct").reset_index()
model_vpt_err = model_melted_vpt[["model_name", "correct"]].groupby("model_name").agg(lambda x: x.std() / np.sqrt(float(len(x)))).reset_index()
vpt_kappas, vpt_kappa_errs = utils.compute_ck(
    model_melted_vpt,
    vpt_sub_res,
    vpt_sub_files,
    vpt_sub_lab)
depth_kappas, depth_kappa_errs = utils.compute_ck(
    model_melted_depth,
    depth_sub_res,
    depth_sub_files,
    depth_sub_lab)

#### Get human ceil/floor
depth_ceil, depth_floor = utils.human_kappas(depth_sub_res, depth_sub_lab, depth_sub_files)
vpt_ceil, vpt_floor = utils.human_kappas(vpt_sub_res, vpt_sub_lab, vpt_sub_files)

# Analysis 7. Stat test of Depth vs. Perspective. Randomization test.
p = utils.randomization_test(X=model_performance["Depth Perception Accuracy"], Y=model_performance["VPT Accuracy"])
print("P value: {}".format(p))

# Plot model acc vs. correlation
xlim = [-0.1, 0.5]
ylim = [0.4, 1]
f, axs = plt.subplots(1, 1)

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

# # Plot lines
# plt.vlines(
#     np.mean(depth_ceil),
#     xlim[0],
#     depth_true_mean,
#     linestyles="dotted",
#     colors=depth_color,
#     alpha=0.5)
# plt.vlines(
#     np.mean(vpt_ceil),
#     xlim[0],
#     vpt_true_mean,
#     linestyles="dotted",
#     colors=vpt_color,
#     alpha=0.5)
# plt.hlines(
#     depth_true_mean,
#     xlim[0],
#     np.mean(depth_ceil),
#     linestyles="dotted",
#     colors=depth_color,
#     alpha=0.5)
# plt.hlines(
#     vpt_true_mean,
#     xlim[0],
#     np.mean(vpt_ceil),
#     linestyles="dotted",
#     colors=vpt_color,
#     alpha=0.5)

# Plot data
axs.scatter(depth_kappas, model_depth_correct.correct.values, color=depth_color, label="Depth perception", alpha=0.5, s=20)
axs.scatter(vpt_kappas, model_vpt_correct.correct.values, color=vpt_color, label="VPT", alpha=0.5, s=20)

# Plot errs
axs.scatter(depth_kappas, model_depth_correct.correct.values, color=depth_color, label="Depth perception", alpha=0.5, s=20)
axs.scatter(vpt_kappas, model_vpt_correct.correct.values, color=vpt_color, label="VPT", alpha=0.5, s=20)
axs.errorbar(depth_kappas, model_depth_correct.correct.values, yerr=model_depth_err.correct.values, xerr=depth_kappa_errs, elinewidth=1, alpha=0.1, ecolor=depth_color, ls='none')
axs.errorbar(vpt_kappas, model_vpt_correct.correct.values, yerr=model_vpt_err.correct.values, xerr=vpt_kappa_errs, elinewidth=1, alpha=0.1, ecolor=vpt_color, ls='none')

# Print corrs
print(np.corrcoef(depth_kappas, model_depth_correct.correct.values)[0, 1])
print(np.corrcoef(vpt_kappas, model_vpt_correct.correct.values)[0, 1])

# Adjust fig
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel("Cohen's Kappa")
plt.ylabel("Task accuracy")
# plt.legend(loc="upper left")
plt.show()