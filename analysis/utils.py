import os
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score as cks
from tqdm import tqdm
from scipy.stats import spearmanr
import statsmodels.api as sm


def bootstrap_cis(s, I=1000, ci=95):
    """Resample with replacement for I iterations"""
    n = len(s)
    t = []
    for _ in range(I):
        idx = np.random.randint(n)
        t.append(s[idx].mean())
    t = np.asarray(t)
    margin = (100 - float(ci)) / 2
    high_ci = 100 - margin
    low_ci = margin
    return np.percentile(t, high_ci), np.percentile(t, low_ci)


def error_consistency(p1, p2, labels):
    """Assuming a binary task with 50% base rate.
    
    p1: subject 1 vector of responses
    p2: subject 2 vector of responses
    labels: label for each exemplar viewed by p1/p2
    
    Implementing the Geirhos approach from R, detailed below:
    ```
        p1 = get.single.accuracy(dat1)
        p2 = get.single.accuracy(dat2)
        expected.consistency = p1*p2 + (1-p1)*(1-p2)

        equal.responses = dat1$is.correct == dat2$is.correct
        num.equal.responses = sum(equal.responses) # gives number of 'TRUE' values
        observed.consistency = num.equal.responses / num.trials

        if(observed.consistency==1.0) {
            cohens.kappa = 1.0
        } else {
            #estimate = cohen.kappa(x=cbind(dat1$is.correct, dat2$is.correct))
            #cohens.kappa = estimate$kappa
            cohens.kappa = (observed.consistency-expected.consistency) / (1.0-expected.consistency)
        }
    ```
    """
    # Make sure everything is in numpy
    p1 = np.asarray(p1).astype(float)
    p2 = np.asarray(p2).astype(float)
    labels = np.asarray(labels).astype(float)

    # Compute scores
    p1_acc = (p1 == labels).mean()
    p2_acc = (p2 == labels).mean()
    expected_consistency = p1_acc * p2_acc + ((1 - p1_acc) * (1 - p2_acc))

    p1_correct = p1 == labels
    p2_correct = p2 == labels

    equal_responses = p1_correct == p2_correct
    num_equal_responses = np.sum(equal_responses).astype(float)
    observed_consistency = float(num_equal_responses) / float(len(labels))

    if observed_consistency == 1:
        kappa = 1.
    else:
        kappa = (observed_consistency - expected_consistency) / (1. - expected_consistency)
    return kappa

def randomization_test(X, Y, N=10000):
    """Randomly permute the sign of X-Y for N iterations to find the probability of True X-Y > Permuted X-Y."""
    diff = X - Y  # True test score
    n = len(diff)
    T = np.mean(diff)
    t = []  # Permuted test scores
    for _ in range(N):
        sv = (np.random.rand(n) > 0.5).astype(np.float32)
        sv[sv == 0] = -1
        t.append((diff * sv).mean())
    t = np.asarray(t)
    p = (np.sum(t > T).astype(np.float32) + 1) / (float(N) + 1.)
    return p


def process_human_data(files, iterations=1000, flip=False):
    """
    Process human data from prolific, collected with Alekh Karkada's webapps.
    
    flip flag is for the human VPT data, where labels/decisions are flipped vs. models.
    """
    res, training, correct_res, subs, stimuli = [], [], [], [], []
    timeout, response = [], []
    sub_perms_res, sub_perms_lab, sub_perms_files = {}, {}, {}
    for idx, f in enumerate(files):
        data = pd.read_csv(f)
        if flip:
            data.response = 1 - data.response
            data.label = 1 - data.label
        sub_perms_res[data.worker_id.values[0]] = data.response
        sub_perms_lab[data.worker_id.values[0]] = data.label
        sub_perms_files[data.worker_id.values[0]] = data.video_url.values
        vids = [x.split(os.path.sep)[-2] for x in data.video_url]
        corrects = data["correctness"]
        times = data["time_taken"]
        stims = data["video_url"]
        X = np.stack((vids, corrects, times, stims), 1)
        df = pd.DataFrame(X, columns=["version", "correct", "times", "stimuli"])
        df.correct = pd.to_numeric(df.correct)
        df.times = pd.to_numeric(df.times)
        correct_res.append(df.correct.values)
        subs.append(idx * np.ones_like(df.correct.values))
        training.append(np.unique(data.correct_during_trial.values).max() * np.ones_like(df.correct.values))
        timeout.append(data.timeout.values.mean() * np.ones_like(df.correct.values))
        response.append(np.abs(np.diff(data.response)).mean() * np.ones_like(df.correct.values))
        stimuli.append(df.stimuli.values)

    # Make a data frame
    training = np.concatenate(training)
    corrects = np.concatenate(correct_res)
    subs = np.concatenate(subs)
    timeout = np.concatenate(timeout)
    response = np.concatenate(response)
    stimuli = np.concatenate(stimuli)
    X = np.stack((training.astype(np.float32) / training.max(), corrects, subs, timeout, response, stimuli), 1)
    df = pd.DataFrame(X, columns=["train_perf", "test_perf", "subject", "timeout", "response", "stimuli"])

    # Compute noise floor
    subjects = sub_perms_lab.keys()
    perms = []
    for i in range(iterations):
        accs = []
        for s in subjects:
            r = sub_perms_res[s].values.astype(np.float32)
            l = sub_perms_lab[s].values.astype(np.float32)
            l = l[np.random.permutation(len(l))]
            accs.append(np.mean(r == l))
        perms.append(np.mean(accs))
    noise_floor = np.asarray(perms)
    return df, noise_floor, sub_perms_res, sub_perms_lab, sub_perms_files


def filter_ps(trim_df, peisen_list):
    """Filter subject files to match our balanced splits."""
    participant_files = np.unique(trim_df.stimuli.values)
    remove_files = participant_files[~np.in1d(participant_files, peisen_list.img_url.values)]
    filtered_df = trim_df.copy()
    for r in remove_files:
        mask = filtered_df.stimuli.str.contains(r)
        filtered_df = filtered_df[~mask]
    print("Removed {} trials to match Peisen's list".format(len(trim_df) - len(filtered_df)))
    return filtered_df


def get_human_moments(filtered_df, ci=34):
    """Compute the mean + CIs of human data."""
    moments_df = pd.DataFrame(np.stack((filtered_df.test_perf.values.astype(np.float32), filtered_df.subject.values.astype(np.float32)), 1), columns=["Performance", "Subject"])
    sub_means = moments_df.groupby("Subject").mean().values
    true_mean = sub_means.mean()
    high_ci, low_ci = bootstrap_cis(sub_means, ci=ci)
    print("Human mean: {}, high_ci: {}, low_ci: {}".format(true_mean, high_ci, low_ci))
    return true_mean, high_ci, low_ci


def plot_human_perf(mu, high_ci, low_ci, xlim, facecolor="black", alpha=0.1, transpose=False):
    """Draw a rectangle for human performance."""
    if transpose:
        rect = patches.Rectangle(
            (low_ci, xlim[0]),
            high_ci - low_ci,
            xlim[1] - xlim[0],
            linewidth=1,
            facecolor=facecolor,
            alpha=alpha)
    else:
        rect = patches.Rectangle(
            (xlim[0], low_ci),
            xlim[1] - xlim[0],
            high_ci - low_ci,
            linewidth=1,
            facecolor=facecolor,
            alpha=alpha)
    return rect

def filter_data(vpt_df, depth_df, use_filters):
    if use_filters:
        raise NotImplemented("Not needed with prolific data.")
        # Plot training acc vs. performance
        grouped = df.groupby("subject").mean().reset_index()
        sns.scatterplot(data=grouped, x="train_perf", y="test_perf")
        plt.title("Training vs. test perf")
        plt.show()

        # Plot timeout acc vs. performance
        grouped = df.groupby("subject").mean().reset_index()
        sns.scatterplot(data=grouped, x="timeout", y="test_perf")
        plt.title("Timeouts vs. test perf")
        plt.show()

        # Plot response repeats vs. performance
        grouped = vpt_df.groupby("subject").mean().reset_index()
        sns.scatterplot(data=grouped, x="response", y="test_perf")
        plt.title("Button repeats vs. test perf")
        plt.show()

        # Find threshold for training perf
        train_threshold = grouped.train_perf.median()
        train_mask = grouped.train_perf > train_threshold

        low_repeat = grouped.response.mean() - (grouped.response.std() * 2)
        high_repeat = grouped.response.mean() + (grouped.response.std() * 2)
        repeat_mask = np.logical_and(grouped.response > low_repeat, grouped.response < high_repeat)

        mask = train_mask & repeat_mask
        keep_subs = grouped.subject.values[mask]

        keep_big_df = np.in1d(vpt_df.subject.values, keep_subs)
        trim_df = df[keep_big_df]
        print("Filtering {}".format(grouped.subject.values[~mask]))
    else:
        print("Not filtering")
        trim_vpt_df = vpt_df
        trim_depth_df = depth_df
    return trim_vpt_df, trim_depth_df


def get_noise_floors(depth_025, depth_975, vpt_025, vpt_975, depth_color, vpt_color, dx=-0.25, vx=0.75, dw=0.5, vw=0.5):
    """Get noise floor rectangles."""
    depth_rect = patches.Rectangle(
        (dx, depth_025),
        dw,
        depth_975 - depth_025,
        linewidth=1,
        facecolor=depth_color,
        alpha=0.1)
    vpt_rect = patches.Rectangle(
        (vx, vpt_025),
        vw,
        vpt_975 - vpt_025,
        linewidth=1,
        facecolor=vpt_color,
        alpha=0.1)
    return depth_rect, vpt_rect


def align_idx(model, human):
    """Find idx that reorders human files to match model files."""
    idx = []
    for h in human:
        idx.append(model.index(h))
    return np.asarray(idx)


def partial_corr(model_labels, model_res, human_res):
    """Get correlation of decisions holding accuracy constant."""
    # X = np.concatenate((sm.add_constant(model_labels), model_res), 1).astype(np.float32)
    X = np.stack((model_labels, model_res.ravel()), 1).astype(np.float32)
    logit_mod = sm.Logit(human_res.astype(np.float32).reshape(-1, 1), X)
    try:
        logit_res = logit_mod.fit()
    except:
        import pdb;pdb.set_trace()
    decision_coef = logit_res.params[-1]
    return decision_coef


def compute_ck(melt_model_df, sub_res, sub_files, sub_labels, errors=False, debug=False, input_type="logits"):
    ks, errs = [], []
    models = np.unique(melt_model_df.model_name)
    subs = np.unique([k for k in sub_res.keys()])
    for m in models:
        model_df = melt_model_df[melt_model_df.model_name == m]
        model_files = model_df.img_name.values
        model_scores = model_df.score.values
        try:
            model_labels = model_df.vpt_label.values
        except:
            model_labels = model_df.depth_label.values
        if input_type == "logits":
            model_res = (model_scores > 0).astype(np.float32)
        else:
            model_res = np.round(model_scores)

        if errors:
            # Focus on rows with errors by humans or models
            m_err = model_labels != model_res

        model_ks = []
        for s in subs:
            sub_file = sub_files[s]
            sub_file = np.asarray([os.path.sep.join(x.split(os.path.sep)[1:]) for x in sub_file.tolist()])
            sub_mask = np.in1d(sub_file, model_files)
            trim_sub_file = sub_file[sub_mask]
            model_reorder = align_idx(
                model_files.tolist(),
                trim_sub_file.tolist())
            assert (trim_sub_file == model_files[model_reorder]).mean() == 1, "Mismatch between model/human order found."
            human_res = sub_res[s][sub_mask].values
            human_lab = sub_labels[s][sub_mask].values
            it_model_res = model_res[model_reorder]
            if errors:
                # Focus on rows with errors by humans or models
                h_err = human_res != human_lab
                mask = np.logical_or(h_err, m_err[model_reorder])
                # Apply mask to data
                human_res = human_res[mask]
                it_model_res = it_model_res[mask]

            kappa = error_consistency(human_res, it_model_res, human_lab)
            # kappa = cks(human_res, it_model_res)
            # kappa = partial_corr(
            #     model_labels=model_labels[model_reorder],
            #     model_res=it_model_res.reshape(-1, 1),
            #     human_res=human_res
            # )
            if debug:
                if kappa < -0.75:
                    import pdb;pdb.set_trace()
                try:
                    model_acc = (model_labels[model_reorder] == model_res[model_reorder]).mean()
                except:
                    import pdb;pdb.set_trace()
                human_acc = (human_res == human_lab).mean()
                print("Model acc: {}, Human acc: {}, Kappa: {}".format(model_acc, human_acc, kappa))
                # import pdb;pdb.set_trace()
                # plt.subplot(121);plt.plot(it_model_res, label="model");plt.subplot(122);plt.plot(human_res, label="human");plt.show()
            model_ks.append(kappa)
        # ks.append(np.mean(model_ks))
        ks.append(np.median(model_ks))
        errs.append(np.std(model_ks) / np.sqrt(float(len(model_ks))))
    return np.asarray(ks), np.asarray(errs)


def compute_partial(melt_model_df, sub_res, sub_files, sub_labels, debug=False, input_type="logits"):
    ks = []
    models = np.unique(melt_model_df.model_name)
    subs = np.unique([k for k in sub_res.keys()])
    for m in models:
        model_df = melt_model_df[melt_model_df.model_name == m]
        model_files = model_df.img_name.values
        model_scores = model_df.score.values
        try:
            model_labels = model_df.vpt_label.values
        except:
            model_labels = model_df.depth_label.values
        if input_type == "logits":
            model_res = (model_scores > 0).astype(np.float32)
        else:
            model_res = np.round(model_scores)
        model_ks = []
        for s in subs:
            sub_file = sub_files[s]
            sub_file = np.asarray([os.path.sep.join(x.split(os.path.sep)[1:]) for x in sub_file.tolist()])
            sub_mask = np.in1d(sub_file, model_files)
            trim_sub_file = sub_file[sub_mask]
            model_reorder = align_idx(
                model_files.tolist(),
                trim_sub_file.tolist())
            assert (trim_sub_file == model_files[model_reorder]).mean() == 1, "Mismatch between model/human order found."
            human_res = sub_res[s][sub_mask].values
            human_lab = sub_labels[s][sub_mask].values
            it_model_res = model_res[model_reorder]
            kappa = partial_corr(
                model_labels=model_labels[model_reorder],
                model_res=it_model_res.reshape(-1, 1),
                human_res=human_res
            )
            if debug:
                if kappa < -0.75:
                    import pdb;pdb.set_trace()
                try:
                    model_acc = (model_labels[model_reorder] == model_res[model_reorder]).mean()
                except:
                    import pdb;pdb.set_trace()
                human_acc = (human_res == human_lab).mean()
                print("Model acc: {}, Human acc: {}, Kappa: {}".format(model_acc, human_acc, kappa))
                # import pdb;pdb.set_trace()
                # plt.subplot(121);plt.plot(it_model_res, label="model");plt.subplot(122);plt.plot(human_res, label="human");plt.show()
            model_ks.append(kappa)
        ks.append(np.mean(model_ks))
    return np.asarray(ks)


def compute_sr(melt_model_df, sub_res, sub_files):
    ks = []
    models = np.unique(melt_model_df.model_name)
    subs = np.unique([k for k in sub_res.keys()])
    for m in models:
        model_df = melt_model_df[melt_model_df.model_name == m]
        model_files = model_df.img_name.values
        model_res = model_df.score.values
        model_ks = []
        participant_res = []
        for s in subs:
            sub_file = sub_files[s]
            sub_file = np.asarray([os.path.sep.join(x.split(os.path.sep)[1:]) for x in sub_file.tolist()])
            sub_mask = np.in1d(sub_file, model_files)
            trim_sub_file = sub_file[sub_mask]
            model_reorder = align_idx(
                model_files.tolist(),
                trim_sub_file.tolist())
            assert (trim_sub_file == model_files[model_reorder]).mean() == 1, "Mismatch between model/human order found."
            human_res = sub_res[s][sub_mask]
            it_model_res = model_res[model_reorder]
            model_ks.append(spearmanr(human_res, it_model_res)[0])
        ks.append(np.mean(model_ks))
    return np.asarray(ks)

def human_kappas(res, labs, fs, iterations=20):
    """Computer real ceiling and randomized floor."""
    #### Get human ceil
    kappa_ceils = []
    for k, v in res.items():
        k_file = fs[k]
        k_labs = labs[k]
        k_idx = np.unique(k_file, return_index=True)[1]
        # k_file = k_file[k_idx]
        v = v[k_idx].values
        l = k_labs[k_idx].values
        f = k_file[k_idx]
        ma = (v == l).mean()
        kappa_ceil = []
        for i, j in res.items():
            if k != i:
                i_file = fs[i]
                i_idx = np.unique(i_file, return_index=True)[1]
                assert np.all(f == i_file[i_idx]), "Mismatch in labels found"
                fixed_j = j[i_idx].values
                # kappa_ceil.append(cks(v, j[i_idx].values))
                mb = (fixed_j == l).mean()
                kappa_ceil.append(error_consistency(v, fixed_j, l))
        # kappa_ceils.append(np.mean(kappa_ceil))
        kappa_ceils.append(np.median(kappa_ceil))

    #### Get human floor
    sim_floor = []
    for i in tqdm(range(iterations), total=iterations, desc="Randomizing floor"):
        kappa_floor = []
        for k, v in res.items():
            k_file = fs[k]
            k_labs = labs[k]
            k_idx = np.unique(k_file, return_index=True)[1]
            # k_file = k_file[k_idx]
            v = v[k_idx].values
            l = k_labs[k_idx].values
            for i, j in res.items():
                if k != i:
                    i_file = fs[i]
                    i_idx = np.unique(i_file, return_index=True)[1]
                    ridx = np.random.permutation(len(v))
                    iv = v[ridx]  # Shuffle
                    il = l[ridx]  # Shuffle
                    # kappa_floor.append(cks(iv, j[i_idx].values))
                    kappa_floor.append(error_consistency(iv, j[i_idx].values, il))
        # sim_floor.append(np.mean(kappa_floor))
        sim_floor.append(np.median(kappa_floor))
    return kappa_ceil, sim_floor


def human_spearmans(res, fs, iterations=20):
    """Computer real ceiling and randomized floor."""
    #### Get human ceil
    kappa_ceil = []
    for k, v in res.items():
        k_file = fs[k]
        k_idx = np.unique(k_file, return_index=True)[1]
        # k_file = k_file[k_idx]
        v = v[k_idx].values
        for i, j in res.items():
            if k != i:
                i_file = fs[i]
                i_idx = np.unique(i_file, return_index=True)[1]
                kappa_ceil.append(cks(v, j[i_idx].values))

    #### Get human floor
    sim_floor = []
    for i in tqdm(range(iterations), total=iterations, desc="Randomizing floor"):
        kappa_floor = []
        for k, v in res.items():
            k_file = fs[k]
            k_idx = np.unique(k_file, return_index=True)[1]
            # k_file = k_file[k_idx]
            v = v[k_idx].values
            for i, j in res.items():
                if k != i:
                    i_file = fs[i]
                    i_idx = np.unique(i_file, return_index=True)[1]
                    iv = v[np.random.permutation(len(v))]
                    kappa_floor.append(cks(iv, j[i_idx].values))
        sim_floor.append(np.mean(kappa_floor))
    return kappa_ceil, sim_floor


def spearman_alignment(model_df, human_res, human_files):
    """Compute spearman between average human repsonse and model score."""
    models = np.unique(model_df.columns[2:])

    # Get a dict of responses per image
    all_files = np.unique(np.concatenate([x for x in human_files.values()]))
    proc_human_res = {k: [] for k in all_files}
    for k, res in human_res.items():
        files = human_files[k]
        for f, r in zip(files, res):
            proc_human_res[f] += [r]

    # Compute average response per image
    avg_human_res = {k: np.mean(v) for k, v in proc_human_res.items()}

    # Trim down to overlap
    avg_human_files = np.asarray([os.path.sep.join(x.split(os.path.sep)[1:]) for x in avg_human_res.keys()])
    model_files = model_df.img_name.tolist()
    mask = np.in1d(avg_human_files, model_files)
    if not np.all(mask):
        keep_keys = avg_human_files[mask]
        rev_avg_human_res = {}
        for k, v in avg_human_res.items():
            if os.path.sep.join(k.split(os.path.sep)[1:]) in keep_keys:
                rev_avg_human_res[k] = v
        avg_human_res = rev_avg_human_res

    # Now compute correlation between each model and the average human
    human_files = [os.path.sep.join(x.split(os.path.sep)[1:]) for x in avg_human_res.keys()]
    human_scores = np.asarray([x for x in avg_human_res.values()])
    model_reorder = align_idx(model_files, human_files)
    reorder_model_df = model_df.iloc[model_reorder]
    assert np.all(np.asarray(reorder_model_df.img_name.tolist()) == np.asarray(human_files)), "File mismatch."
    rhos = []
    for m in models:
        model_scores = reorder_model_df[m].values  # Reordered to match humans
        rhos.append(spearmanr(model_scores, human_scores)[0])
    return rhos


def human_spearmans(human_res, human_files, iterations=1000, shuffle=False):
    """Compute spearman between average human repsonse and model score."""

    if shuffle:
        version = "floor"
    else:
        version = "ceil"

    def get_avg(hres, hfiles):
        all_files = np.unique(np.concatenate([x for x in hfiles.values()]))
        proc_human_res = {k: [] for k in all_files}
        for k, res in hres.items():
            files = hfiles[k]
            for f, r in zip(files, res):
                proc_human_res[f] += [r]
        proc_human_res = {k: np.mean(v) for k, v in proc_human_res.items()}
        return np.asarray([v for v in proc_human_res.values()])
    
    # Take iterations splits and compute correlations between each
    subjects = np.asarray([x for x in human_files.keys()])
    n = len(subjects)
    rhos = []
    for i in tqdm(range(iterations), desc="Simulating human {}".format(version), total=iterations):
        rand_subjects = subjects[np.random.permutation(n)]
        fh = rand_subjects[:n//2]
        sh = rand_subjects[n//2:]
        
        fh_res = {k: v for k, v in human_res.items() if k in fh}
        sh_res = {k: v for k, v in human_res.items() if k in sh}
        fh_files = {k: v for k, v in human_files.items() if k in fh}
        sh_files = {k: v for k, v in human_files.items() if k in sh}

        fh_avg = get_avg(fh_res, fh_files)
        sh_avg = get_avg(sh_res, sh_files)
        if shuffle:
            fh_avg = fh_avg[np.random.permutation(len(fh_avg))]
        rhos.append(spearmanr(fh_avg, sh_avg)[0])
    return np.asarray(rhos)
