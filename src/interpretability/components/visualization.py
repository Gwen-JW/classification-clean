import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


method_visualization_dict = {
    "integrated_gradients": {
        "method_name": "Integrated Gradients ($IG$)",
        "color": "#8b66b8", # purple
    },
    "deeplift": {
        "method_name": "DeepLIFT ($DL$)",
        "color": "#64a138", # green
    },
    "deepliftshap": {
        "method_name": "DeepSHAP ($DS$)",
        "color": "#e7a4c5", # pink
    },
    "gradshap": {
        "method_name": "GradientSHAP ($GS$)",
        "color": "#1571b8", # blue
    },
    "kernelshap": {
        "method_name": "KernelSHAP ($KS$)",
        "color": "#aa732c", # brown
    },
    "shapleyvalue": {
        "method_name": "Shapley Value Sampling ($SVS$)",
        "color": "#fc8002", # orange
    },
    "random": {
        "method_name": "Random",
        "color": "#838383", # grey
    },
}


def ridge_line_vis(save_results, methods):
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 4, figure=fig)
    path_corrupted = glob.glob(os.path.join(save_results, "corrupted_k", "*"))
    percents = sorted(list(set([os.path.split(path)[-1].split("__")[1] for path in path_corrupted])), reverse=True)
    for n, method in enumerate(methods):
        if n == 0:
            ax = fig.add_subplot(gs[n//3, n%3])
        else:
            ax = fig.add_subplot(gs[n//3, n%3], sharey=ax)
        for i, percent in enumerate(percents):
            df_normalized_score_drop = pd.read_csv(os.path.join(save_results, f"corrupted_k/{method}__{percent}/score_drop__top.csv"), index_col=0)
            normalized_score_drop = df_normalized_score_drop["normalized_score_drop"].values
            kde = gaussian_kde(normalized_score_drop)
            x = np.linspace(min(normalized_score_drop), max(normalized_score_drop), 1000)
            ax.plot(x, kde.pdf(x) + i, color="k", zorder=i, linewidth=1)
            ax.fill_between(x, kde.pdf(x) + i-0.02, i, color="#A7C0DE", zorder=-i-10, alpha=0.5)

        ax.yaxis.set_tick_params(tick1On=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 15)
        ax.axvline(0.0, ls="--", lw=0.75, color="black", ymax=0.8)
        ax.set_xlabel(r"$\tilde{\mathcal{S}}(\overline{\mathbf{X}})$", fontsize=11)
        title_name = method_visualization_dict[method]["method_name"]
        ax.set_title(f"{title_name}", ha='center', x=0.5, y=-0.25)

        if n%3 == 0:
            ax.yaxis.set_tick_params(labelleft=True)
            ax.set_yticks(np.arange(11))
            ax.set_yticklabels([percents[i] for i in range(0, 11)])
            ax.set_ylabel(r"$k$", ha='center', x=0.5, y=0.34, fontsize=11)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(10)
                tick.label1.set_verticalalignment("center")
        else:
            ax.yaxis.set_tick_params(labelleft=False)

    ax_random = fig.add_subplot(gs[1, 3], sharey=ax)
    for i, percent in enumerate(percents):
        random_percent = pd.DataFrame()
        for n, method in enumerate(methods):
            df_normalized_score_drop = pd.read_csv(os.path.join(save_results, f"corrupted_k/{method}__{percent}/score_drop__top.csv"), index_col=0)
            random_percent[method] = df_normalized_score_drop["normalized_score_drop_random"]
        normalized_score_drop_random = random_percent.mean(axis=1).values
        kde = gaussian_kde(normalized_score_drop_random)
        x = np.linspace(min(normalized_score_drop_random), max(normalized_score_drop_random), 1000)
        ax_random.plot(x, kde.pdf(x) + i, color="k", zorder=i, linewidth=1)
        ax_random.fill_between(x, kde.pdf(x) + i-0.02, i, color="#838383", zorder=-i-10, alpha=0.5)

        ax_random.get_yaxis().set_visible(False)
        ax_random.spines['top'].set_visible(False)
        ax_random.spines['right'].set_visible(False)
        ax_random.spines['bottom'].set_visible(False)
        ax_random.spines['left'].set_visible(False)
        ax_random.set_xlim(-1, 1)
        ax_random.axvline(0.0, ls="--", lw=0.75, color="black", ymax=0.8)
        ax_random.set_xlabel(r"$\tilde{\mathcal{S}}(\overline{\mathbf{X}})$", fontsize=11)
        ax_random.set_title("Random", ha='center', x=0.5, y=-0.25)

    plt.tight_layout(w_pad=1.5, h_pad=1)

    visualization_path = os.path.join(save_results, "visualization")
    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)
    plt.savefig(os.path.join(visualization_path, "ridge_line_plot.pdf"))


def mean_vis(save_results, methods):
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(2, 4, figure=fig)

    coarse_metrics = pd.read_csv(os.path.join(save_results, "metrics_methods.csv"), index_col=0)
    AUCSm_best_method = coarse_metrics["AUCSm_top"].idxmax()
    F1Sm_best_method = coarse_metrics["F1Sm"].idxmax()

    for n, method in enumerate(methods):
        if n == 0:
            ax = fig.add_subplot(gs[n//3, n%3])
        else:
            ax = fig.add_subplot(gs[n//3, n%3], sharey=ax)

        df_top_tmp = pd.read_csv(os.path.join(save_results, f"summary_robust_{method}__top.csv"), index_col=0)
        df_bottom_tmp = pd.read_csv(os.path.join(save_results, f"summary_robust_{method}__bottom.csv"), index_col=0)

        ax.plot(
            np.insert(df_top_tmp["mean_fraction_corrupted"].values, 0, 0), 
            np.insert(df_top_tmp["mean_normalized_score_drop"].values, 0, 0),
            "X-",
            markersize=7,
            color=method_visualization_dict[method]["color"],
        )

        ax.plot(
            np.insert(df_bottom_tmp["mean_fraction_corrupted"].values, 0, 0),
            np.insert(df_bottom_tmp["mean_normalized_score_drop"].values, 0, 0),
            "o-",
            markersize=7,
            color=method_visualization_dict[method]["color"],
        )
        ax.fill_between(
            np.insert(df_bottom_tmp["mean_fraction_corrupted"].values, 0, 0),
            np.insert(df_top_tmp["mean_normalized_score_drop"].values, 0, 0),
            np.insert(df_bottom_tmp["mean_normalized_score_drop"].values, 0, 0),
            color=method_visualization_dict[method]["color"],
            alpha=0.4,
        )

        ax.grid(lw=0.5, alpha=0.5, ls='dashed')
        ax.set_xlim([0, 1.00])
        ax.set_ylim([0, 1.00])
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.set_xlabel(r"$\tilde{N}$", fontsize=11)

        title_name = method_visualization_dict[method]["method_name"]
        ax.set_title(f"{title_name}", ha='center')

        if n%3 == 0:
            # ax.yaxis.set_tick_params(labelleft=True)
            # ax.set_yticks(np.arange(11))
            ax.set_ylabel(r"$\tilde{\mathcal{S}}_{\mathrm{m}}$", fontsize=11)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(10)
                tick.label1.set_verticalalignment("center")
        else:
            ax.yaxis.set_tick_params(labelleft=False)

        if method == AUCSm_best_method:
            font_weight_AUCSm = "bold"
            font_color_AUCSm = "g"
        else:
            font_weight_AUCSm = "normal"
            font_color_AUCSm = "k"
        
        if method == F1Sm_best_method:
            font_weight_F1Sm = "bold"
            font_color_F1Sm = "g"
        else:
            font_weight_F1Sm = "normal"
            font_color_F1Sm = "k"

        
        AUCSm_top_value = coarse_metrics.loc[method, "AUCSm_top"]
        F1Sm_value = coarse_metrics.loc[method, "F1Sm"]
        
        ax.text(
            0.97, 0.18,
            r"$\operatorname{AUC}\tilde{\mathcal{S}}^{\mathrm{top}}_{\mathrm{m}}$" + f" = {AUCSm_top_value :.2f}",
            ha='right',
            va='bottom',
            size=11,
            weight=font_weight_AUCSm,
            color=font_color_AUCSm,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=font_color_AUCSm, lw=2),
            transform=ax.transAxes,
        )

        ax.text(
            0.97, 0.05,
            r"$\operatorname{F}1\tilde{\mathcal{S}}_{\mathrm{m}}$" + f" = {F1Sm_value :.2f}",
            ha='right',
            va='bottom',
            size=11,
            weight=font_weight_F1Sm,
            color=font_color_F1Sm,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=font_color_F1Sm, lw=2),
            transform=ax.transAxes,
        )
            

    ax_random = fig.add_subplot(gs[1, 3], sharey=ax)

    df_random_top = pd.DataFrame()
    df_random_bottom = pd.DataFrame()
    df_random_top_fraction = pd.DataFrame()
    df_random_bottom_fraction = pd.DataFrame()
    for n, method in enumerate(methods):
        df_top_tmp = pd.read_csv(os.path.join(save_results, f"summary_robust_{method}__top.csv"), index_col=0)
        df_bottom_tmp = pd.read_csv(os.path.join(save_results, f"summary_robust_{method}__bottom.csv"), index_col=0)

        df_random_top[method] = df_top_tmp["mean_normalized_score_drop_random"]
        df_random_top_fraction[method] = df_top_tmp["mean_fraction_corrupted"]
        df_random_bottom[method] = df_bottom_tmp["mean_normalized_score_drop_random"]
        df_random_bottom_fraction[method] = df_bottom_tmp["mean_fraction_corrupted"]

    ax_random.plot(
        np.insert(df_random_top_fraction.mean(axis=1).values, 0, 0),
        np.insert(df_random_top.mean(axis=1).values, 0, 0),
        "X-",
        markersize=7,
        color=method_visualization_dict["random"]["color"],
    )

    ax_random.plot(
        np.insert(df_random_bottom_fraction.mean(axis=1).values, 0, 0),
        np.insert(df_random_bottom.mean(axis=1).values, 0, 0),
        "o-",
        markersize=7,
        color=method_visualization_dict["random"]["color"],
    )

    ax_random.fill_between(
        np.insert(df_random_bottom_fraction.mean(axis=1).values, 0, 0),
        np.insert(df_random_top.mean(axis=1).values, 0, 0),
        np.insert(df_random_bottom.mean(axis=1).values, 0, 0),
        color=method_visualization_dict["random"]["color"],
        alpha=0.4,
    )

    ax_random.grid(lw=0.5, alpha=0.5, ls='dashed')
    ax_random.set_xlim([0, 1.00])
    ax_random.set_xticks(np.arange(0, 1.1, 0.1))
    ax_random.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax_random.yaxis.set_tick_params(labelleft=False)
    ax_random.set_xlabel(r"$\tilde{N}$", fontsize=11)
    ax_random.set_title("Random", ha='center')

    # AUCSm_top_random = coarse_metrics["AUCSm_top_random"].mean()
    # F1Sm_random = coarse_metrics["F1Sm_random"].mean()
    AUCSm_top_random = coarse_metrics.loc["random", "AUCSm_top"]
    F1Sm_random = coarse_metrics.loc["random", "F1Sm"]

    ax_random.text(
        0.97, 0.18,
        r"$\operatorname{AUC}\tilde{\mathcal{S}}^{\mathrm{top}}_{\mathrm{m}}$" + f" = {AUCSm_top_random :.2f}",
        ha='right',
        va='bottom',
        size=11,
        weight="normal",
        color="k",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="k", lw=2),
        transform=ax_random.transAxes,
    )

    ax_random.text(
        0.97, 0.05,
        r"$\operatorname{F}1\tilde{\mathcal{S}}_{\mathrm{m}}$" + f" = {F1Sm_random :.2f}",
        ha='right',
        va='bottom',
        size=11,
        weight="normal",
        color="k",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="k", lw=2),
        transform=ax_random.transAxes,
    )

    plt.tight_layout(w_pad=1.5, h_pad=1)

    visualization_path = os.path.join(save_results, "visualization")
    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)
    plt.savefig(os.path.join(visualization_path, "mean_plot.pdf"))


def skew_kurt_vis(save_results, corruption_percents, methods):
    skew = pd.DataFrame(index=corruption_percents, columns=methods)
    kurt = pd.DataFrame(index=corruption_percents, columns=methods)
    skew_random = pd.DataFrame(index=corruption_percents, columns=methods)
    kurt_random = pd.DataFrame(index=corruption_percents, columns=methods)
    for method in methods:
        df = pd.read_csv(os.path.join(save_results, f"summary_robust_{method}__top.csv"), index_col=0)
        skew[method] = df["skewness_scaled"]
        skew_random[method] = df["skewness_scaled_random"]
        kurt[method] = df["kurtosis_scaled"]
        kurt_random[method] = df["kurtosis_scaled_random"]

    skew["random"] = skew_random.mean(axis=1)
    kurt["random"] = kurt_random.mean(axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(7, 4))
    methods_with_random = methods + ["random"]
    for method in methods_with_random:
        if method == "random":
            method_label = "Random"
        else:
            method_label = method_visualization_dict[method]["method_name"].split("(")[1].split(")")[0]
        ax[0].plot(
            skew[method], 
            "o--",
            label=method_label, 
            markersize=4, 
            color=method_visualization_dict[method]["color"]
        )
        ax[0].grid(lw=0.5, alpha=0.5, ls='dashed')
        ax[0].legend(ncol=1, fontsize=9, loc='lower left')
        ax[0].set_xticks([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0])
        ax[0].set_xticklabels(rotation=30, labels=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, ''], fontsize=9)
        
        ax[0].set_yticks([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])
        ax[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        ax[0].set_xlabel(r"$k$", fontsize=9)
        ax[0].set_ylabel("Scaled skewness", labelpad=0)
        ax[0].set_title("$Skew-k$")

        ax[1].plot(
            kurt[method],
            "^--",
            label=method_label, 
            markersize=6, 
            color=method_visualization_dict[method]["color"]
        )
        ax[1].grid(lw=0.5, alpha=0.5, ls='dashed')
        ax[1].legend(ncol=1, fontsize=9, loc='upper left')
        ax[1].set_xticks([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0])
        ax[1].set_xticklabels(rotation=30, labels=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, ''], fontsize=9)

        ax[1].set_yticks([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])
        ax[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
        ax[1].set_xlabel(r"$k$", fontsize=9)
        ax[1].set_ylabel("Scaled kurtosis", labelpad=0)
        ax[1].set_title("$(E)Kurt-k$")
        
    plt.tight_layout(w_pad=1)

    visualization_path = os.path.join(save_results, "visualization")
    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)
    plt.savefig(os.path.join(visualization_path, "skew_kurt_k.pdf"))


def frac_vis(save_results, corruption_percents, methods):
    frac_neg = pd.DataFrame(index=corruption_percents, columns=methods)
    frac_05 = pd.DataFrame(index=corruption_percents, columns=methods)
    frac_10 = pd.DataFrame(index=corruption_percents, columns=methods)
    frac_neg_random = pd.DataFrame(index=corruption_percents, columns=methods)
    frac_05_random = pd.DataFrame(index=corruption_percents, columns=methods)
    frac_10_random = pd.DataFrame(index=corruption_percents, columns=methods)
    for method in methods:
        df = pd.read_csv(os.path.join(save_results, f"summary_robust_{method}__top.csv"), index_col=0)
        frac_neg[method] = df["frac_neg"]
        frac_neg_random[method] = df["frac_neg_random"]
        frac_05[method] = df["frac_05"]
        frac_05_random[method] = df["frac_05_random"]
        frac_10[method] = df["frac_10"]
        frac_10_random[method] = df["frac_10_random"]

    frac_neg["random"] = frac_neg_random.mean(axis=1)
    frac_05["random"] = frac_05_random.mean(axis=1)
    frac_10["random"] = frac_10_random.mean(axis=1)

    fig, ax = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    methods_with_random = methods + ["random"]
    for method in methods_with_random:
        if method == "random":
            method_label = "Random"
        else:
            method_label = method_visualization_dict[method]["method_name"].split("(")[1].split(")")[0]
        ax[0].plot(
            frac_neg[method],
            marker="x", 
            linestyle="--",
            label=method_label, 
            markersize=6, 
            color=method_visualization_dict[method]["color"]
        )

        ax[0].grid(lw=0.5, alpha=0.5, ls='dashed')
        ax[0].legend(ncol=1, fontsize=9, loc='upper left')
        ax[0].set_xticks([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0])
        ax[0].set_xticklabels(rotation=30, labels=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, ''], fontsize=9)
        
        ax[0].set_yticks([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])
        ax[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        ax[0].set_xlabel("$k$", fontsize=9)
        ax[0].set_ylabel("Fraction")
        ax[0].set_title(r"$\tilde{\mathcal{S}} \leq 0.0$")

        ax[1].plot(
            frac_05[method],
            marker="s",
            linestyle="--",
            label=method_label, 
            markersize=5, 
            color=method_visualization_dict[method]["color"],
        )

        ax[1].grid(lw=0.5, alpha=0.5, ls='dashed')
        ax[1].legend(ncol=1, fontsize=9, loc='upper left')
        ax[1].set_xticks([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0])
        ax[1].set_xticklabels(rotation=30, labels=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, ''], fontsize=9)

        # ax[1].set_yticks([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])
        # ax[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
        ax[1].set_xlabel("$k$", fontsize=9)
        ax[1].set_title(r"$0.0 < \tilde{\mathcal{S}} \leq 0.5$")

        ax[2].plot(
            frac_10[method],
            marker="*",
            linestyle="--",
            label=method_label, 
            markersize=8, 
            color=method_visualization_dict[method]["color"]
        )

        ax[2].grid(lw=0.5, alpha=0.5, ls='dashed')
        ax[2].legend(ncol=1, fontsize=9, loc="upper left")
        ax[2].set_xticks([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0])
        ax[2].set_xticklabels(rotation=30, labels=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, ''], fontsize=9)

        # ax[2].set_yticks([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])
        # ax[2].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        ax[2].set_xlabel("$k$", fontsize=9)
        ax[2].set_title(r"$0.5 < \tilde{\mathcal{S}} \leq 1.0$")
        
    plt.tight_layout(w_pad=1)

    visualization_path = os.path.join(save_results, "visualization")
    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)
    plt.savefig(os.path.join(visualization_path, "frac_k.pdf"))
    