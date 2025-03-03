import glob
import os
import re

import numpy as np
import pandas as pd

from scipy import stats
import scipy.integrate as integrate

import torch
from torch import nn
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from src.interpretability.components.interpret_args import dict_method_arguments

global device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class Robustness:
    def __init__(self, 
        model_interp: nn.Module,
        signals: np.array,
        targets: np.array,
        sample_ids: np.array,
        log_dir: str,
    ):
        self.model_interp = model_interp.to(device)
        self.model_interp = torch.nn.Sequential(self.model_interp, torch.nn.Softmax(dim=-1)).eval()
        self.signals = signals # dim of signal: (num_instances, num_features, num_timesteps)
        self.targets = targets
        self.sample_ids = sample_ids
        self.save_results = os.path.join(log_dir, "interpretability_evaluation")

        if not os.path.exists(self.save_results):
            os.makedirs(self.save_results)
        
        self.original_scores, self.predicted_indices = self.get_original_scores()   


    def get_original_scores(self):
        signals = torch.tensor(self.signals).to(device)
        batch_size = 64
        batches = signals.split(batch_size)
        original_scores = torch.tensor([], device=device, requires_grad=False)
        for batch in batches:
            with torch.no_grad():
                original_scores = torch.cat((original_scores, self.model_interp(batch)), dim=0)
        
        predicted_indices = torch.argmax(original_scores, dim=1)

        original_scores = original_scores.detach().cpu().numpy()
        predicted_indices = predicted_indices.detach().cpu().numpy()
        return original_scores, predicted_indices
    

    def get_relevance(self, method_name):
        
        # TODO (maybe) set up device for LSTM model 
        lstm_network_bool = any(
            [isinstance(module, torch.nn.modules.rnn.LSTM) for module in self.model_interp.modules()]
        )
        if lstm_network_bool:
            torch.backends.cudnn.enabled = False

        print(f"Calculating relevances for {method_name}...")

        signals = torch.tensor(self.signals, device=device)
        pred_class_idx = torch.tensor(self.predicted_indices, device=device)
        relevances = torch.tensor([], device=device, requires_grad=False)

        # get post-hoc interpretability method arguments
        rel_method = dict_method_arguments[method_name]["captum_method"]
        kwargs_method = dict_method_arguments[method_name].get("kwargs_method", {})
        kwargs_attribution = dict_method_arguments[method_name].get("kwargs_attribution", {})
        rel_method = rel_method(self.model_interp, **(kwargs_method))

        if dict_method_arguments[method_name]["require_baseline"]:
            baseline_type = dict_method_arguments[method_name]["baseline_type"]

            num_samples = min(signals.shape[0], 50)
            if baseline_type == "random":
                # return baseline as random values
                baseline = torch.normal(mean=0, std=1, size=signals[:1].shape, device=device, dtype=torch.float32)
            elif baseline_type == "sample":
                if "sample_baseline.npy" in os.listdir(self.save_results):
                    print("Loading baseline from file...")
                    baseline = np.load(os.path.join(self.save_results, "sample_baseline.npy"))
                    baseline = torch.tensor(baseline, device=device)
                else:
                    print("Extracting baseline samples...")
                    # return baseline as sample of given size of signal
                    indices = torch.randperm(signals.shape[0])[:num_samples]
                    baseline = signals[indices]
                    np.save(os.path.join(self.save_results, "sample_baseline.npy"), baseline.detach().cpu().numpy())

            kwargs_attribution["baselines"] = baseline.type(torch.float32).to(device)

        # calculate relevances
        batch_size = dict_method_arguments[method_name].get("batch_size")
        batched_signals = signals.split(batch_size)
        batched_pred_indices = pred_class_idx.split(batch_size)
        for (signal, target) in tqdm(zip(batched_signals, batched_pred_indices), total=len(batched_signals)):
            with torch.no_grad():
                relevance = rel_method.attribute(signal, target=target, **(kwargs_attribution))
                relevances = torch.cat((relevances, relevance), dim=0)

        path_relevances = os.path.join(self.save_results, "relevances")
        if not os.path.exists(path_relevances):
            os.makedirs(path_relevances)
        # save the relevance as .np files
        np.save(os.path.join(path_relevances, f"{method_name}_relevances.npy"), relevances.detach().cpu().numpy())

        if lstm_network_bool:
            torch.backends.cudnn.enabled = True

        return relevances.detach().cpu().numpy()


    def corrupt_per_sample(self, signal, relevance, k, topk):
        corrupted_signal = signal.copy()
        randomly_corrupted_signal = signal.copy()

        pos_rel = relevance[relevance >= 0]
        num_corrupt = int(k * pos_rel.shape[0] - 1)
        if topk:
            pos_rel_soted = np.sort(pos_rel)[::-1]
            threshold = pos_rel_soted[num_corrupt]
            mask = (relevance >= 0) & (relevance >= threshold)
            corrupted_signal[mask] = np.random.normal(scale=1, size=mask.sum())
        else:
            pos_rel_soted = np.sort(pos_rel)
            threshold = pos_rel_soted[num_corrupt]
            mask = (relevance >= 0) & (relevance <= threshold)
            corrupted_signal[mask] = np.random.normal(scale=1, size=mask.sum())

        # num_corrupt = int(k * signal.size - 1)
        # if topk:
        #     rel_sorted = np.sort(relevance.flat)[::-1]
        #     threshold = rel_sorted[num_corrupt]
        #     mask = (relevance >= threshold)
        #     corrupted_signal[mask] = np.random.normal(scale=1, size=mask.sum())
        # else:
        #     rel_sorted = np.sort(relevance.flat)
        #     threshold = rel_sorted[num_corrupt]
        #     mask = (relevance <= threshold)
        #     corrupted_signal[mask] = np.random.normal(scale=1, size=mask.sum())

        random_corrupted_indices = np.random.choice(signal.size, mask.sum(), replace=False)
        randomly_corrupted_signal.flat[random_corrupted_indices] = np.random.normal(scale=1, size=mask.sum())
        randomly_corrupted_signal = randomly_corrupted_signal.reshape(signal.shape)

        dict_results = {
            "num_corrupted": num_corrupt,
            "fraction_corrupted": num_corrupt / signal.size,
            "corrupted_signal": corrupted_signal,
            "randomly_corrupted_signal": randomly_corrupted_signal,
        }
        return dict_results


    def compute_score_drop(self, corrupted_signals, randomly_corrupted_signals):
        self.model_interp.to(device)
        required_columns = [
            "score1",
            "score2",
            "score3",
            "delta_score1",
            "delta_score2",
            "original_acc",
            "corrupted_acc",
            "randomly_corrupted_acc",
            "normalized_score_drop",
            "normalized_score_drop_random",
        ]
        df_score = pd.DataFrame(index=self.sample_ids, columns=required_columns)
        
        correct_classification = (self.predicted_indices == self.targets)
        df_score.loc[:, "score1"] = self.original_scores[np.arange(self.original_scores.shape[0]), self.predicted_indices]
        df_score.loc[:, "original_acc"] = correct_classification
        
        batch_size = 64
        corrupted_signals = torch.tensor(corrupted_signals,dtype=torch.float32, device=device)
        corrupted_batches = corrupted_signals.split(batch_size)
        corrupted_scores = torch.tensor([], device=device, requires_grad=False)
        for batch in corrupted_batches:
            with torch.no_grad():
                corrupted_scores = torch.cat((corrupted_scores, self.model_interp(batch)), dim=0)
        correct_classification = (corrupted_scores.argmax(dim=1).detach().cpu().numpy() == self.targets)
        df_score.loc[:, "score2"] = corrupted_scores[torch.arange(corrupted_scores.shape[0]), self.predicted_indices].detach().cpu().numpy()
        df_score.loc[:, "corrupted_acc"] = correct_classification

        randomly_corrupted_signals = torch.tensor(randomly_corrupted_signals,dtype=torch.float32, device=device)
        randomly_corrupted_batches = randomly_corrupted_signals.split(batch_size)
        randomly_corrupted_scores = torch.tensor([], device=device, requires_grad=False)
        for batch in randomly_corrupted_batches:
            with torch.no_grad():
                randomly_corrupted_scores = torch.cat((randomly_corrupted_scores, self.model_interp(batch)), dim=0)
        correct_classification = (randomly_corrupted_scores.argmax(dim=1).detach().cpu().numpy() == self.targets)
        df_score.loc[:, "score3"] = randomly_corrupted_scores[torch.arange(randomly_corrupted_scores.shape[0]), self.predicted_indices].detach().cpu().numpy()
        df_score.loc[:, "randomly_corrupted_acc"] = correct_classification

        # compute normalized score drop
        df_score.loc[:, "delta_score1"] = df_score.loc[:, "score1"] - df_score.loc[:, "score2"]
        df_score.loc[:, "delta_score2"] = df_score.loc[:, "score1"] - df_score.loc[:, "score3"]

        normalized_score_drop = df_score.loc[:, "delta_score1"] / df_score.loc[:, "score1"]
        normalized_score_drop_random = (df_score.loc[:, "delta_score2"] / df_score.loc[:, "score1"])

        df_score.loc[:, "normalized_score_drop"] = normalized_score_drop
        df_score.loc[:, "normalized_score_drop_random"] = normalized_score_drop_random

        return df_score


    def get_corrupted_scores(self, corruption_percents, method_name):
        print(f"Calculating corrupted scores for {method_name}...")
        for topk in [True, False]:
            corruption_strategy = "top" if topk else "bottom"
            for k in corruption_percents:
                path_save = os.path.join(self.save_results, "corrupted_k", f"{method_name}__{k:.2f}")

                if not os.path.exists(path_save):
                    os.makedirs(path_save)

                corruption_results = pd.DataFrame()
                relevances = np.load(os.path.join(self.save_results, "relevances", f"{method_name}_relevances.npy"))
                corrupted_signals = np.empty(self.signals.shape)
                randomly_corrupted_signals = np.empty(self.signals.shape)

                # Iterate across all samples to compute metrics
                for idx in tqdm(range(self.signals.shape[0])):
                    sample_id = self.sample_ids[idx]
                    signal = self.signals[idx].copy()
                    relevance = relevances[idx]

                    corrupted_results = self.corrupt_per_sample(signal=signal, relevance=relevance, k=k, topk=topk)

                    corruption_summary = {
                        "predicted_class": self.predicted_indices[idx],
                        "min_Relevance": np.min(relevance),
                        "max_Relevance": np.max(relevance),
                        "mean_Relevance": np.mean(relevance),
                        "median_Relevance": np.median(relevance),
                        "num_corrupted": corrupted_results["num_corrupted"],
                        "fraction_corrupted": corrupted_results["fraction_corrupted"],
                    }

                    df_tmp = pd.DataFrame(corruption_summary, index=[sample_id])
                    corrupted_signals[idx] = corrupted_results["corrupted_signal"]
                    randomly_corrupted_signals[idx] = corrupted_results["randomly_corrupted_signal"]
                    # concat corrupted results of each sample
                    corruption_results = pd.concat([corruption_results, df_tmp], axis=0)

                # # if save corrupted signals and randomly corrupted signals
                # np.save(os.path.join(self.save_results, "corrupted_signals.npy"), corrupted_signals)
                # np.save(os.path.join(self.save_results, "randomly_corrupted_signals.npy"), randomly_corrupted_signals)

                # Compute model prediction on initial, corrupted and randomly corrupted signals to get normalized score drop
                df_score = self.compute_score_drop(corrupted_signals, randomly_corrupted_signals)
                if corruption_results.shape[0] != df_score.shape[0]:
                    raise ValueError("Size not matching")
                corruption_results = pd.concat([corruption_results, df_score], axis=1)
                # save results to csv
                corruption_results.to_csv(os.path.join(path_save, f"score_drop__{corruption_strategy}.csv"))


    def summary_robust(self, method_name):
        # Extract csv with results for all k percentages
        path_files = glob.glob(os.path.join(self.save_results, "corrupted_k", f"{method_name}__*", "score_drop__*.csv"))
        required_columns = [
            "fraction_corrupted",
            "normalized_score_drop",
            "normalized_score_drop_random",
            "original_acc",
            "corrupted_acc",
            "randomly_corrupted_acc",
        ]
        corruption_percents = [float(re.search("[\d.]+$", tmp.split(os.sep)[-2]).group()) for tmp in path_files]
        corruption_percents = list(set(corruption_percents))

        robust_top = pd.DataFrame(index=corruption_percents, columns=required_columns)
        robust_bottom = pd.DataFrame(index=corruption_percents, columns=required_columns)

        for path in path_files:
            k = float(re.search("[\d.]+$", path.split(os.sep)[-2]).group())
            corruption_strategy = re.search("(top|bottom)", path.split(os.sep)[-1]).group()
            df_tmp = pd.read_csv(path, index_col=0)[required_columns]
            df_tmp.replace([np.inf, -np.inf], np.nan)
            df_tmp = df_tmp.dropna(axis=1, how="all")

            df_nsd = df_tmp.iloc[:, :3]
            df_acc = df_tmp.iloc[:, 3:]
            if corruption_strategy == "top":
                robust_top.loc[k, df_acc.columns] = (np.sum(df_acc, axis=0) / df_acc.shape[0])
                robust_top.loc[k, df_nsd.columns] = np.nanmean(df_nsd.values, axis=0)
                robust_top.loc[k, "skewness"] = stats.skew(df_nsd["normalized_score_drop"].values, bias=True)
                robust_top.loc[k, "kurtosis"] = stats.kurtosis(df_nsd["normalized_score_drop"].values, bias=True)
                robust_top.loc[k, "skewness_random"] = stats.skew(df_nsd["normalized_score_drop_random"].values, bias=True)
                robust_top.loc[k, "kurtosis_random"] = stats.kurtosis(df_nsd["normalized_score_drop_random"].values, bias=True)

                robust_top.loc[k, "frac_neg"] = (df_nsd["normalized_score_drop"] <= 0).sum()/len(df_nsd)
                robust_top.loc[k, "frac_05"] = ((df_nsd["normalized_score_drop"] > 0) & (df_nsd["normalized_score_drop"] <= 0.5)).sum()/len(df_nsd)
                robust_top.loc[k, "frac_10"] = (df_nsd["normalized_score_drop"] > 0.5).sum()/len(df_nsd)
                robust_top.loc[k, "frac_neg_random"] = (df_nsd["normalized_score_drop_random"] <= 0).sum()/len(df_nsd)
                robust_top.loc[k, "frac_05_random"] = ((df_nsd["normalized_score_drop_random"] > 0) & (df_nsd["normalized_score_drop_random"] <= 0.5)).sum()/len(df_nsd)
                robust_top.loc[k, "frac_10_random"] = (df_nsd["normalized_score_drop_random"] > 0.5).sum()/len(df_nsd)

            elif corruption_strategy == "bottom":
                robust_bottom.loc[k, df_acc.columns] = (np.sum(df_acc, axis=0) / df_acc.shape[0])
                robust_bottom.loc[k, df_nsd.columns] = np.nanmean(df_nsd.values, axis=0)
                robust_bottom.loc[k, "skewness"] = stats.skew(df_nsd["normalized_score_drop"].values, bias=True)
                robust_bottom.loc[k, "kurtosis"] = stats.kurtosis(df_nsd["normalized_score_drop"].values, bias=True)
                robust_bottom.loc[k, "skewness_random"] = stats.skew(df_nsd["normalized_score_drop_random"].values, bias=True)
                robust_bottom.loc[k, "kurtosis_random"] = stats.kurtosis(df_nsd["normalized_score_drop_random"].values, bias=True)

                robust_bottom.loc[k, "frac_neg"] = (df_nsd["normalized_score_drop"] <= 0).sum()/len(df_nsd)
                robust_bottom.loc[k, "frac_05"] = ((df_nsd["normalized_score_drop"] > 0) & (df_nsd["normalized_score_drop"] <= 0.5)).sum()/len(df_nsd)
                robust_bottom.loc[k, "frac_10"] = (df_nsd["normalized_score_drop"] > 0.5).sum()/len(df_nsd)
                robust_bottom.loc[k, "frac_neg_random"] = (df_nsd["normalized_score_drop_random"] <= 0).sum()/len(df_nsd)
                robust_bottom.loc[k, "frac_05_random"] = ((df_nsd["normalized_score_drop_random"] > 0) & (df_nsd["normalized_score_drop_random"] <= 0.5)).sum()/len(df_nsd)
                robust_bottom.loc[k, "frac_10_random"] = (df_nsd["normalized_score_drop_random"] > 0.5).sum()/len(df_nsd)
        
        robust_top = robust_top.rename(columns={
            "fraction_corrupted": "mean_fraction_corrupted",
            "normalized_score_drop": "mean_normalized_score_drop",
            "normalized_score_drop_random": "mean_normalized_score_drop_random",
        })
        robust_top.sort_index(ascending=True, inplace=True)
        robust_top.to_csv(os.path.join(self.save_results, f"summary_robust_{method_name}__top.csv"))

        robust_bottom.loc[:, "normalized_score_drop"] = robust_bottom.loc[:, "normalized_score_drop"].clip(lower=0)
        robust_bottom.loc[:, "normalized_score_drop_random"] = robust_bottom.loc[:, "normalized_score_drop_random"].clip(lower=0)
        robust_bottom = robust_bottom.rename(columns={
            "fraction_corrupted": "mean_fraction_corrupted",
            "normalized_score_drop": "mean_normalized_score_drop",
            "normalized_score_drop_random": "mean_normalized_score_drop_random",
        })
        robust_bottom.sort_index(ascending=True, inplace=True)
        robust_bottom.to_csv(os.path.join(self.save_results, f"summary_robust_{method_name}__bottom.csv"))


    def get_evaluation_metrics(self):
        path_summary = glob.glob(os.path.join(self.save_results, "summary_robust_*.csv"))
        summary_all = {}
        for path in path_summary:
            df_summary = pd.read_csv(path, index_col=0)
            name_file = os.path.split(path)[-1]
            method = re.search("summary_robust_(.+)\.csv", name_file).group(1)
            summary_all[method] = df_summary

        keys_dict = list(summary_all.keys())
        methods = list(set([x.split("__")[0] for x in keys_dict]))
        average_metrics = compute_average_metrics(summary_all, methods)
        updated_summary_all = self.scale_robust_properties(summary_all, methods)
        robust_metrics = compute_robust_metrics(updated_summary_all, methods)
        frac_metrics = compute_frac_metrics(updated_summary_all, methods)
        metrics = pd.concat([average_metrics, robust_metrics, frac_metrics], axis=1)

        AUCSm_top_random = metrics["AUCSm_top_random"].mean()
        F1Sm_random = metrics["F1Sm_random"].mean()
        AUCskew_random = metrics["AUCskew_random"].mean()
        AUCkurt_random = metrics["AUCkurt_random"].mean()
        F1_skew_kurt_random = metrics["F1_skew_kurt_random"].mean()
        AUCfrac_neg_random = metrics["AUCfrac_neg_random"].mean()
        AUCfrac_05_random = metrics["AUCfrac_05_random"].mean()
        AUCfrac_10_random = metrics["AUCfrac_10_random"].mean()

        metrics.drop(columns=["AUCSm_top_random", "F1Sm_random", 
                              "AUCskew_random", "AUCkurt_random", "F1_skew_kurt_random", 
                              "AUCfrac_neg_random", "AUCfrac_05_random", "AUCfrac_10_random"], inplace=True)
        metrics.loc["random"] = [AUCSm_top_random, F1Sm_random, 
                                 AUCskew_random, AUCkurt_random, F1_skew_kurt_random, 
                                 AUCfrac_neg_random, AUCfrac_05_random, AUCfrac_10_random]
        
        metrics.to_csv(os.path.join(self.save_results, "metrics_methods.csv"))
        return True
    

    def scale_robust_properties(self, summary_all, methods):
        updated_summary_all = {}
        skew_top = {}
        kurt_top = {}
        for method in methods:
            df_tmp_top = summary_all[f"{method}__top"]
            skew_top[method] = df_tmp_top["skewness"].values
            skew_top[f"{method}_random"] = df_tmp_top["skewness_random"].values
            kurt_top[method] = df_tmp_top["kurtosis"].values
            kurt_top[f"{method}_random"] = df_tmp_top["kurtosis_random"].values
        skew_scaled_dict = dict(zip(skew_top.keys(), scale_metrics(list(skew_top.values()))))
        kurt_scaled_dict = dict(zip(kurt_top.keys(), scale_metrics(list(kurt_top.values()))))
        for method in methods:
            df_tmp_top = summary_all[f"{method}__top"]
            df_tmp_top.loc[:, "skewness_scaled"] = skew_scaled_dict[method]
            df_tmp_top.loc[:, "skewness_scaled_random"] = skew_scaled_dict[f"{method}_random"]
            df_tmp_top.loc[:, "kurtosis_scaled"] = kurt_scaled_dict[method]
            df_tmp_top.loc[:, "kurtosis_scaled_random"] = kurt_scaled_dict[f"{method}_random"]
            df_tmp_top.to_csv(os.path.join(self.save_results, f"summary_robust_{method}__top.csv"))
            updated_summary_all[f"{method}__top"] = df_tmp_top
        return updated_summary_all


def compute_average_metrics(summary_all, methods):
    name_col = ["AUCSm_top", "F1Sm"]
    average_metrics = pd.DataFrame(index=methods, columns=name_col)
    for method in methods:
        df_tmp_top = summary_all[f"{method}__top"]
        df_tmp_bottom = summary_all[f"{method}__bottom"]

        normalized_score_drop_top = df_tmp_top["mean_normalized_score_drop"]
        normalized_score_drop_bottom = df_tmp_bottom["mean_normalized_score_drop"]

        y_integration_aucsm = np.append(0, normalized_score_drop_top)
        y_integration_aucsm = np.append(y_integration_aucsm, normalized_score_drop_top.iloc[-1])
        x_integration_aucsm = np.append(0, df_tmp_top.loc[:, "mean_fraction_corrupted"])
        # favouring of methods: achieve larger score drop with fewer points assigned with positive relevance
        x_integration_aucsm = np.append(x_integration_aucsm, 1)

        idx_small = np.argwhere(np.diff(x_integration_aucsm) < 10**-2)
        if len(idx_small) > 0:
            print(f"deleting points for {method}")
            # drop points as create instability in integration method
            x_integration_aucsm = np.delete(x_integration_aucsm, idx_small)
            y_integration_aucsm = np.delete(y_integration_aucsm, idx_small)

        average_metrics.loc[method, "AUCSm_top"] = integrate.simpson(y=y_integration_aucsm, x=x_integration_aucsm)

        y_integration_top = np.append(0, normalized_score_drop_top)
        x_integration_top = np.append(0, df_tmp_top.loc[:, "mean_fraction_corrupted"])
        integral_top = integrate.simpson(y=y_integration_top, x=x_integration_top)

        y_integration_bottom = np.append(0, normalized_score_drop_bottom)
        x_integration_bottom = np.append(0, df_tmp_bottom.loc[:, "mean_fraction_corrupted"])
        integral_bottom = integrate.simpson(y=y_integration_bottom, x=x_integration_bottom)

        F1Sm = (integral_top * (1 - integral_bottom)) / (integral_top + (1 - integral_bottom))
        average_metrics.loc[method, "F1Sm"] = F1Sm

        # if method == "integrated_gradients":
        normalized_score_drop_top_random = df_tmp_top.loc[:, "mean_normalized_score_drop_random"]
        normalized_score_drop_bottom_random = df_tmp_bottom.loc[:, "mean_normalized_score_drop_random"]

        y_integration_aucsm_random = np.append(0, normalized_score_drop_top_random)
        y_integration_aucsm_random = np.append(y_integration_aucsm_random, normalized_score_drop_top_random.iloc[-1])
        x_integration_aucsm_random = np.append(0, df_tmp_top.loc[:, "mean_fraction_corrupted"])
        x_integration_aucsm_random = np.append(x_integration_aucsm_random, 1)
        average_metrics.loc[method, "AUCSm_top_random"] = integrate.simpson(y=y_integration_aucsm_random, x=x_integration_aucsm_random)

        y_integration_top_random = np.append(0, normalized_score_drop_top_random)
        x_integration_top_random = np.append(0, df_tmp_top.loc[:, "mean_fraction_corrupted"])
        integral_top_random = integrate.simpson(y=y_integration_top_random, x=x_integration_top_random)

        y_integration_bottom_random = np.append(0, normalized_score_drop_bottom_random)
        x_integration_bottom_random = np.append(0, df_tmp_bottom.loc[:, "mean_fraction_corrupted"])
        integral_bottom_random = integrate.simpson(y=y_integration_bottom_random, x=x_integration_bottom_random)
        F1Sm_random = (integral_top_random * (1 - integral_bottom_random)) / (integral_top_random + (1 - integral_bottom_random))
        average_metrics.loc[method, "F1Sm_random"] = F1Sm_random

    return average_metrics


def compute_robust_metrics(updated_summary_all, methods):
    name_col = ["AUCskew", "AUCkurt"]
    robust_metrics = pd.DataFrame(index=methods, columns=name_col)
    for method in methods:
        df_tmp_top = updated_summary_all[f"{method}__top"]
        x_integration = df_tmp_top.index.values

        y_integration_skew = df_tmp_top["skewness_scaled"].values
        integral_skew = integrate.simpson(y=y_integration_skew, x=x_integration)

        y_integration_kurt = df_tmp_top["kurtosis_scaled"].values
        integral_kurt = integrate.simpson(y=y_integration_kurt, x=x_integration)

        robust_metrics.loc[method, "AUCskew"] = integral_skew
        robust_metrics.loc[method, "AUCkurt"] = integral_kurt

        F1_skew_kurt = (integral_kurt * (1 - integral_skew)) / (integral_kurt + (1 - integral_skew))
        robust_metrics.loc[method, "F1_skew_kurt"] = F1_skew_kurt

        y_integration_skew_random = df_tmp_top["skewness_scaled_random"].values
        integral_skew_random = integrate.simpson(y=y_integration_skew_random, x=x_integration)

        y_integration_kurt_random = df_tmp_top["kurtosis_scaled_random"].values
        integral_kurt_random = integrate.simpson(y=y_integration_kurt_random, x=x_integration)

        robust_metrics.loc[method, "AUCskew_random"] = integral_skew_random
        robust_metrics.loc[method, "AUCkurt_random"] = integral_kurt_random

        F1_skew_kurt_random = (integral_kurt_random * (1 - integral_skew_random)) / (integral_kurt_random + (1 - integral_skew_random))
        robust_metrics.loc[method, "F1_skew_kurt_random"] = F1_skew_kurt_random
    return robust_metrics


def compute_frac_metrics(updated_summary_all, methods):
    name_col = ["AUCfrac_neg", "AUCfrac_05", "AUCfrac_10"]
    frac_metrics = pd.DataFrame(index=methods, columns=name_col)
    for method in methods:
        df_tmp_top = updated_summary_all[f"{method}__top"]
        x_integration = df_tmp_top.index.values

        y_integration_neg = df_tmp_top["frac_neg"].values
        integral_neg = integrate.simpson(y=y_integration_neg, x=x_integration)

        y_integration_05 = df_tmp_top["frac_05"].values
        integral_05 = integrate.simpson(y=y_integration_05, x=x_integration)

        y_integration_10 = df_tmp_top["frac_10"].values
        integral_10 = integrate.simpson(y=y_integration_10, x=x_integration)

        frac_metrics.loc[method, "AUCfrac_neg"] = integral_neg
        frac_metrics.loc[method, "AUCfrac_05"] = integral_05
        frac_metrics.loc[method, "AUCfrac_10"] = integral_10

        y_integration_neg_random = df_tmp_top["frac_neg_random"].values
        integral_neg_random = integrate.simpson(y=y_integration_neg_random, x=x_integration)

        y_integration_05_random = df_tmp_top["frac_05_random"].values
        integral_05_random = integrate.simpson(y=y_integration_05_random, x=x_integration)

        y_integration_10_random = df_tmp_top["frac_10_random"].values
        integral_10_random = integrate.simpson(y=y_integration_10_random, x=x_integration)

        frac_metrics.loc[method, "AUCfrac_neg_random"] = integral_neg_random
        frac_metrics.loc[method, "AUCfrac_05_random"] = integral_05_random
        frac_metrics.loc[method, "AUCfrac_10_random"] = integral_10_random
    return frac_metrics


def scale_metrics(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))
