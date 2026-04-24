#!/usr/bin/env python3
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


PATTERN = "*_actstd_tstart_sweep.csv"
FILE_RE = re.compile(r"^(?P<scenario>.+)__(?P<variant>clean|adv_act|compare)_actstd_tstart_sweep\.csv$")


def collect_csvs(base_dir):
    grouped = {}
    for path in glob.glob(os.path.join(base_dir, PATTERN)):
        name = os.path.basename(path)
        m = FILE_RE.match(name)
        if not m:
            continue

        scenario = m.group("scenario")
        variant = m.group("variant")
        grouped.setdefault(scenario, {})[variant] = path
    return grouped


def load_curve(csv_path):
    df = pd.read_csv(csv_path)
    required = {"action_noise_std", "reward_noise_no_diffusion", "reward_no_noise"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing columns {} in {}".format(sorted(missing), csv_path))

    df = df.sort_values("action_noise_std")
    x = df["action_noise_std"].astype(float).values
    y = df["reward_noise_no_diffusion"].astype(float).values
    baseline = float(df["reward_no_noise"].iloc[0])
    return x, y, baseline


def load_compare_curve(csv_path):
    df = pd.read_csv(csv_path)
    required = {"action_noise_std", "reward_base_noise_no_diffusion", "reward_adv_noise_no_diffusion", "reward_clean_no_noise"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing columns {} in {}".format(sorted(missing), csv_path))

    df = df.sort_values("action_noise_std")
    x = df["action_noise_std"].astype(float).values
    base_y = df["reward_base_noise_no_diffusion"].astype(float).values
    adv_y = df["reward_adv_noise_no_diffusion"].astype(float).values
    clean_base = float(df["reward_clean_no_noise"].iloc[0])

    diff_cols = [col for col in df.columns if col.startswith("reward_adv_with_diff_t")]
    diff_y = None
    diff_label = None
    if diff_cols:
        best_col = None
        best_values = None
        best_mean = None
        for col in diff_cols:
            values = df[col].astype(float).values
            mean_value = float(values.mean())
            if best_mean is None or mean_value > best_mean:
                best_col = col
                best_values = values
                best_mean = mean_value
        diff_y = best_values
        diff_label = best_col

    return x, base_y, adv_y, diff_y, clean_base, diff_label


def make_plot(scenario, files, out_dir):
    if "compare" in files:
        x, base_y, adv_y, diff_y, clean_base, diff_label = load_compare_curve(files["compare"])

        plt.figure(figsize=(8, 5))
        plt.plot(x, base_y, marker="o", linewidth=2, label="base model noisy")
        plt.plot(x, adv_y, marker="s", linewidth=2, label="adv model noisy")

        if diff_y is not None:
            diff_text = "adv + diffusion"
            if diff_label is not None:
                diff_text = "adv + diffusion ({})".format(diff_label.replace("reward_adv_with_diff_t", "t="))
            plt.plot(x, diff_y, marker="^", linewidth=2, label=diff_text)

        plt.axhline(clean_base, linestyle="--", linewidth=1.2, alpha=0.8, label="clean baseline")

        plt.title("{}: reward vs action noise std".format(scenario))
        plt.xlabel("Action noise std")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "{}_compare_with_diffusion.png".format(scenario))
        plt.savefig(out_path, dpi=160)
        plt.close()
        return out_path

    clean_x, clean_y, clean_base = load_curve(files["clean"])
    adv_x, adv_y, adv_base = load_curve(files["adv_act"])

    plt.figure(figsize=(8, 5))
    plt.plot(clean_x, clean_y, marker="o", linewidth=2, label="clean model")
    plt.plot(adv_x, adv_y, marker="s", linewidth=2, label="adv_act model")

    plt.axhline(clean_base, linestyle="--", linewidth=1.2, alpha=0.8, label="clean baseline")
    plt.axhline(adv_base, linestyle=":", linewidth=1.2, alpha=0.8, label="adv_act baseline")

    plt.title("{}: reward vs action noise std".format(scenario))
    plt.xlabel("Action noise std")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "{}_clean_vs_adv_act.png".format(scenario))
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "plots", "action_noise_ablation")

    grouped = collect_csvs(here)
    if not grouped:
        print("No matching CSV files found in {}".format(here))
        return

    generated = []
    skipped = []

    for scenario in sorted(grouped.keys()):
        files = grouped[scenario]
        if "compare" in files:
            out_path = make_plot(scenario, files, out_dir)
            generated.append(out_path)
            continue

        if "clean" not in files or "adv_act" not in files:
            skipped.append(scenario)
            continue

        out_path = make_plot(scenario, files, out_dir)
        generated.append(out_path)

    if generated:
        print("Generated {} plots:".format(len(generated)))
        for path in generated:
            print(" - {}".format(path))
    else:
        print("No plots generated (missing clean/adv_act pairs).")

    if skipped:
        print("Skipped scenarios without both clean and adv_act CSVs:")
        for scenario in skipped:
            print(" - {}".format(scenario))


if __name__ == "__main__":
    main()