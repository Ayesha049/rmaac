"""
Plot joint noise sweep results from results/noise_sweep_maddpg/.

One figure with 6 subplots (one per MPE scenario).
X-axis: (act_noise_std, obs_noise_std) pairs, ordered by act then obs.
Y-axis: mean reward.
Three series per subplot: act-only noise, obs-only noise, joint noise.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.transforms import blended_transform_factory

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", "noise_sweep_m3ddpg")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "plots", "noise_sweep_m3ddpg.png")

SCENARIOS = [
    "simple_tag_m3ddpg",
    "simple_push_m3ddpg",
    "simple_adversary_m3ddpg",
    "simple_spread_m3ddpg",
    "simple_speaker_listener_m3ddpg",
    "simple_crypto_m3ddpg",
]

SERIES = [
    ("act_only_reward", "Act-only noise",  "#e05c5c", "o"),
    ("obs_only_reward", "Obs-only noise",  "#4a90d9", "s"),
    ("joint_reward",    "Joint noise",     "#2ca02c", "^"),
]


def load_scenario(scenario: str) -> pd.DataFrame:
    pattern = os.path.join(RESULT_DIR, "{}_expbest_noise_sweep.csv".format(scenario))
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError("No CSV found for scenario '{}' at {}".format(scenario, pattern))
    df = pd.read_csv(matches[0])
    df = df.sort_values(["act_noise_std", "obs_noise_std"]).reset_index(drop=True)
    df["label"] = df.apply(
        lambda r: "({:.1f}, {:.1f})".format(r.act_noise_std, r.obs_noise_std), axis=1
    )
    return df


def make_figure():
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()

    legend_handles = None

    for ax, scenario in zip(axes, SCENARIOS):
        try:
            df = load_scenario(scenario)
        except FileNotFoundError as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="red")
            ax.set_title(scenario)
            continue

        x = range(len(df))
        handles = []
        for col, label, color, marker in SERIES:
            (line,) = ax.plot(x, df[col], marker=marker, color=color, label=label,
                              linewidth=1.8, markersize=5)
            handles.append(line)

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

        # X-axis ticks: show every label but rotate to avoid crowding
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["label"].tolist(), rotation=60, ha="right", fontsize=7)
        ax.xaxis.set_minor_locator(ticker.NullLocator())

        # Mark boundaries between act_noise_std groups with vertical lines
        act_vals = df["act_noise_std"].unique()
        group_size = len(df["obs_noise_std"].unique())
        for i, _ in enumerate(act_vals[1:], 1):
            ax.axvline(i * group_size - 0.5, color="grey", linewidth=0.7,
                       linestyle=":", alpha=0.6)

        # Annotate act_noise_std group labels along the top (blended: data-x, axes-y)
        top_transform = blended_transform_factory(ax.transData, ax.transAxes)
        for i, act_val in enumerate(act_vals):
            mid = i * group_size + (group_size - 1) / 2
            ax.text(mid, 1.02, "act={:.1f}".format(act_val),
                ha="center", va="bottom", fontsize=7, color="dimgray",
                transform=top_transform, clip_on=False)

        ax.set_title(scenario.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("(act_noise_std, obs_noise_std)", fontsize=9)
        ax.set_ylabel("Mean reward", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        if legend_handles is None:
            legend_handles = handles

    if legend_handles:
        fig.legend(legend_handles, [s[1] for s in SERIES],
                   loc="lower center", ncol=3, fontsize=10,
                   bbox_to_anchor=(0.5, 0.0), frameon=True)

    fig.suptitle("MADDPG Noise Sensitivity Sweep — 6 MPE Scenarios",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    return fig


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig = make_figure()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print("Saved to {}".format(OUTPUT_PATH))
    plt.show()
