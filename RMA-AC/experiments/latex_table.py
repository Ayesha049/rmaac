import pandas as pd

# === Load CSV ===
df = pd.read_csv("ccRun1_test_rewards.csv")

# Rename columns if needed (optional)
col_map = {
    "gauss_obs_only": "gauss\_obs",
    "gauss_obs+action": "gauss\_obs+act",
    "gauss_act_only": "gauss\_act",
    "shift_obs_only": "shift\_obs",
    "shift_obs+action": "shift\_obs+act",
    "shift_act_only": "shift\_act",
    "uniform_obs_only": "uniform\_obs",
    "uniform_obs+action": "uniform\_obs+act",
    "uniform_act_only": "uniform\_act"
}
df.rename(columns=col_map, inplace=True)

# Reorder columns for consistent table
desired_order = ["run", "none", "gauss\_obs", "gauss\_obs+act", "gauss\_act",
                 "shift\_obs", "shift\_obs+act", "shift\_act",
                 "uniform\_obs", "uniform\_obs+act", "uniform\_act"]
df = df[[c for c in desired_order if c in df.columns]]

# === Start LaTeX table ===
latex = []
latex.append("\\begin{table*}[!ht]")
latex.append("\\centering")
latex.append("\\begin{tabular}{|c|" + "c" * (len(df.columns) - 1) + "|}")
latex.append("\\hline")
latex.append("Run & " + " & ".join(df.columns[1:]) + " \\\\")
latex.append("\\hline")

# === Row-wise processing ===
for _, row in df.iterrows():
    vals = row.values
    run_id = str(int(row["run"]))
    numeric = [row[c] for c in df.columns if c != "run" and isinstance(row[c], (int, float))]

    min_v = min(numeric)
    max_v = max(numeric)

    entries = []
    for c in df.columns:
        v = row[c]
        if c == "run":
            entries.append(run_id)
        elif v == max_v:
            entries.append("\\textbf{\\textcolor{green}{" + "{0:.2f}".format(v) + "}}")
        elif v == min_v:
            entries.append("\\textbf{\\textcolor{red}{" + "{0:.2f}".format(v) + "}}")
        else:
            entries.append("{0:.2f}".format(v))
    latex.append(" & ".join(entries) + " \\\\")

# === Average row ===
avg = df.drop("run", axis=1).mean()
min_v = avg.min()
max_v = avg.max()
entries = ["\\textbf{Avg.}"]
for v in avg:
    if v == max_v:
        entries.append("\\textbf{\\textcolor{green}{" + "{0:.2f}".format(v) + "}}")
    elif v == min_v:
        entries.append("\\textbf{\\textcolor{red}{" + "{0:.2f}".format(v) + "}}")
    else:
        entries.append("{0:.2f}".format(v))
latex.append("\\hline")
latex.append(" & ".join(entries) + " \\\\")
latex.append("\\hline")

latex.append("\\end{tabular}")
latex.append("\\vspace{2mm}")
latex.append("\\caption{\\textbf{RMA-AC-\\textcolor{green}{taking best rewarding model}} - evaluation on 1000 episodes with observation and/or action perturbation with different noises on \\textbf{MPE Cooperative Communication scenario}}")
latex.append("\\end{table*}")

# === Print or Save ===
latex_output = "\n".join(latex)
print(latex_output)
