import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator


def visualise_flagnum_heatmap(df):
    # Pivot the data: Rows=t, Columns=n, Values=acceptance_rate
    pivot_table = df.pivot_table(index='t', columns='n', values='num_flags', aggfunc='mean')

    pivot_table.sort_index(ascending=False, inplace=True)

    # --- DYNAMIC SIZING LOGIC ---
    num_rows = len(pivot_table.index)
    num_cols = len(pivot_table.columns)

    cell_size = 0.6
    fig_width = (num_cols * cell_size)
    fig_height = (num_rows * cell_size) + (3 * cell_size)

    # Create figure with calculated dimensions
    plt.figure(figsize=(fig_width, fig_height), dpi=300)

    # Plot Heatmap
    ax = sns.heatmap(
        pivot_table,
        annot=True,
        # fmt=".1%",
        cmap="RdYlBu",
        linewidths=0.5,
        linecolor='white',
        square=True,
        annot_kws={"size": 10},
        cbar_kws={'label': 'Number of flags', 'shrink': 0.7}
    )

    # Styling
    ax.set_title("Number of flags", fontsize=14)
    ax.set_xlabel("Cat State Size (n)", fontsize=12)
    ax.set_ylabel("Fault-distance (t)", fontsize=12)

    # Ensure X and Y ticks are horizontal and visible
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    ax.tick_params(left=True, bottom=True, length=5)

    plt.tight_layout()
    plt.savefig(f"simulation_data/AR_heatmap.png")
    # plt.show()
    plt.close()


def visualise_clean_stacked_comparison(methods_data_dict):
    # 1. Prepare Data and find Global Limits
    pivots = {}
    all_n = set()
    all_t = set()
    v_min, v_max = float('inf'), float('-inf')

    for name, data in methods_data_dict.items():
        df = pd.DataFrame(data)
        df_filtered = df[(df['n'] <= 50) & ((df['n'] // 2) > df['t'])].copy()
        pt = df_filtered.pivot_table(index='t', columns='n', values='num_cx', aggfunc='mean')
        pt.sort_index(ascending=False, inplace=True)
        pivots[name] = pt

        # Track all possible n and t values to create a master grid
        all_n.update(pt.columns)
        all_t.update(pt.index)
        v_min = min(v_min, pt.min().min())
        v_max = max(v_max, pt.max().max())

    # Sort the master coordinates
    master_n = sorted(list(all_n))
    master_t = sorted(list(all_t), reverse=True)

    # --- DISCRETE COLORBAR LOGIC ---
    # Create discrete boundaries (integers from v_min to v_max)
    # If num_flags are floats, you can adjust the step (e.g., np.arange(v_min, v_max + 0.5, 0.5))
    boundaries = np.arange(np.floor(v_min), np.ceil(v_max) + 1, 1)
    n_colors = len(boundaries) - 1
    cmap = plt.get_cmap("cubehelix_r", n_colors)
    norm = BoundaryNorm(boundaries, n_colors)

    # 2. Setup Figure
    num_methods = len(pivots)
    # Taller figure, slightly wider to accommodate the n=100 case
    fig, axes = plt.subplots(num_methods, 1, figsize=(20, 3.3 * num_methods),
                             sharex=False, sharey=False, dpi=150)

    if num_methods == 1: axes = [axes]

    # Create trivial mask: n <= t/2
    trivial_mask = pd.DataFrame(False, index=master_t, columns=master_n)
    for t_val in master_t:
        for n_val in master_n:
            if t_val >= n_val // 2:
                trivial_mask.loc[t_val, n_val] = True

    # 3. Plot each method
    for i, (name, pt) in enumerate(pivots.items()):
        # IMPORTANT: Reindex so every plot has the same columns/rows as the largest one
        # This aligns the "Spider Cat" perfectly with the smaller methods
        aligned_pt = pt.reindex(index=master_t, columns=master_n)

        # Create a mask: True for cells with no data (NaN)
        mask = aligned_pt.isnull()

        sns.heatmap(
            aligned_pt,
            ax=axes[i],
            mask=mask,  # This hides the "empty" cells entirely
            annot=True,
            fmt=".0f",
            square=True,
            cmap=cmap,
            norm=norm,  # Apply the discrete norm here
            cbar=False,  # We will add one single colorbar at the end
            linewidths=.5,
            linecolor='#eeeeee',
            annot_kws={"size": 12}
        )

        # PASS 1: Plot the "Trivial" grey squares
        # We use a solid grey color for anything in the trivial mask
        sns.heatmap(
            trivial_mask.astype(int),
            mask=~trivial_mask,  # Hide non-trivial cells
            ax=axes[i],
            cmap=ListedColormap(['#d3d3d3']),  # Light Grey
            cbar=False,
            annot=False,
            square=True,
            linewidths=.5,
        )

        if name == "SpiderCat":
            special_mask = pd.DataFrame(0, index=master_t, columns=master_n)
            special_mask_label = pd.DataFrame("", index=master_t, columns=master_n)
            special_mask.loc[5, 12] = 1
            special_mask_label.loc[5, 12] = "*"
            special_mask.loc[7, 26] = 3
            special_mask_label.loc[7, 26] = "#"
            special_mask.loc[6, range(14, 21)] = 2
            special_mask.loc[7, range(16, 24)] = 2
            special_mask_label.loc[6, range(14, 21)] = "$\\dagger$"
            special_mask_label.loc[7, range(16, 24)] = "$\\dagger$"
            sns.heatmap(
                special_mask.astype(float),
                mask=special_mask == 0,  # Hide non-trivial cells
                ax=axes[i],
                cmap=sns.color_palette("bright"),
                cbar=False,
                annot=special_mask_label,
                fmt='',
                square=True,
                linewidths=.5,
                annot_kws={"size": 12}
            )

        axes[i].set_title(f"{name} (Number of CNOTs)", fontweight='bold', loc='left', fontsize=18, pad=10)
        axes[i].set_ylabel("Fault-distance (t)", fontsize=12)
        axes[i].set_xlabel("")
        axes[i].tick_params(labelsize=12)
        # axes[i].set_xticklabels(axes[i].get_xticks(), size=12)
        # axes[i].set_yticklabels(axes[i].get_yticks(), size=12)
        # axes[i].tick_params(axis='both', which='both', length=0)  # Clean look

    # 4. Global Styling
    axes[-1].set_xlabel("Cat State Size (n)", fontsize=15, labelpad=15)

    # 5. Manual Layout & Colorbar
    # subplots_adjust is better than tight_layout when using add_axes
    # We leave a large bottom margin (0.15) for the colorbar and X-labels
    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.2, hspace=0.35)

    # Position the colorbar AXIS relative to the figure: [left, bottom, width, height]
    # We put it lower (0.07) so it doesn't hit the X-axis label
    cbar_ax = fig.add_axes([0.125, -0.02, 0.75, 0.02])

    # Create the discrete colorbar
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation='horizontal',
        ticks=np.arange(np.floor(v_min), np.ceil(v_max) + 1, 4)  # Ensures ticks land on the discrete boundaries
    )
    cb.ax.tick_params(labelsize=12)
    cb.set_label('Number of CNOTs', fontsize=15, labelpad=12)

    plt.tight_layout()
    plt.savefig("simulation_data/cx_heatmap.pdf", bbox_inches='tight', dpi=1200)
    plt.close()


# Run it with your methods_data_dict


def visualise_pk_per_n(df, t):
    # Filter for only 1 <= k <= 5
    df_filtered = df[(df['n'] >= 8) & (df['t'] == t) & df['k'].between(1, 10)].copy()

    if df_filtered.empty:
        print("No data found for 1 <= k <= 5.")
        return

    # Create the label for the legend
    # FIX: We use 'k' to generate the label "k=1", "k=2", etc.
    df_filtered['k_label'] = df_filtered['k'].apply(lambda x: f"k={int(x)}")

    # Sort to ensure legend appears in order k=1, k=2, ...
    df_filtered.sort_values(by=['k', 'n'], inplace=True)

    # ---------------------------------------------------------
    # 3. Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=120)

    # Use the 'viridis' palette exactly as requested before
    sns.lineplot(
        data=df_filtered,
        x='n',
        y='probability',  # This matches the key in your new stats dict
        hue='k_label',
        style='k_label',
        markers=['o'] * 5,
        dashes=False,
        palette='viridis',
        markersize=8,
        linewidth=2
    )

    # Y-Axis Log Scale
    plt.yscale('log')

    # Axis Labels
    plt.xlabel("Cat State Size", fontsize=12)
    plt.ylabel("$P_k$", fontsize=12)

    # Grid Styling (Light dashed lines)
    plt.grid(True, which="both", ls="-", color='lightgrey', alpha=0.5)

    # Legend Styling (Top, Horizontal, No Box)
    # bbox_to_anchor moves it above the plot, ncol=5 makes it horizontal
    plt.legend(
        title="",
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(f"simulation_data/Pk_per_n_at_t{t}.png", dpi=1200)
    # plt.show()
    plt.close()


def visualise_pk_per_t_1(df, n):
    results = []

    grouped = df.groupby(['n', 't', 'p'])

    for (n, t, p), group in grouped:
        acc_rate = group['acceptance_rate'].iloc[0]

        # Calculate expected value: Sum(k * probability)
        # 'probability' here is P(k | accepted)
        mean_faults = (group['k'] * group['probability']).sum()

        results.append({
            'n': n,
            't': t,
            'p': p,
            'acceptance_rate': acc_rate,
            'mean_faults': mean_faults
        })

    df_summary = pd.DataFrame(results)
    df_summary['mean_faults'] = df_summary['mean_faults'].replace(0, np.nan)

    plt.figure(figsize=(10, 6), dpi=300)
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df_summary,
        x='p',
        y='mean_faults',
        hue='t',
        palette='viridis',
        marker='s',
        linewidth=2
    )
    plt.xscale('log')
    plt.title(f"Threshold Plot: Average Faults vs Physical Error @ n={n}")
    plt.ylabel(r"Average Number of Faults ($\mathbb{E}[k]$)")
    plt.xlabel("Physical Error Rate ($p$)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title="Error Probability", loc='upper left')
    ax2 = plt.twinx()

    sns.lineplot(
        data=df_summary,
        x='p',
        y='acceptance_rate',
        hue='t',
        palette='viridis',
        marker='o',
        linestyle=':',
        linewidth=1.5,
        alpha=0.5,
        ax=ax2,
    )
    plt.ylabel(r"Acceptance Rate")
    plt.legend(title="Acceptance Rate", loc='upper right')
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    plt.tight_layout()
    plt.savefig(f"simulation_data/EPk_per_p_at_n{n}.pdf", dpi=1200)
    # plt.show()
    plt.close()


def visualise_pk_per_t_2(df, n):
    results = []

    df_filtered = df[df['k'] <= 4]
    grouped = df_filtered.groupby(['n', 't', 'p'])

    for (n, t, p), group in grouped:
        acc_rate = group['acceptance_rate'].iloc[0]

        # Calculate expected value: Sum(k * probability)
        # 'probability' here is P(k | accepted)
        mean_faults = (group['probability']).sum()

        results.append({
            'n': n,
            't': t,
            'p': p,
            'acceptance_rate': 1 - acc_rate,
            'mean_faults': mean_faults
        })

    df_summary = pd.DataFrame(results)
    df_summary['mean_faults'] = df_summary['mean_faults'].replace(0, np.nan)

    plt.figure(figsize=(10, 6), dpi=100)
    sns.lineplot(
        data=df_summary,
        x='p',
        y='mean_faults',
        hue='t',
        palette='viridis',
        marker='s',
        linewidth=2
    )
    plt.xscale('log')
    plt.title(f"Probability of 4 or less faults vs Physical Error @ n={n}")
    plt.ylabel(r"Probability of 4 or less faults ($\mathbb{P}(k \leq 4)$)")
    plt.xlabel("Physical Error Rate ($p$)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title="Error Probability", loc='upper left')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax2 = plt.twinx()

    sns.lineplot(
        data=df_summary,
        x='p',
        y='acceptance_rate',
        hue='t',
        palette='viridis',
        marker='o',
        linestyle=':',
        linewidth=1.5,
        alpha=0.5,
        ax=ax2,
    )
    plt.ylabel(r"Rate of Post-Selection")
    plt.legend(title="Post-Selection", loc='upper right')
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    plt.tight_layout()
    plt.savefig(f"simulation_data/k_less_4_per_p_at_n{n}.png")
    # plt.show()
    plt.close()


def visualise_method_comparison(methods_data_dict, t):
    """
    Compares multiple methods for a fixed fault distance t with Dual Axis.

    Args:
        methods_data_dict (dict): Keys represent method names, Values are data lists.
        t (int): The fault distance to filter by.
        plot_as_failure_rate (bool): If True, plots failure rate (Log Scale).
    """
    results = []

    # 1. Data Aggregation
    for method_name, raw_data in methods_data_dict.items():

        # Filter for relevant scope
        # Note: We filter n >= 8 and t == t
        if isinstance(raw_data, tuple):
            t_extra = raw_data[1]
            raw_data = raw_data[0]
            df = pd.DataFrame(raw_data)
            scope_df = df[df['n'].between(10, 50) & (df['t'] == (t + t_extra))]
        else:
            df = pd.DataFrame(raw_data)
            scope_df = df[df['n'].between(10, 50) & (df['t'] == t)]

        if scope_df.empty:
            print(f"Warning: No data for method '{method_name}' at t={t}")
            continue

        # Group by 'n' to calculate metrics per cat state size
        for n, group in scope_df.groupby('n'):
            # Metric 1: Probability of success (k < t)
            # Sum probability of all k where k < t
            # success_prob = group[group['k'] <= t]['probability'].sum()
            success_prob = (group['k'] * group['count']).sum() / group['count'].sum()
            # if 1.0 - success_prob < 1e-8:
            #     continue
            # Metric 2: Acceptance Rate (Constant for a specific simulation n,t)
            # We take the mean or just the first value
            acc_rate = group['acceptance_rate'].iloc[0]
            num_flags = group['num_flags'].iloc[0]
            num_cx = group['num_cx'].iloc[0]
            depth = group['depth'].iloc[0]

            results.append({
                'n': n,
                'method': method_name,
                'success_prob': success_prob,
                'failure_prob': 1.0 - success_prob,
                'acceptance_rate': acc_rate,
                'num_flags': num_flags,
                'num_cx': num_cx,
                'depth': depth,
            })

    if not results:
        print("No valid data found to plot.")
        return

    plot_df = pd.DataFrame(results)

    # 2. Setup Plot
    fig, ax1 = plt.subplots(figsize=(10, 7), dpi=120)
    ax2 = ax1.twinx()  # Create secondary Y-axis

    # Assign distinct colors to each method
    unique_methods = plot_df['method'].unique()
    palette = sns.color_palette("bright", len(unique_methods))
    method_colors = dict(zip(unique_methods, palette))

    # 3. Plotting Loop
    for method in unique_methods:
        subset = plot_df[plot_df['method'] == method].sort_values('n')
        color = method_colors[method]

        # --- Primary Axis (Left): Probability ---
        y_val = subset['success_prob']

        ax1.plot(
            subset['n'], y_val,
            color=color, linestyle='-', linewidth=2, marker='o',
            label=method  # Label for legend
        )

        # --- Secondary Axis (Right): Acceptance Rate ---
        ax2.plot(
            subset['n'], (subset['n'] + subset['num_flags']) * subset['depth'] / subset['acceptance_rate'],
            color=color, linestyle=':', linewidth=1.5, marker='x', alpha=0.7
        )

    # 4. Styling & Legends

    # Left Axis Styling
    ax1.set_ylabel(f"Probability of $> {t}$ Faults (Failure)", fontsize=12)
    # ax1.set_yscale('log')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_xlabel("Cat State Size (n)", fontsize=12)
    ax1.grid(True, which="both", ls="--", color='lightgrey', alpha=0.5)

    second_y_axis_label = {
        "acceptance_rate": "Acceptance Rate",
        "num_flags": "Number of Flags",
        "num_cx": "Number of CNOTs",
    }
    int_secondary_y_axis = ("num_flags", "num_cx")

    # Right Axis Styling
    ax2.set_ylabel("Expected Circuit Volume", fontsize=12, rotation=270, labelpad=15)
    # if second_y_axis in int_secondary_y_axis:
    #     ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    #     ax2.invert_yaxis()
    # else:
    # ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax2.set_yscale('log')
    # ax1.invert_yaxis()

    # Combined Legend Construction
    # Part A: Method Colors
    handles, labels = ax1.get_legend_handles_labels()
    legend1 = ax1.legend(handles, labels, title="Method", loc='upper center')
    ax1.add_artist(legend1)  # Preserve first legend

    # Part B: Line Styles (Explanation)
    style_lines = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', marker='o'),
        Line2D([0], [0], color='black', lw=1.5, linestyle=':', marker='x')
    ]

    style_labels = ['Error Probability', "Expected Circuit Volume"]
    ax1.legend(style_lines, style_labels, loc='lower center')

    plt.title(f"Method Comparison: Error Probability vs CAT state size (t={t})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"simulation_data/k_less_t_per_n_at_t{t}.png")
    # plt.show()
    plt.close()


def visualise_two_panel_hybrid(methods_data_dict, t):
    """
    Compares multiple methods using a 2-panel plot (3:2 ratio).
    Top panel: Error Probability.
    Bottom panel: Dual Y-axis for Acceptance Rate and Number of Flags.
    """
    results = []

    # 1. Data Aggregation (Unchanged)
    for method_name, raw_data in methods_data_dict.items():
        if isinstance(raw_data, tuple):
            t_extra = raw_data[1]
            df = pd.DataFrame(raw_data[0])
            scope_df = df[df['n'].between(10, 50) & (df['t'] == (t + t_extra))]
        else:
            df = pd.DataFrame(raw_data)
            scope_df = df[df['n'].between(10, 50) & (df['t'] == t)]

        if scope_df.empty:
            continue

        for n, group in scope_df.groupby('n'):
            success_prob = group[group['k'] <= t]['probability'].sum()
            if 1.0 - success_prob < 1e-8:
                continue

            results.append({
                'n': n,
                'method': method_name,
                'failure_prob': 1.0 - success_prob,
                'acceptance_rate': group['acceptance_rate'].iloc[0],
                'num_flags': group['num_flags'].iloc[0],
                'depth': group['depth'].iloc[0],
            })

    if not results:
        print("No valid data found to plot.")
        return

    plot_df = pd.DataFrame(results)

    # 2. Setup 2-Panel Plot (3:2 Ratio)
    fig, (ax1, ax3) = plt.subplots(
        2, 1,
        figsize=(8, 11),
        dpi=1200,
        sharex=True,
        gridspec_kw={'height_ratios': [1, 1]}  # Top is 3 parts, Bottom is 2 parts
    )

    # Create the dual Y-axis for the bottom panel
    ax2 = ax3.twinx()

    unique_methods = plot_df['method'].unique()
    palette = sns.color_palette("colorblind", n_colors=4)
    method_colors = {
        "Flag at Origin": palette[2],
        "SpiderCat": palette[3],
        "MQT": palette[0],
    }

    # 3. Plotting Loop
    for method in unique_methods:
        subset = plot_df[plot_df['method'] == method].sort_values('n')
        color = method_colors[method]

        # Top Panel: Failure Probability
        ax1.plot(subset['n'], subset['failure_prob'] / subset['acceptance_rate'], color=color, linestyle='-', marker='o', label=method)

        # Bottom Panel (Left Axis): Acceptance Rate
        ax3.plot(subset['n'], subset['depth'], color=color, linestyle='--', marker='s', alpha=0.8, markersize=5)
        ax2.plot(subset['n'], subset['acceptance_rate'], color=color, linestyle=':', marker='*', alpha=0.8)
        ax2.set_yscale("log")

        # Bottom Panel (Right Axis): Number of Flags

    # 4. Styling & Legends

    # --- Top Panel (Error Rate) ---
    ax1.set_ylabel(f"Probability of $> {t}$ Faults", fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", color='lightgrey', alpha=0.5)
    # ax1.set_title(f"Method Comparison vs Cat State Size (n) at t={t}", fontsize=14)

    # Legend for the methods (Top Panel)
    ax1.legend(title="Method", loc='best')

    # Right Y-Axis (Number of Flags)
    ax3.set_ylabel("Depth", fontsize=15)
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Bottom Panel (Acceptance Rate & Flags) ---
    ax3.set_xlabel("Cat State Size (n)", fontsize=15)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Left Y-Axis (Acceptance Rate)
    ax2.set_ylabel("Acceptance Rate", fontsize=15, rotation=270, labelpad=15)
    ax3.grid(True, ls="--", color='lightgrey', alpha=0.5)

    ax2.tick_params(labelsize=12)
    ax3.tick_params(labelsize=12)

    # Custom legend for the bottom panel to explain line styles
    style_lines = {
        'Acceptance Rate': Line2D([0], [0], color='gray', linestyle=':', marker='*'),
        'Number of Flags': Line2D([0], [0], color='gray', linestyle='--', marker='s', markersize=5),
    }
    # stile_lines2 = {
    #     m: Line2D([0], [0], color=color, linestyle='--', marker='s', markersize=5) for m, color in method_colors.items()
    #     if m in unique_methods
    # }
    # ax3.legend(stile_lines2.values(), stile_lines2.keys(), loc='center left', title="Method")
    ax2.legend(style_lines.values(), style_lines.keys(), loc='center left')

    # Bring panels closer together
    plt.subplots_adjust(hspace=0.08)

    plt.savefig(f"simulation_data/two_panel_hybrid_t{t}.pdf", dpi=1200, bbox_inches='tight')
    plt.savefig(f"simulation_data/two_panel_hybrid_t{t}.png", bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import json

    with open(f"simulation_data/simulation_results_t_n_spider-cat_p1.json", "r") as f:
        df_sc_tree = pd.DataFrame(json.load(f))
    # with open(f"simulation_data/simulation_results_t_n_spider-cat_p5.json", "r") as f:
    #     df_sc_p5 = pd.DataFrame(json.load(f))
    # with open(f"simulation_data/simulation_results_t_n_spider-cat_p10.json", "r") as f:
    #     df_sc_p10 = pd.DataFrame(json.load(f))
    # with open(f"simulation_data/simulation_results_t_n_spider-cat_p20.json", "r") as f:
    #     df_sc_p20 = pd.DataFrame(json.load(f))
    with open(f"simulation_data/simulation_results_t_n_flag-at-origin_p1.json", "r") as f:
        df_FAO = pd.DataFrame(json.load(f))
    with open(f"simulation_data/simulation_results_t_n_MQT_p1.json", "r") as f:
        df_MQT = pd.DataFrame(json.load(f))
    methods = {
        "SpiderCat": df_sc_tree,
        "MQT": df_MQT,
        "Flag at Origin": df_FAO,
        # "SpiderCat (H-Path)": df_sc_ham,
        # "SpiderCat (T≈13)": (df_sc_inf, math.inf),
        # "SpiderCat (Prime Inv.)": (df_sc_inf_prime, math.inf),
        # "SpiderCat (Tree T-1)": (df_sc_tree, -1),
        # "SpiderCat (Tree T+1)": (df_sc_tree, 1),
        # "SpiderCat (Tree T+2)": (df_sc_tree, 2),
        # "SpiderCat (Tree T+3)": (df_sc_tree, 3),
        # "SpiderCat (3-Forest)": df_sc_p3,
        # "SpiderCat (5-Forest)": df_sc_p5,
        # "SpiderCat (5-Forest T+1)": (df_sc_p5, 1),
        # "SpiderCat (5-Forest T+2)": (df_sc_p5, 2),
        # "SpiderCat (5-Forest T+3)": (df_sc_p5, 3),
        # "SpiderCat (5-Forest)": df_sc_p5,
        # "SpiderCat (2-Path)": df_sc_p2,
        # "SpiderCat (3-Path)": df_sc_p3,
        # "SpiderCat (4 forest)": df_sc_p4,
        # "SpiderCat (5 forest)": df_sc_p5,
        # "SpiderCat (10 forest)": df_sc_p10,
        # "SpiderCat (20 forest)": df_sc_p20,
    }
    visualise_method_comparison(methods, t=1)
    visualise_method_comparison(methods, t=2)
    visualise_method_comparison(methods, t=3)
    # visualise_method_comparison(methods, t=4)
    visualise_method_comparison(methods, t=5)
    # visualise_method_comparison(methods, t=6)
    # visualise_method_comparison(methods, t=4, second_y_axis='num_flags')
    # visualise_two_panel_hybrid(methods, t=3)
    # visualise_two_panel_hybrid(methods, t=4)
    # visualise_two_panel_hybrid(methods, t=5)
    # visualise_two_panel_hybrid(methods, t=6)
    # visualise_clean_stacked_comparison(methods)
    # visualise_method_comparison(methods, t=6, second_y_axis='num_flags')
    # visualise_method_comparison(methods, t=7, second_y_axis='num_flags')
    #
    # # with open(f"simulation_data/simulation_results_t_n.json", "r") as f:
    # #     collected_data = json.load(f)
    # # df_t_n = pd.DataFrame(collected_data)
    # #
    # visualise_acceptance_heatmap(df_sc_tree)
    # for t in [3, 4, 5, 6, 7]:
    #     visualise_pk_per_n(df_sc_tree, t)
    #
    # for n in [24, 50, 80]:
    #     with open(f"simulation_data/simulation_results_t_p_n{n}.json", "r") as f:
    #         collected_data = json.load(f)
    #     df_t_p = pd.DataFrame(collected_data)
    #     visualise_pk_per_t_1(df_t_p, n)
    #     visualise_pk_per_t_2(df_t_p, n)
