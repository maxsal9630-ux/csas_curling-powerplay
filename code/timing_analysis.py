import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("outputs/team_end_table.csv")

    # Keep regulation ends only (PP not allowed in extra end)
    df = df[df["EndID"].between(1, 8)].copy()

    df["P2plus"] = (df["Result"] >= 2).astype(int)
    df["P3plus"] = (df["Result"] >= 3).astype(int)

    # Context: PP usage rate by situation
    usage = (
        df.groupby(["EndID", "ScoreBucket"])["PowerPlayUsed"]
        .agg(pp_rate="mean", n="size")
        .reset_index()
    )
    usage.to_csv("outputs/pp_usage_rate_by_state.csv", index=False)

    # Performance summary (PP vs no-PP)
    summary = (
        df.groupby(["EndID", "ScoreBucket", "PowerPlayUsed"])
        .agg(
            n=("Result", "size"),
            mean_points=("Result", "mean"),
            p2plus=("P2plus", "mean"),
            p3plus=("P3plus", "mean"),
        )
        .reset_index()
    )
    summary.to_csv("outputs/timing_summary.csv", index=False)

    # Lift table: PP minus no-PP
    pp = summary[summary["PowerPlayUsed"] == 1].copy()
    no = summary[summary["PowerPlayUsed"] == 0].copy()

    merged = pp.merge(no, on=["EndID", "ScoreBucket"], how="left", suffixes=("_pp", "_no"))
    merged["lift_points"] = merged["mean_points_pp"] - merged["mean_points_no"]
    merged["lift_p2plus"] = merged["p2plus_pp"] - merged["p2plus_no"]

    merged = merged.sort_values(["ScoreBucket", "EndID"])
    merged.to_csv("outputs/timing_lift.csv", index=False)

    # One simple plot: lift in tied ends (filter tiny samples)
    plot_bucket = "Tied"
    plot_df = merged[merged["ScoreBucket"] == plot_bucket].copy()
    plot_df = plot_df[(plot_df["n_pp"] >= 10) & (plot_df["n_no"] >= 10)]

    plt.figure()
    plt.title("Power Play Lift (Mean Points) when Tied")
    plt.xlabel("End")
    plt.ylabel("Lift = PP mean - No-PP mean")
    plt.axhline(0)

    plt.bar(plot_df["EndID"].astype(int), plot_df["lift_points"].astype(float))
    plt.xticks(sorted(plot_df["EndID"].unique()))

    plt.tight_layout()
    plt.savefig("figures/lift_points_tied.png", dpi=200)
    plt.close()

    print("Saved outputs/pp_usage_rate_by_state.csv")
    print("Saved outputs/timing_summary.csv")
    print("Saved outputs/timing_lift.csv")
    print("Saved figures/lift_points_tied.png")

if __name__ == "__main__":
    main()
