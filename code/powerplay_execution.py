import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("outputs").mkdir(parents=True, exist_ok=True)
Path("figures").mkdir(parents=True, exist_ok=True)


DATA_PATH = "data/raw"

# task labels
TASK_MAP = {
    0: "Draw",
    1: "Front",
    2: "Guard",
    3: "Raise/Tap-back",
    4: "Wick/Soft peel",
    5: "Freeze",
    6: "Take-out",
    7: "Hit & Roll",
    8: "Clearing",
    9: "Double Take-out",
    10: "Promotion Take-out",
    11: "Through",
    13: "No stats",
}


def points_bin(x):
    if pd.isna(x):
        return "Unknown"
    if x <= 1:
        return "0-1 (miss)"
    if x == 2:
        return "2 (ok)"
    if x == 3:
        return "3 (good)"
    return "4 (perfect)"


def main():
    end_df = pd.read_csv("outputs/team_end_table.csv")
    stones = pd.read_csv(f"{DATA_PATH}/Stones.csv")

    stones["MatchID"] = (
        stones["CompetitionID"].astype(str)
        + "-" + stones["SessionID"].astype(str)
        + "-" + stones["GameID"].astype(str)
    )

    pp_ends = end_df[end_df["PowerPlayUsed"] == 1].copy()
    pp_ends = pp_ends[pp_ends["EndID"].between(1, 8)].copy()

    pp_shots = stones.merge(
        pp_ends[["MatchID", "EndID", "TeamID", "Result", "ScoreBucket", "ScoreDiffBeforeEnd"]],
        on=["MatchID", "EndID", "TeamID"],
        how="inner",
    )

    pp_shots["TaskLabel"] = pp_shots["Task"].map(TASK_MAP).fillna("Other")

    pp_shots["P2plus"] = (pp_shots["Result"] >= 2).astype(int)
    pp_shots["P0"] = (pp_shots["Result"] == 0).astype(int)

    pp_shots = pp_shots.sort_values(["MatchID", "EndID", "TeamID", "ShotID"]).copy()
    pp_shots["TeamShotOrder"] = (
        pp_shots.groupby(["MatchID", "EndID", "TeamID"]).cumcount() + 1
    )

    # first 3 shots
    opening = pp_shots[pp_shots["TeamShotOrder"].isin([1, 2, 3])].copy()

    task_rates = (
        opening.groupby(["TeamShotOrder", "TaskLabel"])
        .size()
        .reset_index(name="n")
        .sort_values(["TeamShotOrder", "n"], ascending=[True, False])
    )
    task_rates["pct"] = task_rates.groupby("TeamShotOrder")["n"].transform(lambda s: s / s.sum())
    task_rates.to_csv("outputs/pp_opening_task_rates.csv", index=False)

    shot1 = opening[opening["TeamShotOrder"] == 1].copy()
    
    #end result probabilities
    shot1["P0"] = (shot1["Result"] == 0).astype(int)
    shot1["P1"] = (shot1["Result"] == 1).astype(int)
    shot1["P2plus"] = (shot1["Result"] >= 2).astype(int)
    shot1["P3plus"] = (shot1["Result"] >= 3).astype(int)

    shot_probs = (
        shot1.groupby("TaskLabel")
        .agg(
            n=("Result", "size"),
            mean_points=("Result", "mean"),
            p0=("P0", "mean"),
            p1=("P1", "mean"),
            p2plus=("P2plus", "mean"),
            p3plus=("P3plus", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )

    # optional: only keep tasks with enough data
    shot_probs.to_csv("outputs/pp_shot1_task_probs.csv", index=False)

    shot1_summary = (
        shot1.groupby("TaskLabel")
        .agg(
            n=("Result", "size"),
            mean_points=("Result", "mean"),
            p2plus=("P2plus", "mean"),
            p0=("P0", "mean"),
            mean_shot_points=("Points", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    shot1_summary.to_csv("outputs/pp_opening_task_outcomes.csv", index=False)

    shot1["Shot1PointsBin"] = shot1["Points"].apply(points_bin)

    points_quality = (
        shot1.groupby("Shot1PointsBin")
        .agg(
            n=("Result", "size"),
            mean_points=("Result", "mean"),
            p2plus=("P2plus", "mean"),
            p0=("P0", "mean"),
        )
        .reset_index()
    )

    order = ["0-1 (miss)", "2 (ok)", "3 (good)", "4 (perfect)", "Unknown"]
    points_quality["Shot1PointsBin"] = pd.Categorical(
        points_quality["Shot1PointsBin"],
        categories=order,
        ordered=True
    )
    points_quality = points_quality.sort_values("Shot1PointsBin")
    points_quality.to_csv("outputs/pp_points_quality_outcomes.csv", index=False)

    plot_df = shot1_summary[shot1_summary["n"] >= 10].copy()

    plt.figure()
    plt.title("Power Play: Mean End Points by PP Team Shot 1 Task (n>=10)")
    plt.xlabel("PP Team Shot 1 Task")
    plt.ylabel("Mean points scored in end")
    plt.bar(plot_df["TaskLabel"], plot_df["mean_points"])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("figures/pp_shot1_task_mean_points.png", dpi=200)
    plt.close()

    plt.figure()
    plt.title("Power Play: P(2+) by PP Team Shot 1 Task (n>=10)")
    plt.xlabel("PP Team Shot 1 Task")
    plt.ylabel("Probability of scoring 2+")
    plt.bar(plot_df["TaskLabel"], plot_df["p2plus"])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("figures/pp_shot1_task_p2plus.png", dpi=200)
    plt.close()

    print("Saved outputs/pp_opening_task_rates.csv")
    print("Saved outputs/pp_opening_task_outcomes.csv")
    print("Saved outputs/pp_points_quality_outcomes.csv")
    print("Saved figures/pp_shot1_task_mean_points.png")
    print("Saved figures/pp_shot1_task_p2plus.png")
    print()
    print("Shot1 summary (top 10 by frequency):")
    print(shot1_summary.head(10).to_string(index=False))
    print()
    print("Shot1 execution quality summary:")
    print(points_quality.to_string(index=False))


if __name__ == "__main__":
    main()
