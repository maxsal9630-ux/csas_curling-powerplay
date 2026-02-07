import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("outputs").mkdir(parents=True, exist_ok=True)
Path("figures").mkdir(parents=True, exist_ok=True)


DATA_PATH = "data/raw"

def main():
    df = pd.read_csv("outputs/team_end_table.csv")

    df = df[df["EndID"].between(1, 8)].copy()

    teams = pd.read_csv(f"{DATA_PATH}/Teams.csv")
    teams = teams[["CompetitionID", "TeamID", "NOC"]].drop_duplicates()

    df = df.merge(teams, on=["CompetitionID", "TeamID"], how="left")

    # performance (PP vs no PP)
    g = (
        df.groupby(["NOC", "PowerPlayUsed"])
        .agg(
            n=("Result", "size"),
            mean_points=("Result", "mean"),
            p2plus=("Result", lambda s: (s >= 2).mean())
        )
        .reset_index()
    )

    pp = g[g["PowerPlayUsed"] == 1].copy()
    no = g[g["PowerPlayUsed"] == 0].copy()

    out = pp.merge(no, on="NOC", how="left", suffixes=("_pp", "_no"))

    out["lift_points"] = out["mean_points_pp"] - out["mean_points_no"]
    out["lift_p2plus"] = out["p2plus_pp"] - out["p2plus_no"]

    out = out[(out["n_pp"] >= 10) & (out["n_no"] >= 30)].copy()

    out = out.sort_values("lift_points", ascending=False)

    out.to_csv("outputs/team_benchmark.csv", index=False)
    print("Saved outputs/team_benchmark.csv")

    # t12
    top = out.head(12).copy()

    plt.figure()
    plt.title("Power Play Boost by Team (Mean Points Lift)")
    plt.xlabel("Team (NOC)")
    plt.ylabel("Lift = PP mean points - non-PP mean points")

    plt.axhline(0)
    plt.bar(top["NOC"], top["lift_points"])
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("figures/team_benchmark_lift_points.png", dpi=200)
    plt.close()

    print("Saved figures/team_benchmark_lift_points.png")

if __name__ == "__main__":
    main()
