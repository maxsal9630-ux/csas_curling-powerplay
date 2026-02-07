import pandas as pd

DATA_PATH = "data/raw"

def main():
    df = pd.read_csv("outputs/team_end_table.csv")
    teams = pd.read_csv(f"{DATA_PATH}/Teams.csv")

    df = df[df["EndID"].between(1, 8)].copy()

    team_noc = (
        teams.groupby("TeamID")["NOC"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
    )

    df = df.merge(team_noc, on="TeamID", how="left")

    df["P2plus"] = (df["Result"] >= 2).astype(int)
    df["P3plus"] = (df["Result"] >= 3).astype(int)

    # overall PP usage by team
    usage = (
        df.groupby("NOC")
        .agg(
            ends=("Result", "size"),
            pp_used=("PowerPlayUsed", "sum"),
            pp_rate=("PowerPlayUsed", "mean"),
        )
        .reset_index()
    )

    # PP vs non-PP
    perf = (
        df.groupby(["NOC", "PowerPlayUsed"])
        .agg(
            n=("Result", "size"),
            mean_points=("Result", "mean"),
            p2plus=("P2plus", "mean"),
            p3plus=("P3plus", "mean"),
        )
        .reset_index()
    )

    pp = perf[perf["PowerPlayUsed"] == 1].copy()
    no = perf[perf["PowerPlayUsed"] == 0].copy()

    merged = pp.merge(no, on="NOC", how="left", suffixes=("_pp", "_no"))

    merged["lift_points"] = merged["mean_points_pp"] - merged["mean_points_no"]
    merged["lift_p2plus"] = merged["p2plus_pp"] - merged["p2plus_no"]

    out = merged.merge(usage, on="NOC", how="left")

    cols = [
        "NOC",
        "ends",
        "pp_rate",
        "n_pp",
        "mean_points_pp",
        "p2plus_pp",
        "n_no",
        "mean_points_no",
        "p2plus_no",
        "lift_points",
        "lift_p2plus",
    ]
    out = out[cols].sort_values("lift_points", ascending=False)

    out.to_csv("outputs/team_benchmark.csv", index=False)

    print("Saved outputs/team_benchmark.csv")
    print(out.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
