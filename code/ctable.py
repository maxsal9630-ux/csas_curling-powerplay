import pandas as pd
from pathlib import Path

Path("outputs").mkdir(parents=True, exist_ok=True)
Path("figures").mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv("outputs/team_end_table.csv")

    df = df[df["EndID"].between(1, 8)].copy()

    df["P2plus"] = (df["Result"] >= 2).astype(int)

    # usage
    usage = (
        df.groupby(["EndID", "ScoreBucket"])
        .agg(
            pp_rate=("PowerPlayUsed", "mean"),
            n_total=("Result", "size"),
        )
        .reset_index()
    )

    # outcomes
    perf = (
        df.groupby(["EndID", "ScoreBucket", "PowerPlayUsed"])
        .agg(
            n=("Result", "size"),
            mean_points=("Result", "mean"),
            p2plus=("P2plus", "mean"),
        )
        .reset_index()
    )

    pp = perf[perf["PowerPlayUsed"] == 1].copy()
    no = perf[perf["PowerPlayUsed"] == 0].copy()

    merged = pp.merge(
        no,
        on=["EndID", "ScoreBucket"],
        how="left",
        suffixes=("_pp", "_no"),
    )

    merged["lift_points"] = merged["mean_points_pp"] - merged["mean_points_no"]
    merged["lift_p2plus"] = merged["p2plus_pp"] - merged["p2plus_no"]

    out = merged.merge(
        usage,
        on=["EndID", "ScoreBucket"],
        how="left",
    )

    def confidence(row):
        npp = row["n_pp"]
        nno = row["n_no"]
        if pd.isna(npp) or pd.isna(nno):
            return "Low"
        if npp >= 30 and nno >= 30:
            return "High"
        if npp >= 10 and nno >= 10:
            return "Medium"
        return "Low"

    out["confidence"] = out.apply(confidence, axis=1)

    out["recommend_pp"] = (
        (out["lift_points"] >= 0.5)
        & (out["lift_p2plus"] >= 0.15)
        & (out["confidence"].isin(["High", "Medium"]))
    ).astype(int)

    cols = [
        "EndID",
        "ScoreBucket",
        "pp_rate",
        "n_total",
        "n_pp",
        "mean_points_pp",
        "p2plus_pp",
        "n_no",
        "mean_points_no",
        "p2plus_no",
        "lift_points",
        "lift_p2plus",
        "confidence",
        "recommend_pp",
    ]

    out = out[cols].sort_values(["EndID", "ScoreBucket"])
    out.to_csv("outputs/decision_table.csv", index=False)

    print("Saved outputs/decision_table.csv")
    print()
    print("Top recommended situations:")
    top = (
        out[out["recommend_pp"] == 1]
        .sort_values("lift_points", ascending=False)
        .head(12)
    )
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
