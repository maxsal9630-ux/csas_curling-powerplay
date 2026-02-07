import pandas as pd
import numpy as np
from pathlib import Path

Path("outputs").mkdir(parents=True, exist_ok=True)
Path("figures").mkdir(parents=True, exist_ok=True)


DATA_PATH = "data/raw"


def main():
    ends = pd.read_csv(f"{DATA_PATH}/Ends.csv")
    games = pd.read_csv(f"{DATA_PATH}/Games.csv")

    for df in (ends, games):
        df["MatchID"] = (
            df["CompetitionID"].astype(str)
            + "-" + df["SessionID"].astype(str)
            + "-" + df["GameID"].astype(str)
        )

    ends["PowerPlayUsed"] = ends["PowerPlay"].notna().astype(int)

    game_teams = games[["MatchID", "TeamID1", "TeamID2"]].copy()
    end_df = ends.merge(game_teams, on="MatchID", how="left")

    end_df["OppTeamID"] = np.where(
        end_df["TeamID"] == end_df["TeamID1"],
        end_df["TeamID2"],
        np.where(
            end_df["TeamID"] == end_df["TeamID2"],
            end_df["TeamID1"],
            np.nan,
        ),
    )

    end_df["Result"] = (
        pd.to_numeric(end_df["Result"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    end_df = end_df.sort_values(
        ["MatchID", "TeamID", "EndID"]
    ).reset_index(drop=True)

    end_df["TeamScoreBeforeEnd"] = (
        end_df.groupby(["MatchID", "TeamID"])["Result"].cumsum()
        - end_df["Result"]
    )

    opp_scores = end_df[
        ["MatchID", "EndID", "TeamID", "TeamScoreBeforeEnd"]
    ].copy()
    opp_scores = opp_scores.rename(
        columns={
            "TeamID": "OppTeamID",
            "TeamScoreBeforeEnd": "OppScoreBeforeEnd",
        }
    )

    end_df = end_df.merge(
        opp_scores,
        on=["MatchID", "EndID", "OppTeamID"],
        how="left",
    )

    end_df["ScoreDiffBeforeEnd"] = (
        end_df["TeamScoreBeforeEnd"] - end_df["OppScoreBeforeEnd"]
    )

    def bucket(sd):
        if sd <= -2:
            return "Down2+"
        if sd == -1:
            return "Down1"
        if sd == 0:
            return "Tied"
        if sd == 1:
            return "Up1"
        return "Up2+"

    end_df["ScoreBucket"] = end_df["ScoreDiffBeforeEnd"].apply(bucket)

    end_df["ResultBin"] = pd.cut(
        end_df["Result"],
        bins=[-0.1, 0.5, 1.5, 2.5, 99],
        labels=["0", "1", "2", "3+"],
    )

    end_df["IsOvertime"] = (end_df["EndID"] >= 9).astype(int)

    end_df.to_csv("outputs/team_end_table.csv", index=False)

    print("Saved outputs/team_end_table.csv")
    print("Rows:", end_df.shape[0])
    print(
        end_df[
            [
                "EndID",
                "PowerPlayUsed",
                "Result",
                "TeamScoreBeforeEnd",
                "OppScoreBeforeEnd",
                "ScoreBucket",
                "IsOvertime",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()
