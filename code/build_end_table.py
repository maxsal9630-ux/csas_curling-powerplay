import pandas as pd
import numpy as np

DATA_PATH = "data/raw"

def main():
    ends = pd.read_csv(f"{DATA_PATH}/Ends.csv")
    games = pd.read_csv(f"{DATA_PATH}/Games.csv")

    # Make MatchID
    for df in (ends, games):
        df["MatchID"] = (
            df["CompetitionID"].astype(str)
            + "-" + df["SessionID"].astype(str)
            + "-" + df["GameID"].astype(str)
        )

    # Power play flag
    ends["PowerPlayUsed"] = ends["PowerPlay"].notna().astype(int)

    # Join team identities (TeamID1/TeamID2) into end rows
    game_teams = games[["MatchID", "TeamID1", "TeamID2"]].copy()
    end_df = ends.merge(game_teams, on="MatchID", how="left")

    # Identify opponent TeamID for each row
    end_df["OppTeamID"] = np.where(
        end_df["TeamID"] == end_df["TeamID1"], end_df["TeamID2"],
        np.where(end_df["TeamID"] == end_df["TeamID2"], end_df["TeamID1"], np.nan)
    )

    # Clean points scored in the end
    end_df["Result"] = pd.to_numeric(end_df["Result"], errors="coerce").fillna(0).astype(int)

    # Sort for cumulative scoring
    end_df = end_df.sort_values(["MatchID", "TeamID", "EndID"]).reset_index(drop=True)

    # Score before the end = cumulative sum of prior ends
    end_df["TeamScoreBeforeEnd"] = (
        end_df.groupby(["MatchID", "TeamID"])["Result"].cumsum() - end_df["Result"]
    )

    # Opponent score before end: merge back on MatchID+EndID using OppTeamID
    opp_scores = end_df[["MatchID", "EndID", "TeamID", "TeamScoreBeforeEnd"]].copy()
    opp_scores = opp_scores.rename(columns={
        "TeamID": "OppTeamID",
        "TeamScoreBeforeEnd": "OppScoreBeforeEnd"
    })

    end_df = end_df.merge(opp_scores, on=["MatchID", "EndID", "OppTeamID"], how="left")

    # Score diff before end
    end_df["ScoreDiffBeforeEnd"] = end_df["TeamScoreBeforeEnd"] - end_df["OppScoreBeforeEnd"]

    # Buckets
    def bucket(sd):
        if sd <= -2: return "Down2+"
        if sd == -1: return "Down1"
        if sd == 0:  return "Tied"
        if sd == 1:  return "Up1"
        return "Up2+"

    end_df["ScoreBucket"] = end_df["ScoreDiffBeforeEnd"].apply(bucket)

    # Result bins (for later)
    end_df["ResultBin"] = pd.cut(
        end_df["Result"],
        bins=[-0.1, 0.5, 1.5, 2.5, 99],
        labels=["0", "1", "2", "3+"]
    )

    # Helpful flags
    end_df["IsOvertime"] = (end_df["EndID"] >= 9).astype(int)

    # Save output
    end_df.to_csv("outputs/team_end_table.csv", index=False)

    print("Saved outputs/team_end_table.csv")
    print("Rows:", end_df.shape[0])
    print(end_df[["EndID","PowerPlayUsed","Result","TeamScoreBeforeEnd","OppScoreBeforeEnd","ScoreBucket","IsOvertime"]].head(10))

if __name__ == "__main__":
    main()
