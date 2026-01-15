import pandas as pd 

DATA_PATH = "data/raw"

def load_csv(filename):
    path = f"{DATA_PATH}/{filename}.csv"
    df = pd.read_csv(path)
    print(f"{filename}: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("-" * 50)
    return df

def main():
    competition = load_csv("Competition")
    competitors = load_csv("Competitors")
    teams = load_csv("Teams")
    games = load_csv("Games")
    ends = load_csv("Ends")
    stones = load_csv("Stones")

    # quick checks to understand the data
    print("Number of competitions:", competition["CompetitionID"].nunique())
    print(
        "Number of games:",
        games[["CompetitionID", "SessionID", "GameID"]].drop_duplicates().shape[0]
    )

    print("Ends per game (sample):")
    print(
        ends.groupby(["CompetitionID", "SessionID", "GameID"])["EndID"]
        .nunique()
        .value_counts()
        .head()
    )

if __name__ == "__main__":
    main()