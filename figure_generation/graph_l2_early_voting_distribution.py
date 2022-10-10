import sys
sys.path.append("../")

import datetime as dt
import os
import pandas as pd
import plotly.express as px

from dotenv import load_dotenv
from pymongo import MongoClient

from utils import *

load_dotenv()


class GraphEarlyVoterDist:
    def __init__(self, arguments):
        self.logger = get_logger()
        self.db = MongoClient(
            os.getenv("LAKE_CLIENT_URI")
        ).get_database(
            os.getenv("LAKE_DB")
        ).get_collection(
            os.getenv("LAKE_COLLECTION")
        )
        self.full = arguments.full
        self.input_dir_base = "matched_bookings"
        if self.full:
            self.input_dir_base = "full_bookings"

        # Determine input filename from arguments.
        self.path = create_combo_path(arguments)
        self.v_line_date = dt.datetime.strptime(pd.read_csv(f"../{self.input_dir_base}/out/balance_iteration/{self.path}/experimental_windows.csv")["earliest_viable_date"].max(), "%Y-%m-%d")
        if self.full:
            self.input_dir_base = "full_bookings"

        # Specify output filename.
        self.output_html_filename = f"../{self.input_dir_base}/out/figures/{self.path}/l2_early_voting_by_state.html"

    def main(self):
        self.logger.info("Collecting L2 early voting data by state...")
        dfs = thread(self.get_early_voting_by_state, states)
        df = pd.concat(dfs)

        # Remove outliers votes (outside of legal voting days).
        df = set_to_datetime(df)
        df = df[(df["date"] >= earliest_voting_date) & (df["date"] <= election_day)]

        # Sum across states.
        df = df.groupby("date")["votes"].sum().reset_index().sort_values(by="date")

        # Generate graph.
        fig = px.ecdf(df, x="date", y="votes", labels={"date": "Date"})
        fig = fig.update_layout(yaxis_title="Proportion of Votes Cast")
        fig = fig.add_vline(x=self.v_line_date, line_dash="dot")
        fig = fig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)"})
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
        fig.write_html(self.output_html_filename)
        self.logger.info(f"Saved as: {self.output_html_filename}.")

    def get_early_voting_by_state(self, state):
        total_voters = self.db.count_documents({
            "state": state,
            "filename": "VOTEHISTORY"
        })

        voters_without_return_date = self.db.count_documents({
            "state": state,
            "filename": "VOTEHISTORY",
            "BallotReturnDate_General_2020_11_03": {"$exists": False},
        })

        votes_per_day = list(self.db.aggregate([
            {"$match": {
                "state": state,
                "filename": "VOTEHISTORY",
                "BallotReturnDate_General_2020_11_03": {"$exists": True},
            }},
            {"$group": {
                "_id": "$BallotReturnDate_General_2020_11_03",
                "count": {"$sum": 1}
            }}
        ]))

        if len(votes_per_day) < 1:
            return

        for index, record in enumerate(votes_per_day):
            votes_per_day[index] = {
                "state": state,
                "total_voters": total_voters,
                "date": record["_id"],
                "votes": record["count"]
            }

        df = pd.DataFrame(votes_per_day)
        if df["votes"].sum() != total_voters - voters_without_return_date:
            raise ValueError(f"Issue with vote tally in {state}.")
        return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--active",
        action="store_true",
        help="Only consider voters demarcated as Active by L2."
    )
    parser.add_argument(
        "-c", "--column",
        choices=["score_weighted", "score_unweighted"],
        required=True,
        help="Match probability column on which to threshold data (choose from [score_weighted, score_unweighted])."
    )
    parser.add_argument(
        "-r", "--registered",
        action="store_true",
        help="Only consider voters registered prior to Election Day, 2020."
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.75,
        help="Threshold above which to consider matched records as matches."
    )
    parser.add_argument(
        "-xb", "--exclude_no_bond",
        action="store_true",
        help="Only consider voters from jails that report bond amounts."
    )
    parser.add_argument(
        "-xc", "--exclude_no_charge",
        action="store_true",
        help="Only consider voters from jails that report charges."
    )
    parser.add_argument(
        "-f", "--full",
        action="store_true",
        help="Use data for full bookings process (including non-L2 matches)."
    )
    args = parser.parse_args()
    w = GraphEarlyVoterDist(args)
    w.main()
