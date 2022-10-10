import sys
sys.path.append("../")

import datetime as dt
import os
import pandas as pd
import plotly.express as px

from utils import create_combo_path, get_logger


class GraphBalancePValues:
    def __init__(self, arguments):
        self.logger = get_logger()
        self.full = arguments.full
        self.input_dir_base = "matched_bookings"
        if self.full:
            self.input_dir_base = "full_bookings"

        # Determine input filename from arguments.
        self.input_dir = f"{self.input_dir_base}/out/balance_iteration/"
        self.path = create_combo_path(arguments)
        self.input_filename = self.input_dir + self.path + "/full_splits.csv"

        # Set up output filename.
        if not os.path.exists(f"../{self.input_dir_base}/out/figures"):
            os.makedirs(f"../{self.input_dir_base}/out/figures")
        if not os.path.exists(f"../{self.input_dir_base}/out/figures/{self.path}"):
            os.makedirs(f"../{self.input_dir_base}/out/figures/{self.path}")
        self.output_dir = f"../{self.input_dir_base}/out/figures/{self.path}"

    def main(self):
        self.logger.info("Reading in balance iteration splits...")
        df = pd.read_csv("../" + self.input_filename)
        df = df.rename(columns={
            "control_days": "Control Window",
            "earliest_date": "Earliest 2020 Voting Date",
            "p_value": "Joint F-Test p-value"
        })
        df["Control Window"] = df["Control Window"].apply(lambda i: str(i) + " days")
        df["Joint F-Test p-value"] = df["Joint F-Test p-value"].astype(float)
        fig = px.line(df, x="Earliest 2020 Voting Date", y="Joint F-Test p-value", color="Control Window")
        fig.add_hline(y=0.1, line_dash="dot")
        fig.add_hline(y=0)
        fig.add_vline(x=dt.datetime(2020, 9, 4, 0, 0))
        fig.update_layout({
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        })
        fig.write_html(f"{self.output_dir}/balance_check_p_values.html")
        self.logger.info(f"Saved figure as: {self.output_dir}/balance_check_p_values.html.")


if __name__ == "__main__":
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
    w = GraphBalancePValues(args)
    w.main()
