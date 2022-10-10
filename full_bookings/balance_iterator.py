import sys
sys.path.append("../")

import os

from itertools import product

from utils import *


class BalanceProcessFull:
    def __init__(self, arguments):
        self.logger = get_logger()
        self.no_charge = arguments.exclude_no_charge
        self.no_bond = arguments.exclude_no_bond
        self.election_day = election_day
        self.earliest_voting_date = earliest_voting_date

        # Determine input filename from arguments.
        self.input_dir = "out/prepped_data/"
        self.path = create_combo_path(arguments)
        self.input_filename = self.input_dir + self.path + "/merged.csv"

        # Read in data.
        self.base_df = pd.read_csv(self.input_filename, low_memory=False)

        # Set up output filename.
        if not os.path.exists("out/balance_iteration"):
            os.makedirs("out/balance_iteration")
        if not os.path.exists(f"out/balance_iteration/{self.path}"):
            os.makedirs(f"out/balance_iteration/{self.path}")
        self.output_dir = f"out/balance_iteration/{self.path}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def main(self):
        self.logger.info(f"Records read: {len(self.base_df)}.")
        self.base_df = set_to_datetime(self.base_df)
        self.logger.info("Processing balance splits...")

        # Run through combinations of control windows and treatment rollback days to model balance.
        balance_checks = thread(self.balance_one_window, list(product(control_windows, range(0, 54))))
        balance_checks = list(element for sub_list in balance_checks for element in sub_list)
        out = pd.DataFrame(balance_checks).sort_values(by=["control_days", "earliest_date"])
        out.to_csv(self.output_dir + "/full_splits.csv", index=False)
        self.logger.info(f"Saved balance results as: {self.output_dir + '/full_splits.csv'}.")

    def balance_one_window(self, split):
        max_voting_window = (self.election_day - self.earliest_voting_date).days

        # Split data into treatment and control windows.
        to_model = treatment_control_split_full_bookings(
            base_df=self.base_df,
            control=split[0],
            treatment_rollback=split[1],
            no_charge=self.no_charge,
            no_bond=self.no_bond,
        )

        # Fit model to prepped data.
        res = model(
            to_model=to_model,
            dependent="treatment",
            independent=full_bookings_balance_co_variates,
            entity_fx=True,
            time_fx=False,
        )
        
        # Output prepped modeling data to CSV.
        if not os.path.exists(self.output_dir + f"/c_{split[0]}"):
            os.makedirs(self.output_dir + f"/c_{split[0]}")
        to_model = to_model.reset_index()
        to_model.to_csv(self.output_dir + f"/c_{split[0]}/t_{max_voting_window - split[1]}.csv", index=False)

        # Return relevant statistics for p-value/balance checking.
        return [{
            "control_days": split[0],
            "rollback_days": split[1],
            "earliest_date": self.earliest_voting_date + dt.timedelta(days=split[1]),
            "f_statistic": res.f_statistic.stat,
            "p_value": res.f_statistic.pval,
        }]


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
    args = parser.parse_args()
    w = BalanceProcessFull(args)
    w.main()
