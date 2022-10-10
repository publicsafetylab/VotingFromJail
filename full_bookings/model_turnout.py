import sys
sys.path.append("../")

import json
import os
import pandas as pd

from utils import *


class ModelTurnoutFull:
    def __init__(self, arguments):
        self.logger = get_logger()

        # Determine input filename from arguments.
        self.input_dir = "out/balance_iteration/"
        self.path = create_combo_path(arguments)
        self.input_dir += self.path
        self.splits = list(pd.read_csv(
            self.input_dir + "/experimental_windows.csv"
        )[["control_days", "treatment_days"]].to_records(index=False))

        # Set up output directory.
        if not os.path.exists("out/modeled_turnout"):
            os.makedirs("out/modeled_turnout")
        self.output_dir = "out/modeled_turnout"

    def main(self):
        turnout_models = list()
        for split in self.splits:
            split = (int(split[0]), int(split[1]))
            to_model = pd.read_csv(self.input_dir + f"/c_{split[0]}/t_{split[1]}.csv", low_memory=False)
            to_model = set_to_datetime(to_model)
            to_model = to_model.set_index(["jail_id", "week"])

            # Set up 4 turnout modeling variations (confinement, proportion of confinement, w/ and w/o co_variates).
            fits = list()
            for design in [
                ("treatment", "no_co_variates"), ("treatment", "co_variates"),
                ("pct_votable_days_in_custody", "no_co_variates"), ("pct_votable_days_in_custody", "co_variates")
            ]:
                independent = [design[0]]
                if design[1] == "co_variates":
                    independent += full_bookings_turnout_co_variates
                fit = model(
                    to_model=to_model,
                    dependent="l2_voted_indicator",
                    independent=independent,
                    entity_fx=True,
                    time_fx=True,
                )
                fits.append({
                    "design": design,
                    "coefficient": fit.params[design[0]],
                    "p_value": fit.pvalues[design[0]],
                    "std_error": fit.std_errors[design[0]],
                })

            # Set up output for T/C split.
            turnout_models.append({
                "split": split,
                "observations": fit.nobs,
                "fits": fits,
                "mean_control_turnout": len(to_model[
                    (to_model["treatment"] == 0) & (to_model["l2_voted_indicator"] == 1)
                ]) / len(to_model[to_model["treatment"] == 0]),
                "mean_proportion_confined": to_model[to_model["treatment"] == 1]["pct_votable_days_in_custody"].mean(),
                "max_proportion_confined": to_model[to_model["treatment"] == 1]["pct_votable_days_in_custody"].max()
            })

        with open(f"{self.output_dir}/{self.path}.json", "w") as json_file:
            json.dump(turnout_models, json_file)
        self.logger.info(f"Saved turnout modeling results as: {self.output_dir}/{self.path}.json.")


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
    w = ModelTurnoutFull(args)
    w.main()
