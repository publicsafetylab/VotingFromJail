import sys
sys.path.append("../")

import json
import os
import pandas as pd

from utils import *


class ModelTurnoutHeterogeneous:
    def __init__(self, arguments):
        self.logger = get_logger()
        self.exclude_modeled_race = arguments.exclude_modeled_race

        # Determine input filename from arguments.
        self.input_dir = "out/balance_iteration/"
        self.path = create_combo_path(arguments)
        self.input_dir += self.path
        self.splits = list(pd.read_csv(
            self.input_dir + "/experimental_windows.csv"
        )[["control_days", "treatment_days"]].to_records(index=False))

        # Set up output directory.
        if not os.path.exists("out/modeled_turnout_heterogeneous"):
            os.makedirs("out/modeled_turnout_heterogeneous")
        self.output_dir = "out/modeled_turnout_heterogeneous"
        if self.exclude_modeled_race:
            if not os.path.exists("out/modeled_turnout_heterogeneous_race_reporting"):
                os.makedirs("out/modeled_turnout_heterogeneous_race_reporting")
            self.output_dir = "out/modeled_turnout_heterogeneous_race_reporting"

    def main(self):
        turnout_models = list()
        for split in self.splits:
            split = (int(split[0]), int(split[1]))
            to_model = pd.read_csv(self.input_dir + f"/c_{split[0]}/t_{split[1]}.csv", low_memory=False)
            to_model = set_to_datetime(to_model)
            to_model = to_model.set_index(["jail_id", "week"])

            # Reduce race values to check heterogeneity.
            to_model = to_model[to_model["l2_race"].isin(["Black", "White"])]
            to_model["treatment_x_black"] = to_model["treatment"] * to_model["l2_race_Black"]
            to_model["pct_votable_days_in_custody_x_black"] = to_model["pct_votable_days_in_custody"] * to_model["l2_race_Black"]

            # If specified, subset states that report l2_race direct.
            if self.exclude_modeled_race:
                to_model = to_model[to_model["state"].isin(race_reporting_states)]

            # Set up 4 turnout modeling variations (confinement, proportion of confinement, w/ and w/o co_variates).
            fits = list()
            for design in ["treatment", "pct_votable_days_in_custody"]:
                independent = [design, "l2_race_Black", f"{design}_x_black"]
                independent += turnout_heterogeneity_co_variates
                fit = model(
                    to_model=to_model,
                    dependent="l2_voted_indicator",
                    independent=independent,
                    entity_fx=True,
                    time_fx=True,
                )
                to_json = {"design": design, "params": list()}
                for variable in [design, f"{design}_x_black"]:
                    to_json["params"].append({
                        "parameter": variable,
                        "coefficient": fit.params[variable],
                        "p_value": fit.pvalues[variable],
                        "std_error": fit.std_errors[variable],
                    })
                fits.append(to_json)

            # Set up output for T/C split.
            turnout_models.append({
                "split": split,
                "observations": fit.nobs,
                "fits": fits,
                "mean_control_turnout_black": len(to_model[
                    (to_model["treatment"] == 0) & (to_model["l2_voted_indicator"] == 1) & (to_model["l2_race_Black"] == 1)
                ]) / len(to_model[(to_model["treatment"] == 0) & (to_model["l2_race_Black"] == 1)]),
                "mean_control_turnout_white": len(to_model[
                    (to_model["treatment"] == 0) & (to_model["l2_voted_indicator"] == 1) & (to_model["l2_race_Black"] == 0)
                ]) / len(to_model[(to_model["treatment"] == 0) & (to_model["l2_race_Black"] == 0)]),
                "mean_proportion_confined_black": to_model[
                    (to_model["treatment"] == 1) & (to_model["l2_race_Black"] == 1)]
                ["pct_votable_days_in_custody"].mean(),
                "mean_proportion_confined_white": to_model[
                    (to_model["treatment"] == 1) & (to_model["l2_race_Black"] == 0)]
                ["pct_votable_days_in_custody"].mean(),
                "max_proportion_confined_black": to_model[
                    (to_model["treatment"] == 1) & (to_model["l2_race_Black"] == 1)]
                ["pct_votable_days_in_custody"].max(),
                "max_proportion_confined_white": to_model[
                    (to_model["treatment"] == 1) & (to_model["l2_race_Black"] == 0)]
                ["pct_votable_days_in_custody"].max(),
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
    parser.add_argument(
        "-xr", "--exclude_modeled_race",
        action="store_true",
        help="Only consider voters from states that report l2_race directly (i.e. it is not modeled)."
    )
    args = parser.parse_args()
    w = ModelTurnoutHeterogeneous(args)
    w.main()
