import sys
sys.path.append("../")

import json
import os
import pandas as pd

from utils import *


class ModelTurnoutPlacebo:
    def __init__(self, arguments):
        self.logger = get_logger()
        self.previous_elections = {"2016": dt.datetime(2016, 11, 8, 0, 0), "2012": dt.datetime(2012, 11, 6, 0, 0)}

        # Determine input filename from arguments.
        self.input_dir = "out/balance_iteration/"
        self.path = create_combo_path(arguments)
        self.input_dir += self.path
        self.splits = list(pd.read_csv(
            self.input_dir + "/experimental_windows.csv"
        )[["control_days", "treatment_days"]].to_records(index=False))

        # Set up output directory.
        if not os.path.exists("out/modeled_turnout_placebo"):
            os.makedirs("out/modeled_turnout_placebo")
        self.output_dir = "out/modeled_turnout_placebo"

    def main(self):
        placebo_models = list()
        for split in self.splits:
            split = (int(split[0]), int(split[1]))
            to_model = pd.read_csv(self.input_dir + f"/c_{split[0]}/t_{split[1]}.csv", low_memory=False)
            to_model = set_to_datetime(to_model)
            to_model = to_model.set_index(["jail_id", "week"])

            # Set up 4 modeling variations (treatment on 2012 and 2016 turnout, and 2020 turnout for those subsets).
            years = list()
            for year in self.previous_elections.keys():
                subset_to_model = to_model[to_model["l2_date_registered_calculated"] <= self.previous_elections[year]]
                on_self = model(
                    to_model=subset_to_model,
                    dependent=f"l2_voted_indicator_{year}",
                    independent=["treatment"] + turnout_co_variates,
                    entity_fx=True,
                    time_fx=False,
                )
                on_2020 = model(
                    to_model=subset_to_model,
                    dependent=f"l2_voted_indicator",
                    independent=["treatment"] + turnout_co_variates,
                    entity_fx=True,
                    time_fx=False,
                )
                if on_self.nobs != on_2020.nobs:
                    raise ValueError(f"Observations modeled unequal for {year} != 2020.")
                years.append({
                    "years": (year, year),
                    "coefficient": on_self.params["treatment"],
                    "p_value": on_self.pvalues["treatment"],
                    "std_error": on_self.std_errors["treatment"],
                    "mean_control_turnout": len(subset_to_model[
                        (subset_to_model["treatment"] == 0) & (subset_to_model[f"l2_voted_indicator_{year}"] == 1)
                    ]) / len(subset_to_model[subset_to_model["treatment"] == 0]),
                    "observations": on_self.nobs,
                })
                years.append({
                    "years": (year, "2020"),
                    "coefficient": on_2020.params["treatment"],
                    "p_value": on_2020.pvalues["treatment"],
                    "std_error": on_2020.std_errors["treatment"],
                    "mean_control_turnout": len(subset_to_model[
                        (subset_to_model["treatment"] == 0) & (subset_to_model["l2_voted_indicator"] == 1)
                    ]) / len(subset_to_model[subset_to_model["treatment"] == 0]),
                    "observations": on_2020.nobs,
                })

            # Set up output for T/C split.
            placebo_models.append({
                "split": split,
                "years": years
            })

        with open(f"{self.output_dir}/{self.path}.json", "w") as json_file:
            json.dump(placebo_models, json_file)
        self.logger.info(f"Saved placebo modeling results as: {self.output_dir}/{self.path}.json.")


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
    w = ModelTurnoutPlacebo(args)
    w.main()
