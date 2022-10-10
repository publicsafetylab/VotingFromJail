import sys
sys.path.append("../")

import json
import os

from utils import *


class ModelBalanceFull:
    def __init__(self, arguments):
        self.logger = get_logger()
        self.election_day = election_day

        # Determine input filename from arguments.
        self.input_dir = "out/balance_iteration/"
        self.path = create_combo_path(arguments)
        self.input_filename = self.input_dir + self.path + "/full_splits.csv"

        # Set up output directory.
        if not os.path.exists("out/modeled_balance"):
            os.makedirs("out/modeled_balance")
        self.output_dir = "out/modeled_balance"

    def main(self):
        # Get the earliest date by control window.
        splits_df = pd.read_csv(self.input_filename, low_memory=False)
        controls = splits_df.groupby("control_days")["earliest_date"].min().reset_index()

        # Find the earliest date after which all p-values are <= 0.1 by control window.
        imbalanced = splits_df[splits_df["p_value"] <= 0.1]
        imbalanced = imbalanced.groupby("control_days")["earliest_date"].max().reset_index()
        imbalanced["earliest_date"] = pd.to_datetime(imbalanced["earliest_date"])
        imbalanced["earliest_date"] = imbalanced["earliest_date"].apply(lambda d: d + dt.timedelta(days=1))
        imbalanced = imbalanced.rename(columns={"earliest_date": "earliest_viable_date"})
        treatment_ranges = pd.merge(controls, imbalanced, how="left", on="control_days")
        treatment_ranges["earliest_viable_date"] = np.where(
            treatment_ranges["earliest_viable_date"].isna(),
            treatment_ranges["earliest_date"],
            treatment_ranges["earliest_viable_date"]
        )
        treatment_ranges = treatment_ranges.drop(columns=["earliest_date"])
        treatment_ranges["earliest_viable_date"] = pd.to_datetime(treatment_ranges["earliest_viable_date"])
        treatment_ranges["treatment_days"] = (self.election_day - treatment_ranges["earliest_viable_date"]).dt.days
        treatment_ranges = treatment_ranges[treatment_ranges["treatment_days"] >= 7]
        treatment_ranges.to_csv(self.input_dir + self.path + "/experimental_windows.csv", index=False)
        self.logger.info(f"Saved experimental windows as: {self.input_dir + self.path}/experimental_windows.csv.")

        # Run balance checks for only relevant window pairs.
        balance_models = list()
        for split in list(treatment_ranges[["control_days", "treatment_days"]].to_records(index=False)):
            split = (int(split[0]), int(split[1]))
            to_model = pd.read_csv(self.input_dir + self.path + f"/c_{split[0]}/t_{split[1]}.csv", low_memory=False)
            to_model = set_to_datetime(to_model)
            to_model = to_model.set_index(["jail_id", "week"])
            fit = model(
                to_model=to_model,
                dependent="treatment",
                independent=full_bookings_balance_co_variates,
                entity_fx=True,
                time_fx=False,
            )

            # Save fit statistics for this split.
            coefficients = pd.Series({param_map[key]: value for key, value in fit.params.to_dict().items()})
            std_errors = pd.Series({param_map[key]: value for key, value in fit.std_errors.to_dict().items()})
            p_values = pd.Series({param_map[key]: value for key, value in fit.pvalues.to_dict().items()})
            params = pd.merge(coefficients.reset_index(), std_errors.reset_index(), on="index")
            params = pd.merge(params, p_values.reset_index(), on="index").rename(columns={
                "index": "parameter", "0_x": "coefficient", "0_y": "std_error", 0: "p_value"
            }).sort_values(by="parameter").to_dict("records")
            balance_models.append({
                "split": split,
                "observations": fit.nobs,
                "p_value": fit.f_statistic.pval,
                "params": params
            })
        with open(f"{self.output_dir}/{self.path}.json", "w") as json_file:
            json.dump(balance_models, json_file)
        self.logger.info(f"Saved balancing results as: {self.output_dir}/{self.path}.json.")


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
    w = ModelBalanceFull(args)
    w.main()
