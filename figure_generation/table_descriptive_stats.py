import sys
sys.path.append("../")

import json
import os
import pandas as pd
import re

from functools import reduce

from utils import *


class TableDescriptive:
    def __init__(self, arguments):
        self.logger = get_logger()
        self.full = arguments.full
        self.input_dir_base = "matched_bookings"
        if self.full:
            self.input_dir_base = "full_bookings"
        self.election_day = election_day

        # Determine input filename from arguments.
        self.input_dir = f"../{self.input_dir_base}/out/balance_iteration/"
        self.path = create_combo_path(arguments)
        self.input_dir += self.path
        self.splits = list(pd.read_csv(
            self.input_dir + "/experimental_windows.csv"
        )[["control_days", "treatment_days"]].to_records(index=False))

        # Set up output filename.
        if not os.path.exists(f"../{self.input_dir_base}/out/figures"):
            os.makedirs(f"../{self.input_dir_base}/out/figures")
        if not os.path.exists(f"../{self.input_dir_base}/out/figures/{self.path}"):
            os.makedirs(f"../{self.input_dir_base}/out/figures/{self.path}")
        self.output_dir = f"../{self.input_dir_base}/out/figures/{self.path}"

        # Specify columns by value types (e.g., indicator, continuous, primary variables, etc.).
        self.dummies = [
            "l2_gender_M",
            "l2_race_White",
            "l2_race_Black",
            "l2_party_Republican",
            "l2_party_Democratic",
            "l2_party_Non_Partisan_or_Other",
            "jdi_charge_types_violent",
            "jdi_charge_types_public_order",
            "jdi_charge_types_property",
            "jdi_charge_types_dui",
            "jdi_charge_types_drug",
            "jdi_charge_types_criminal_traffic",
        ]
        if self.full:
            self.dummies = [
                "jdi_gender_M",
                "jdi_race_White",
                "jdi_race_Black",
                "jdi_charge_types_violent",
                "jdi_charge_types_public_order",
                "jdi_charge_types_property",
                "jdi_charge_types_dui",
                "jdi_charge_types_drug",
                "jdi_charge_types_criminal_traffic",
                "matched",
                "matched_registered"
            ]
        self.continuous = ["l2_age", "jdi_length_of_stay", "jdi_num_charges"]
        if self.full:
            self.continuous = ["jdi_age", "jdi_length_of_stay", "jdi_num_charges"]
        self.primary = ["treatment", "pct_votable_days_in_custody", "l2_voted_indicator"]

    def main(self):
        dfs = list()
        for split in self.splits:
            split = (int(split[0]), int(split[1]))
            to_model = pd.read_csv(self.input_dir + f"/c_{split[0]}/t_{split[1]}.csv", low_memory=False)
            to_model = set_to_datetime(to_model)

            if self.full:
                to_model["matched_registered"] = np.where(
                    ((to_model["matched"] == 1) & (to_model["l2_date_registered_calculated"] <= self.election_day)), 1, 0
                )

            rows = list()
            for column in self.dummies + self.continuous + self.primary:
                rows.append({
                    "\\textbf{Days:}": param_map[column],
                    f"{split[1]}d_{split[0]}": format(to_model[to_model.treatment == 1][column].mean(), rounding),
                    f"{split[0]}d": format(to_model[to_model.treatment == 0][column].mean(), rounding),
                })
            rows.append({
                "\\textbf{Days:}": "Observations",
                f"{split[1]}d_{split[0]}": len(to_model[to_model.treatment == 1]),
                f"{split[0]}d": len(to_model[to_model.treatment == 0])
            })
            dfs.append(pd.DataFrame(rows))

        # Combine all T/C splits into single DataFrame.
        df = reduce(lambda df1, df2: pd.merge(df1, df2, on=["\\textbf{Days:}"]), dfs)
        df.columns = [re.sub(r"_\d+", r"", column) for column in df.columns]
        df.sort_values(by="\\textbf{Days:}", key=lambda col: col.map(lambda e: table_sorter.index(e)), inplace=True)

        # Write LaTeX.
        latex = " & ".join(df.columns)
        latex += " \\\\\n\\midrule\n"
        for i in range(len(self.dummies + self.continuous)):
            latex += " & ".join(df.iloc[i].values)
            latex += " \\\\\n"
        latex += "\\midrule\n"
        for i in range(len(self.dummies + self.continuous), len(self.dummies + self.continuous) + 2):
            latex += " & ".join(df.iloc[i].values)
            latex += " \\\\\n"
        latex += "\\midrule\n"
        latex += " & ".join(df.iloc[-2].values)
        latex += " \\\\\n\\midrule\n"
        latex += " & ".join([str(i) for i in df.iloc[-1].values])
        latex += " \\\\\n\\bottomrule"

        filename = self.output_dir + "/table_descriptive_stats.txt"
        with open(filename, "w") as output_txt:
            output_txt.write(latex)
        self.logger.info(f"Saved balance table as: {filename}.")


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
    w = TableDescriptive(args)
    w.main()
