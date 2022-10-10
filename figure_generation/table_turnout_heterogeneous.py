import sys
sys.path.append("../")

import json
import os
import pandas as pd

from utils import *


class TableTurnoutHeterogeneity:
    def __init__(self, arguments):
        self.logger = get_logger()
        self.exclude_modeled_race = arguments.exclude_modeled_race

        # Determine input filename from arguments.
        if self.exclude_modeled_race:
            self.input_dir = "../matched_bookings/out/modeled_turnout_heterogeneous_race_reporting/"
        else:
            self.input_dir = "../matched_bookings/out/modeled_turnout_heterogeneous/"
        self.path = create_combo_path(arguments)
        self.input_filename = f"{self.input_dir}/{self.path}.json"

        # Set up output filename.
        if not os.path.exists(f"../matched_bookings/out/figures"):
            os.makedirs(f"../matched_bookings/out/figures")
        if not os.path.exists(f"../matched_bookings/out/figures/{self.path}"):
            os.makedirs(f"../matched_bookings/out/figures/{self.path}")
        self.output_dir = f"../matched_bookings/out/figures/{self.path}"

    def main(self):
        self.logger.info(f"Reading in turnout heterogeneity modeling results...")
        with open(self.input_filename, "r") as input_json:
            data = input_json.read()
        data = json.loads(data)

        # Prep results as DataFrame to write LaTeX.
        dfs = list()
        for d in data:
            df = pd.DataFrame(
                [["", "Treatment:", str(d["split"][1]) + " days"]],
                columns=["", "Control:", str(d["split"][0]) + " days"]
            )
            for f in d["fits"]:
                for p in f["params"]:
                    df.loc[df.shape[0]] = [param_map[p["parameter"]], "", signify(p["coefficient"], p["p_value"])]
                    df.loc[df.shape[0]] = ["", "", parenthesize(format(p["std_error"], rounding))]
            df.loc[df.shape[0]] = ["Mean Proportion of Voting Days Confined Black", "", format(d["mean_proportion_confined_black"], rounding)]
            df.loc[df.shape[0]] = ["Max. Proportion of Voting Days Confined Black", "", format(d["max_proportion_confined_black"], rounding)]
            df.loc[df.shape[0]] = ["Mean Proportion of Voting Days Confined White", "", format(d["mean_proportion_confined_white"], rounding)]
            df.loc[df.shape[0]] = ["Max. Proportion of Voting Days Confined White", "", format(d["max_proportion_confined_white"], rounding)]
            df.loc[df.shape[0]] = ["Mean Control Turnout Black", "", format(d["mean_control_turnout_black"], rounding)]
            df.loc[df.shape[0]] = ["Mean Control Turnout White", "", format(d["mean_control_turnout_white"], rounding)]
            df.loc[df.shape[0]] = ["Observations", "", str(d["observations"])]
            dfs.append(df)
        to_merge = pd.concat([df.drop(columns=["", "Control:"]) for df in dfs[1:]], axis=1)
        df = pd.merge(dfs[0], to_merge, left_index=True, right_index=True)
        dfs = (df.iloc[:, [0, 1, 2, 3, 4]], df.iloc[:, [0, 1, 5, 6, 7]])

        # Write LaTeX.
        latex = f"\\begin{{tabular}}{{{'l' * len(dfs[0].columns)}}}\n\\toprule\n"
        latex += " & ".join([f"\\textbf{{{column}}}" for column in dfs[0].columns])
        latex += "\\\\\n"
        latex += " & ".join([f"\\textbf{{{column}}}" for column in dfs[0].iloc[0].values])
        latex += "\\\\\n"
        latex += "\\midrule\n"
        for i in range(1, 5):
            latex += " & ".join(dfs[0].iloc[i].values)
            latex += " \\\\\n"
        latex += "\\midrule\n"
        for i in range(5, 9):
            latex += " & ".join(dfs[0].iloc[i].values)
            latex += " \\\\\n"
        latex += "\\midrule\n"
        for i in range(9, 16):
            latex += " & ".join(dfs[0].iloc[i].values)
            latex += " \\\\\n"
        latex += "\n\\toprule\n"
        latex += " & ".join([f"\\textbf{{{column}}}" for column in dfs[1].columns])
        latex += "\\\\\n"
        latex += " & ".join([f"\\textbf{{{column}}}" for column in dfs[1].iloc[0].values])
        latex += "\\\\\n"
        latex += "\\midrule\n"
        for i in range(1, 5):
            latex += " & ".join(dfs[1].iloc[i].values)
            latex += " \\\\\n"
        latex += "\\midrule\n"
        for i in range(5, 9):
            latex += " & ".join(dfs[1].iloc[i].values)
            latex += " \\\\\n"
        latex += "\\midrule\n"
        for i in range(9, 16):
            latex += " & ".join(dfs[1].iloc[i].values)
            latex += " \\\\\n"
        latex += "\\bottomrule\n\\end{tabular}"
        if self.exclude_modeled_race:
            filename = self.output_dir + "/table_turnout_heterogeneous_race_reporting.txt"
        else:
            filename = self.output_dir + "/table_turnout_heterogeneous.txt"
        with open(filename, "w") as output_txt:
            output_txt.write(latex)
        self.logger.info(f"Saved turnout table as: {filename}.")


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
    w = TableTurnoutHeterogeneity(args)
    w.main()
