import sys
sys.path.append("../")

import json
import os
import pandas as pd

from utils import *


class TableTurnout:
    def __init__(self, arguments):
        self.logger = get_logger()
        self.full = arguments.full
        self.input_dir_base = "matched_bookings"
        if self.full:
            self.input_dir_base = "full_bookings"

        # Determine input filename from arguments.
        self.input_dir = f"../{self.input_dir_base}/out/modeled_turnout/"
        self.path = create_combo_path(arguments)
        self.input_filename = f"{self.input_dir}/{self.path}.json"

        # Set up output filename.
        if not os.path.exists(f"../{self.input_dir_base}/out/figures"):
            os.makedirs(f"../{self.input_dir_base}/out/figures")
        if not os.path.exists(f"../{self.input_dir_base}/out/figures/{self.path}"):
            os.makedirs(f"../{self.input_dir_base}/out/figures/{self.path}")
        self.output_dir = f"../{self.input_dir_base}/out/figures/{self.path}"

    def main(self):
        self.logger.info(f"Reading in turnout modeling results...")
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
                df.loc[df.shape[0]] = [param_map["_".join(f["design"])], "", signify(f["coefficient"], f["p_value"])]
                df.loc[df.shape[0]] = ["", "", parenthesize(format(f["std_error"], rounding))]
            df.loc[df.shape[0]] = ["Mean Proportion of Voting Days Confined", "", format(d["mean_proportion_confined"], rounding)]
            df.loc[df.shape[0]] = ["Max. Proportion of Voting Days Confined", "", format(d["max_proportion_confined"], rounding)]
            df.loc[df.shape[0]] = ["Mean Control Turnout", "", format(d["mean_control_turnout"], rounding)]
            df.loc[df.shape[0]] = ["Observations", "", str(d["observations"])]
            dfs.append(df)
        to_merge = pd.concat([df.drop(columns=["", "Control:"]) for df in dfs[1:]], axis=1)
        df = pd.merge(dfs[0], to_merge, left_index=True, right_index=True)
        if self.full:
            dfs = [df]
        else:
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
        for i in range(9, 13):
            latex += " & ".join(dfs[0].iloc[i].values)
            latex += " \\\\\n"
        if not self.full:
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
            for i in range(9, 13):
                latex += " & ".join(dfs[1].iloc[i].values)
                latex += " \\\\\n"
        latex += "\\bottomrule\n\\end{tabular}"

        filename = self.output_dir + "/table_turnout.txt"
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
    w = TableTurnout(args)
    w.main()
