import sys
sys.path.append("../")

import json
import os
import pandas as pd

from utils import *


class TableTurnoutPlacebo:
    def __init__(self, arguments):
        self.logger = get_logger()

        # Determine input filename from arguments.
        self.input_dir = "../matched_bookings/out/modeled_turnout_placebo/"
        self.path = create_combo_path(arguments)
        self.input_filename = f"{self.input_dir}/{self.path}.json"

        # Set up output filename.
        if not os.path.exists(f"../matched_bookings/out/figures"):
            os.makedirs(f"../matched_bookings/out/figures")
        if not os.path.exists(f"../matched_bookings/out/figures/{self.path}"):
            os.makedirs(f"../matched_bookings/out/figures/{self.path}")
        self.output_dir = f"../matched_bookings/out/figures/{self.path}"

    def main(self):
        self.logger.info(f"Reading in turnout modeling results...")
        with open(self.input_filename, "r") as input_json:
            data = input_json.read()
        data = json.loads(data)

        # Prep results as DataFrame to write LaTeX.
        dfs = list()
        for d in data:
            df = pd.DataFrame(
                columns=[f"\\textbf{{Control/Treatment: ({str(d['split'][0])} days, {str(d['split'][1])} days)}}", "", "2016", "2020", "2012", "2020"]
            )
            df.loc[df.shape[0]] = ["Confined During Voting Days", ""] + [signify(y["coefficient"], y["p_value"]) for y in d["years"]]
            df.loc[df.shape[0]] = ["", ""] + [parenthesize(format(y["std_error"], rounding)) for y in d["years"]]
            df.loc[df.shape[0]] = ["Mean Control Turnout", ""] + [format(y["mean_control_turnout"], rounding) for y in d["years"]]
            df.loc[df.shape[0]] = ["Observations", ""] + [y["observations"] for y in d["years"]]
            dfs.append(df)

        # Write LaTeX.
        latex = f"\\begin{{tabular}}{{{'l' * len(dfs[0].columns)}}}\n\\toprule\n"
        latex += "& & \\textbf{2016 Placebo} & & \\textbf{2012 Placebo} & \\\\\n\\midrule\n"
        for df in dfs:
            latex += " & ".join(df.columns)
            latex += " \\\\\n\\midrule\n"
            for i in range(len(df)):
                latex += " & ".join([str(v) for v in df.iloc[i].values])
                latex += " \\\\\n"
            latex += " \\midrule\n"
        latex = latex.removesuffix(" \\midrule\n")
        latex += "\\bottomrule\n\\end{tabular}"
        with open(self.output_dir + "/table_turnout_placebo.txt", "w") as output_txt:
            output_txt.write(latex)
        self.logger.info(f"Saved placebo table as: {self.output_dir}/table_turnout_placebo.txt.")


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
    w = TableTurnoutPlacebo(args)
    w.main()
