import sys
sys.path.append("../")

import json
import os
import pandas as pd

from functools import reduce

from utils import *


class TableBalance:
    def __init__(self, arguments):
        self.logger = get_logger()
        self.full = arguments.full
        self.input_dir_base = "matched_bookings"
        if self.full:
            self.input_dir_base = "full_bookings"

        # Determine input filename from arguments.
        self.input_dir = f"../{self.input_dir_base}/out/modeled_balance/"
        self.path = create_combo_path(arguments)
        self.input_filename = f"{self.input_dir}/{self.path}.json"

        # Set up output filename.
        if not os.path.exists(f"../{self.input_dir_base}/out/figures"):
            os.makedirs(f"../{self.input_dir_base}/out/figures")
        if not os.path.exists(f"../{self.input_dir_base}/out/figures"):
            os.makedirs(f"../{self.input_dir_base}/out/figures/{self.path}")
        self.output_dir = f"../{self.input_dir_base}/out/figures/{self.path}"

    def main(self):
        self.logger.info(f"Reading in balance modeling results...")
        with open(self.input_filename, "r") as input_json:
            data = input_json.read()
        data = json.loads(data)

        dfs = list()
        for d in data:
            control = str(d["split"][0]) + " days"
            treatment = str(d["split"][1]) + " days"

            # Stack coefficients and standard errors for LaTeX.
            df = pd.DataFrame(d["params"])
            df["coefficient"] = df.apply(lambda x: signify(x["coefficient"], x["p_value"]), axis=1)
            df["std_error"] = df["std_error"].apply(lambda f: parenthesize(format(f, rounding)))
            df.sort_values(by="parameter", key=lambda column: column.map(lambda e: table_sorter.index(e)), inplace=True)
            df["_list"] = df[["coefficient", "std_error"]].values.tolist()
            df = df.explode("_list")
            df = pd.concat([pd.DataFrame([{"parameter": "", "_list": treatment}]), df]).reset_index(drop=True)
            df = df.rename(columns={"parameter": "Window:", "_list": f"{control}"})
            df = df.drop(columns=["coefficient", "p_value", "std_error"])
            df["Window:"] = np.where(df[control].str.contains("\\("), "", df["Window:"])

            # Add number of observations and p-value.
            df = df.append(pd.DataFrame({
                "Window:": "Observations", f"{control}": str(d["observations"])
            }, index=[0]), ignore_index=True)
            df = df.append(pd.DataFrame({
                "Window:": "Joint F-Test p-value", f"{control}": str(format(d["p_value"], rounding))
            }, index=[0]), ignore_index=True)
            dfs.append(df)

        # Combine all T/C splits into single DataFrame.
        df = reduce(lambda df1, df2: pd.merge(df1, df2.drop(columns=["Window:"]), left_index=True, right_index=True), dfs)

        # Set up for LaTeX.
        df.insert(loc=1, column="Control:", value=list("_" * len(df)))
        df = df.rename(columns={"Window:": ""})
        df["Control:"] = ""
        df.loc[0, "Control:"] = "Treatment:"

        # Write LaTeX.
        latex = f"\\begin{{tabular}}{{{'l' * len(df.columns)}}}\n\\toprule\n"
        latex += " & \\textbf{Control:} & "
        if self.full:
            latex += " & ".join([f"\\textbf{{{control}}}" for control in df.columns[-4:]])
        else:
            latex += " & ".join([f"\\textbf{{{control}}}" for control in df.columns[-6:]])
        latex += "\\\\"
        latex += "\n & \\textbf{Treatment:} & "
        if self.full:
            latex += " & ".join([f"\\textbf{{{treatment}}}" for treatment in df.iloc[0].values[-4:]])
        else:
            latex += " & ".join([f"\\textbf{{{treatment}}}" for treatment in df.iloc[0].values[-6:]])
        latex += "\\\\\n\\midrule\n"
        for i in range(1, len(df) - 2):
            latex += " & ".join(df.iloc[i].values)
            latex += " \\\\\n"
        latex += "\\midrule\n"
        for i in range(len(df) - 2, len(df)):
            latex += " & ".join(df.iloc[i].values)
            latex += " \\\\\n"
        latex += "\\bottomrule\n\\end{tabular}"
        filename = self.output_dir + "/table_balance.txt"
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
    w = TableBalance(args)
    w.main()
