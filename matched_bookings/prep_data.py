import sys
sys.path.append("../")

from dotenv import load_dotenv

from utils import *

load_dotenv()


class MatchDataPrep:
    def __init__(self, arguments):
        self.logger = get_logger()
        self.thresholding_column = arguments.column
        self.threshold = arguments.threshold
        self.active = arguments.active
        self.registered = arguments.registered
        self.no_charge = arguments.exclude_no_charge
        self.no_bond = arguments.exclude_no_bond
        self.election_day = election_day
        self.earliest_date = self.election_day - dt.timedelta(days=90)
        self.latest_date = self.election_day + dt.timedelta(days=90)
        self.voting_dates_by_state = voting_dates_by_state

        # Log input arguments.
        self.logger.info("Initializing prep_data.py with the following criteria...")
        self.logger.info(f"Match probability threshold: {self.threshold}.")
        self.logger.info(f"Column to threshold: {self.thresholding_column}.")
        self.logger.info(f"Filter out inactive voters? {self.active}.")
        self.logger.info(f"Filter out voters registered after Election Day 2020? {self.registered}.")
        self.logger.info(f"Filter out no-charge jails? {self.no_charge}.")
        self.logger.info(f"Filter out no-bond jails? {self.no_bond}.")
        self.logger.info(f"Matched bookings date range: {self.earliest_date.date()} to {self.latest_date.date()}.")

        # Specify input filename.
        self.input_filename = f"s3://{os.getenv('S3_BUCKET')}/{os.getenv('MATCH_FILE')}"

        # Set up output filename.
        if not os.path.exists("out"):
            os.makedirs("out")
        if not os.path.exists("out/prepped_data"):
            os.makedirs("out/prepped_data")
        self.output_dir = "out/prepped_data/"
        self.path = create_combo_path(arguments)
        self.output_filename = self.output_dir + self.path + ".csv"

    def main(self):
        self.logger.info("Reading in matched records...")
        df = pd.read_csv(self.input_filename, low_memory=False)

        # Subset to desired date range (+/- 90 days).
        df = self.filter_date_range(df)
        df = set_to_datetime(df)
        self.logger.info(f"Matched records: {len(df)}.")

        # Merge in earliest voting date by state.
        df = pd.merge(df, voting_dates_by_state, how="left", on="state")

        # Threshold matches on probability score (from input column).
        self.logger.info(f"Thresholding > {self.threshold} on {self.thresholding_column}.")
        df = df[df[self.thresholding_column] > self.threshold]
        self.logger.info(f"Matched records: {df.shape}.")

        # Subset to active voters if specified (in input).
        if self.active:
            self.logger.info("Filtering to L2-Active voters.")
            df = df[df["l2_active"] == 1]
            self.logger.info(f"Matched records: {df.shape}.")

        # Subset to voters registered pre-Election Day.
        if self.registered:
            self.logger.info("Filtering to voters registered by Election Day.")
            df = df[df["l2_date_registered_calculated"] <= self.election_day]
            self.logger.info(f"Matched records: {df.shape}.")

        # Create length of stay (LOS) feature.
        df["jdi_length_of_stay"] = (df["jdi_date_release"] - df["jdi_date_admission"]).dt.days + 1

        # Simplify L2 race/ethnicity and party encodings.
        df = simplify_l2_race(df)
        df = simplify_l2_party(df)

        # Subset to voters from jails that report charges.
        if self.no_charge:
            # First, find and exclude jails that have only 0 jdi_num_charges.
            num_charges = df.groupby("jail_id")["jdi_num_charges"].unique().reset_index()
            num_charges["num_charges"] = num_charges["jdi_num_charges"].apply(lambda l: sum(l))
            zero_charges = num_charges[num_charges["num_charges"] == 0]["jail_id"].unique()
            df = df[~df["jail_id"].isin(zero_charges)]
            self.logger.info(f"Excluded records from {len(zero_charges)} jails with no charge data.")
            self.logger.info(f"Matched records: {len(df)}.")

            # Second, find and exclude jails with no charge types.
            charge_types = df.groupby("jail_id")["jdi_charge_types"].unique().reset_index()
            charge_types["missing"] = charge_types["jdi_charge_types"].apply(lambda l: pd.isna(l).all())
            missing_charge_types = charge_types[charge_types["missing"]]["jail_id"].unique()
            df = df[~df["jail_id"].isin(missing_charge_types)]
            self.logger.info(f"Excluded records from {len(missing_charge_types)} jails with no charge types.")
            self.logger.info(f"Matched records: {len(df)}.")

        # Subset to voters from jails that report bond amounts.
        if self.no_bond:
            total_bond = df.groupby("jail_id")["jdi_bond"].sum().reset_index()
            missing_bond = total_bond[total_bond["jdi_bond"] == 0]["jail_id"].unique()
            df = df[~df["jail_id"].isin(missing_bond)]
            self.logger.info(f"Excluded records from {len(missing_bond)} jails with no bond data.")
            self.logger.info(f"Matched records: {len(df)}.")

        # Set up independent variable columns.
        df["votable_days_in_custody"] = df.apply(
            lambda x: len(set(list(
                pd.date_range(x["earliest_voting_date"], self.election_day)
            )).intersection(set(list(
                pd.date_range(x["jdi_date_admission"], x["jdi_date_release"])
            )))), axis=1
        )
        df["votable_days"] = df.apply(
            lambda x: len(set(list(
                pd.date_range(x["earliest_voting_date"], self.election_day)
            ))), axis=1
        )
        df["pct_votable_days_in_custody"] = df["votable_days_in_custody"] / df["votable_days"]

        # Create dummy columns for categorical features.
        for column in dummy_columns:
            df = make_column_dummies(df, column)

        # Replace spaces and hyphens in columns to appease PanelOLS.from_formula.
        df.columns = [s.replace(" ", "_").replace("-", "_") for s in df.columns]

        # Output to CSV.
        df.to_csv(self.output_filename, index=False)
        self.logger.info(f"Wrote file to CSV:")
        self.logger.info(f"{self.output_filename}.")

    def filter_date_range(self, df):
        df["jdi_date_admission"] = pd.to_datetime(df["jdi_date_admission"])
        return df[
            (df["jdi_date_admission"] >= self.earliest_date) &
            (df["jdi_date_admission"] <= self.latest_date)
        ]


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
    w = MatchDataPrep(args)
    w.main()
