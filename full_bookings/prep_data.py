import sys
sys.path.append("../")

import os

from dotenv import load_dotenv
from pymongo import MongoClient

from utils import *

load_dotenv()


class JdiDataPrep:
    def __init__(self, arguments):
        self.logger = get_logger()
        self.db = MongoClient(
            os.getenv("JDI_CLIENT_URI")
        ).get_database(
            os.getenv("JDI_DB")
        ).get_collection(
            os.getenv("JDI_COLLECTION")
        )

        # Collect arguments to pull correct match data file.
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

        # Determine input filename from arguments.
        self.input_dir = "../matched_bookings/out/prepped_data"
        self.path = create_combo_path(arguments)
        self.input_filename = f"{self.input_dir}/{self.path}.csv"

        # Set up output directory.
        if not os.path.exists("out"):
            os.makedirs("out")
        if not os.path.exists("out/prepped_data"):
            os.makedirs("out/prepped_data")
        if not os.path.exists(f"out/prepped_data/{self.path}"):
            os.makedirs(f"out/prepped_data/{self.path}")
        self.output_dir = "out/prepped_data/" + self.path

    def main(self):
        self.logger.info("Reading match data...")
        match_records = pd.read_csv(self.input_filename, low_memory=False)
        self.logger.info(f"Matched records: {len(match_records)}.")

        # Get rosters for sample to pass to bookings collection process.
        rosters = sorted(list(match_records["jail_id"].unique()))

        # Collect bookings from rosters.
        self.logger.info(f"Collecting bookings from {len(rosters)} rosters...")
        bookings = thread(self.get_bookings, rosters)
        bookings = pd.concat(bookings)
        self.logger.info(f"Records found: {len(bookings)}.")

        # Prep bookings DataFrame.
        bookings = self.clean_bookings(bookings)
        bookings.to_csv(self.output_dir + "/un_merged.csv", index=False)
        self.logger.info(f"Un-merged bookings records saved as: {self.output_dir}/un_merged.csv.")

        # Merge matches into bookings data.
        # Note: Some matched records may fall off for the following reasons:
        #  - scrape records have since been expunged during cleaning.
        #  - record was updated to denote age < 18.
        #  - other mismatch on merge columns due to updates.
        self.logger.info("Merging match records...")
        bookings = pd.merge(bookings, match_records, how="left", on=["jail_id", "jdi_id_person", "jdi_id_booking"])
        self.logger.info(f"Unmatched bookings: {len(bookings[bookings['l2_id'].isna()])}.")
        self.logger.info(f"Matched bookings: {len(bookings[bookings['l2_id'].notna()])}.")

        # Select matched data value in case of column mismatch.
        for column in merge_check_fields:
            bookings[column] = np.where(
                (bookings[column + "_x"] != bookings[column + "_y"]) & (bookings[column + "_y"].notna()),
                bookings[column + "_y"], bookings[column + "_x"]
            )
            bookings = bookings.drop(columns=[column + "_x", column + "_y"])
        bookings["matched"] = np.where(bookings["l2_id"].notna(), 1, 0)

        # Recalculate derived features for full bookings (e.g., LoS).
        self.logger.info("Recalculating derived features...")
        bookings = self.recalculate_co_variates(bookings, match_records)
        self.logger.info(f"Records processed: {len(bookings)}.")

        bookings.to_csv(self.output_dir + "/merged.csv", index=False)
        self.logger.info(f"Merged bookings records saved as: {self.output_dir}/merged.csv.")

    def clean_bookings(self, bookings):
        # Set up and rename columns.
        bookings["_id"] = bookings["_id"].astype(str)
        bookings["jail_id"] = bookings["state"] + "-" + bookings["jail"]
        bookings = bookings.rename(columns={
            "Name": "jdi_full_name",
            "_id": "jdi_id_booking",
            "Sex_Gender_Standardized": "jdi_gender",
            "Race_Ethnicity_Standardized": "jdi_race",
            "Age_Standardized": "jdi_age",
        })

        # Ensure no issues with jdi_num_charges and jdi_charge_types.
        assert len(bookings[(bookings["jdi_num_charges"] == 0) & (bookings["jdi_charge_types"].notna())]) == 0
        assert len(bookings[(bookings["jdi_num_charges"] > 0) & (bookings["jdi_charge_types"].isna())]) == 0

        # Only include people 18 years old at detention.
        self.logger.info("Removing underage voters...")
        bookings = bookings[(bookings["jdi_age"].isna()) | (bookings["jdi_age"] >= 18)]
        self.logger.info(f"Records found: {len(bookings)}.")

        # Simplify race values.
        bookings["jdi_race"] = np.where(bookings["jdi_race"].isin(["AAPI", "Indigenous", "Other POC"]), "Other", bookings["jdi_race"])
        bookings["jdi_race"] = np.where(bookings["jdi_race"] == "Unknown Race", np.nan, bookings["jdi_race"])

        # Simplify gender values.
        bookings["jdi_gender"] = np.where(~bookings["jdi_gender"].isin(["Male", "Female", "M", "F"]), np.nan, bookings["jdi_gender"])
        bookings["jdi_gender"] = np.where(bookings["jdi_gender"] == "Male", "M", bookings["jdi_gender"])
        bookings["jdi_gender"] = np.where(bookings["jdi_gender"] == "Female", "F", bookings["jdi_gender"])

        # Map most severe charge type.
        bookings["jdi_charge_types"] = bookings["jdi_charge_types"].fillna("")
        bookings["jdi_charge_types"] = np.where(bookings["jdi_charge_types"].str.contains("Violent"), "violent", bookings["jdi_charge_types"])
        bookings["jdi_charge_types"] = np.where(bookings["jdi_charge_types"].str.contains("Property"), "property", bookings["jdi_charge_types"])
        bookings["jdi_charge_types"] = np.where(bookings["jdi_charge_types"].str.contains("Drug"), "drug", bookings["jdi_charge_types"])
        bookings["jdi_charge_types"] = np.where(bookings["jdi_charge_types"].str.contains("Public Order"), "public order", bookings["jdi_charge_types"])
        bookings["jdi_charge_types"] = np.where(bookings["jdi_charge_types"].str.contains("DUI"), "dui", bookings["jdi_charge_types"])
        bookings["jdi_charge_types"] = np.where(bookings["jdi_charge_types"].str.contains("Criminal traffic"), "criminal traffic", bookings["jdi_charge_types"])
        bookings["jdi_charge_types"] = np.where(bookings["jdi_charge_types"].str.contains("TBD"), "", bookings["jdi_charge_types"])
        bookings["jdi_charge_types"] = np.where(bookings["jdi_charge_types"] == "", None, bookings["jdi_charge_types"])

        return bookings

    def get_bookings(self, roster):
        cursor = list(self.db.find({
            "meta.State": roster.split("-", 1)[0],
            "meta.County": roster.split("-", 1)[1],
            "meta.first_seen": {"$gte": self.earliest_date, "$lte": self.latest_date}
        }, {field: 1 for field in fields}))
        for d in cursor:
            d["jdi_date_admission"] = d["meta"]["first_seen"]
            d["jdi_date_release"] = d["meta"]["last_seen"]
            d["jdi_id_person"] = d["meta"]["jdi_inmate_id"]
            d["state"] = d["meta"]["State"]
            d["jail"] = d["meta"]["County"]
            del d["meta"]
            if "Charges" in d and type(d["Charges"]) == list:
                d["jdi_num_charges"] = len(d["Charges"])
                d["jdi_charge_types"] = ";".join(
                    [c["Charge_Standardized"]["l1"] for c in d["Charges"] if "Charge_Standardized" in c]
                )
                del d["Charges"]
        return pd.DataFrame(cursor)

    def recalculate_co_variates(self, bookings, match_records):
        # Recombine roster-level features.
        merge_columns = [
            "jail_first_scrape_date", "absentee_mailout_starts", "in_person_opens", "earliest_voting_date"
        ]
        to_merge = match_records[["jail_id"] + merge_columns].drop_duplicates()
        bookings = pd.merge(bookings.drop(columns=merge_columns), to_merge, how="left", on="jail_id")

        # Recalculate duration features.
        bookings = set_to_datetime(bookings)
        bookings["jdi_length_of_stay"] = (bookings["jdi_date_release"] - bookings["jdi_date_admission"]).dt.days + 1
        bookings["votable_days_in_custody"] = bookings.apply(
            lambda x: len(set(list(
                pd.date_range(x["earliest_voting_date"], self.election_day)
            )).intersection(set(list(
                pd.date_range(x["jdi_date_admission"], x["jdi_date_release"])
            )))), axis=1
        )
        bookings["votable_days"] = bookings.apply(
            lambda x: len(set(list(
                pd.date_range(x["earliest_voting_date"], self.election_day)
            ))), axis=1
        )
        bookings["pct_votable_days_in_custody"] = bookings["votable_days_in_custody"] / bookings["votable_days"]

        # Recreate dummy features.
        bookings = self.make_column_dummies(bookings, "jdi_charge_types")

        # Create new JDI-field-based demographic dummies where available.
        bookings = self.make_column_dummies(bookings, "jdi_gender")
        bookings["jdi_race"] = np.where(bookings["jdi_race"].isin(["AAPI", "Indigenous", "Other POC"]), "Other", bookings["jdi_race"])
        bookings["jdi_race"] = np.where(bookings["jdi_race"] == "Unknown Race", np.nan, bookings["jdi_race"])
        bookings = self.make_column_dummies(bookings, "jdi_race")

        return bookings

    @staticmethod
    def make_column_dummies(df, column):
        dummies = pd.get_dummies(df[column], prefix=column)
        dummies.loc[(dummies == 0).all(axis=1)] = None
        dummies.columns = [s.replace(" ", "_").replace("-", "_") for s in dummies.columns]
        for column in dummies.columns:
            if column in df.columns:
                df = df.drop(columns=[column])
        return pd.concat([df, dummies], axis=1)


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
    w = JdiDataPrep(args)
    w.main()
