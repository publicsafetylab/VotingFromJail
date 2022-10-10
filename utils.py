import datetime as dt
import logging
import numpy as np
import os
import pandas as pd
import tqdm

from dotenv import load_dotenv
from linearmodels.panel import PanelOLS
from multiprocessing.dummy import Pool as ThreadPool

load_dotenv()


def get_logger():
    """
    Establishes files for logging outputs.
    """
    # Specify output filepath and logging specs.
    logging.basicConfig(format=f"%(asctime)s %(filename)s %(levelname)s %(message)s",
                        level=logging.INFO,
                        datefmt="%m/%d/%Y %I:%M:%S %p",
                        filename="logger.log")
    logger = logging.getLogger(__name__)

    # Also, print to console.
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def thread(worker, jobs, n=15):
    """
    Generic method to parallelize a function over a list of inputs.

    :param (func) worker: Method to run on each element of jobs.
    :param (list) jobs: List of objects on which to run worker.
    :param (int) n: Number of threads to parallelize.
    :return: List of results of pool process.
    """
    pool = ThreadPool(n)
    results = []
    for result in tqdm.tqdm(pool.imap_unordered(worker, jobs), total=len(jobs)):
        results.append(result)
    pool.close()
    pool.join()
    return results


def create_combo_path(arguments):
    """
    Takes input arguments and stitches together a path from common parameters.

    :param arguments: Python argparse NameSpace.
    :return: String output in format "a_{}_c_{}_r_{}_t_{}_xb_{}_xc_{}".
    """
    path = f"a_{str(arguments.active).lower()[0]}_"
    path += f"c_{str(arguments.column).lower()}_"
    path += f"r_{str(arguments.registered).lower()[0]}_"
    path += f"t_{arguments.threshold}_"
    path += f"xb_{str(arguments.exclude_no_bond).lower()[0]}_"
    path += f"xc_{str(arguments.exclude_no_charge).lower()[0]}"
    return path


def make_column_dummies(df, column):
    """
    Takes a pandas.DataFrame column and creates dummies from it.

    :param df: Input pandas.DataFrame of booking records.
    :param column: Column of pandas.DataFrame from whose values to create dummy columns.
    :return: Output pandas.DataFrame with new dummy columns.
    """
    dummies = pd.get_dummies(df[column], prefix=column)
    dummies.loc[(dummies == 0).all(axis=1)] = None
    return pd.concat([df, dummies], axis=1)


def simplify_l2_party(df):
    """
    Takes in pandas.DataFrame and reduces L2 party affiliation values.

    :param df: Input pandas.DataFrame of booking records with L2 voter matches.
    :return: Output pandas.DataFrame with new l2_party values.
    """
    df["l2_party"] = np.where(
        ~df["l2_party"].isin(["Democratic", "Republican", "Unknown"]),
        "Non-Partisan or Other", df["l2_party"]
    )
    df["l2_party"] = np.where(df["l2_party"] == "Unknown", np.nan, df["l2_party"])
    return df


def simplify_l2_race(df):
    """
    Takes in pandas.DataFrame and reduces L2 race/ethnicity values.

    :param df: Input pandas.DataFrame of booking records with L2 voter matches.
    :return: Output pandas.DataFrame with new l2_race values.
    """
    df["l2_race"] = df["l2_race"].map(l2_race_map)
    df["l2_race"] = np.where(df["l2_race"].isin(["Hispanic", "Other", "Asian"]), "Other", df["l2_race"])
    df["l2_race"] = np.where(df["l2_race"] == "Unknown Race", np.nan, df["l2_race"])
    return df


def signify(coefficient, p_value):
    """
    Takes in a coefficient and asterisks based on associated p-values.

    :param coefficient: OLS variable coefficient.
    :param p_value: Associated feature p-value.
    :return: Rounded, asterisked coefficient value.
    """
    s = format(coefficient, rounding)
    if p_value < 0.1:
        s += "*"
    if p_value < 0.05:
        s += "*"
    if p_value < 0.01:
        s += "*"
    return s


def parenthesize(f):
    """
    Parenthesizes standard errors associated with coefficients.

    :param f: Standard error.
    :return: Parenthesized standard error.
    """
    return "(" + str(f) + ")"


def set_to_datetime(df):
    """
    Takes pandas.DataFrame and converts date columns to datetime.

    :param (pandas.DataFrame) df: pandas.DataFrame of matched records.
    :return: pandas.DataFrame with date columns set to datetime.
    """
    for column in df.columns:
        if "date" in column:
            df[column] = pd.to_datetime(df[column])
    return df


def treatment_control_split(base_df, control, treatment_rollback, no_charge, no_bond):
    """
    Takes in a pandas.DataFrame and splits it into Treatment/Control based on input arguments.

    :param base_df: pandas.DataFrame of booking records.
    :param control: Number of days in control window.
    :param treatment_rollback: Number of days to remove largest voting window.
    :param no_charge: Indicator to exclude records missing charge data.
    :param no_bond: Indicator to exclude records missing bond data.
    :return: Recombined pandas.DataFrame of Treatment/Control split data.
    """
    # Subset to admissions in range [first voting + treatment_rollback, Election Day + control].
    df = base_df.copy()
    df = df[df["jdi_date_admission"] >= earliest_voting_date + dt.timedelta(days=treatment_rollback)]
    df = df[df["jdi_date_admission"] <= election_day + dt.timedelta(days=control)]

    # Assign Treatment (T) vs. Control (C).
    df["treatment"] = np.where(df["jdi_date_admission"] <= election_day, 1, 0)

    # Filter to admissions within voting period for each state.
    df = df[df["jdi_date_admission"] >= df["earliest_voting_date"]]

    # Drop duplicate bookings and ensure mutually exclusive cohorts.
    treatment = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]
    treatment = treatment.sort_values(by=["l2_id", "jdi_date_admission"]).drop_duplicates("l2_id", keep="last")
    control = control.sort_values(by=["l2_id", "jdi_date_admission"]).drop_duplicates("l2_id", keep="last")
    control = control[~control["l2_id"].isin(treatment["l2_id"].unique())]
    to_model = pd.concat([treatment, control])

    # Filter to exclude rows missing co-variates.
    to_model = to_model[
        (to_model["l2_age"].notna()) &
        (to_model["l2_gender"].notna()) &
        (to_model["l2_race"].notna()) &
        (to_model["l2_party"].notna())
    ]
    if no_charge:
        to_model = to_model[(to_model["jdi_charge_types"].notna()) & (to_model["jdi_num_charges"].notna())]
    if no_bond:
        to_model = to_model[(to_model["jdi_bond"].notna())]

    # Set week number and re-index in case time effects modeled downstream.
    to_model["week"] = to_model["jdi_date_admission"].dt.isocalendar().week.astype(int)
    to_model = to_model.set_index(["jail_id", "week"])
    return to_model


def model(to_model, dependent, independent, entity_fx, time_fx):
    """
    Takes in a pandas.DataFrame and runs it through PanelOLS based on input arguments.

    :param to_model: pandas.DataFrame of booking records.
    :param dependent: Dependent variable (outcome) in model.
    :param independent: Independent variables (features) in model.
    :param entity_fx: Indicator to include fixed entity effects.
    :param time_fx: Indicator to include fixed time effects.
    :return: PanelOLS.fit class with modeling results.
    """
    # Specify model formula as string.
    formula = f"{dependent} ~ "
    if independent:
        formula += " + ".join(independent)
    if entity_fx:
        formula += " + EntityEffects"
    if time_fx:
        formula += " + TimeEffects"

    print(formula)

    # Model and fit with clustered co-variance, optional entity and time effects.
    panel_model = PanelOLS.from_formula(formula=formula, data=to_model)
    panel_fit = panel_model.fit(cov_type="clustered", cluster_entity=entity_fx, cluster_time=time_fx)
    return panel_fit


def treatment_control_split_full_bookings(base_df, control, treatment_rollback, no_charge, no_bond):
    """
    Takes in a pandas.DataFrame and splits it into Treatment/Control based on input arguments.

    :param base_df: pandas.DataFrame of booking records.
    :param control: Number of days in control window.
    :param treatment_rollback: Number of days to remove largest voting window.
    :param no_charge: Indicator to exclude records missing charge data.
    :param no_bond: Indicator to exclude records missing bond data.
    :return: Recombined pandas.DataFrame of Treatment/Control split data.
    """
    # Subset to admissions in range [first voting + treatment_rollback, Election Day + control].
    df = base_df.copy()
    df = df[df["jdi_date_admission"] >= earliest_voting_date + dt.timedelta(days=treatment_rollback)]
    df = df[df["jdi_date_admission"] <= election_day + dt.timedelta(days=control)]

    # Assign Treatment (T) vs. Control (C).
    df["treatment"] = np.where(df["jdi_date_admission"] <= election_day, 1, 0)

    # Filter to admissions within voting period for each state.
    df = df[df["jdi_date_admission"] >= df["earliest_voting_date"]]

    # Set up single person_id on which to deduplicate.
    df["jdi_id_person_joint"] = df["jail_id"] + "-" + df["jdi_id_person"]

    # Drop duplicate bookings and ensure mutually exclusive cohorts.
    treatment = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]
    treatment = treatment.sort_values(by=["jail_id", "jdi_id_person", "jdi_date_admission"]).drop_duplicates(["jdi_id_person_joint"], keep="last")
    control = control.sort_values(by=["jail_id", "jdi_id_person", "jdi_date_admission"]).drop_duplicates(["jdi_id_person_joint"], keep="last")
    control = control[~control["jdi_id_person_joint"].isin(treatment["jdi_id_person_joint"].unique())]
    to_model = pd.concat([treatment, control])
    to_model = to_model.drop(columns=["jdi_id_person_joint"])

    # Filter to exclude rows missing co-variates.
    to_model = to_model[
        (to_model["jdi_age"].notna()) &
        (to_model["jdi_gender"].notna()) &
        (to_model["jdi_race"].notna())
    ]
    if no_charge:
        to_model = to_model[(to_model["jdi_charge_types"].notna()) & (to_model["jdi_num_charges"].notna())]
    if no_bond:
        to_model = to_model[(to_model["jdi_bond"].notna())]

    # Assume matched = 0 implies l2_voted_indicator = 0
    to_model["l2_voted_indicator"] = np.where(to_model["l2_voted_indicator"].isna(), 0, to_model["l2_voted_indicator"])

    # Set week number and re-index in case time effects modeled downstream.
    to_model["week"] = to_model["jdi_date_admission"].dt.isocalendar().week.astype(int)
    to_model = to_model.set_index(["jail_id", "week"])
    return to_model


# Earliest voting dates by state and overall.
voting_dates_by_state = pd.read_csv(f"s3://{os.getenv('S3_BUCKET')}/{os.getenv('VOTING_DATES_FILE')}")
earliest_voting_date = pd.to_datetime(voting_dates_by_state["earliest_voting_date"]).min()


# Election Day 2020 (November 3rd).
election_day = dt.datetime(2020, 11, 3, 0, 0)


# List of states included in data.
states = sorted(list({
    "AL", "AR", "AZ", "CA", "CO",
    "FL", "GA", "IA", "ID", "IL",
    "IN", "KS", "KY", "LA", "MD",
    "ME", "MI", "MN", "MO", "MS",
    "MT", "NC", "ND", "NE", "NH",
    "NJ", "NM", "NV", "NY", "OH",
    "OK", "OR", "PA", "SC", "SD",
    "TN", "TX", "UT", "VA", "WA",
    "WI", "WY"
}))


# States missing from data.
missing_states = sorted(list({"AK", "CT", "DE", "HI", "MA", "RI", "VT", "WV"}))


# Simplification mapping of L2 race/ethnicity categories.
l2_race_map = {
    "East and South Asian": "Asian",
    "European": "White",
    "Hispanic and Portuguese": "Hispanic",
    "Likely African-American": "Black",
    "Other": "Other"
}


# Columns for which to create dummy variables.
dummy_columns = [
    "jdi_charge_types",
    "l2_gender",
    "l2_party",
    "l2_race"
]


# Control windows (multiples of 7).
control_windows = list(7 * n for n in range(1, 7))


# Balance co-variates.
balance_co_variates = [
    "l2_age",
    "l2_gender_M",
    # "l2_gender_F", Note: leave-one-out l2_gender.
    "l2_race_White",
    "l2_race_Black",
    # "l2_race_Other", Note: leave-one-out l2_race.
    "l2_party_Republican",
    "l2_party_Democratic",
    # "l2_party_Non_Partisan_or_Other", Note: leave-one-out l2_party.
    "jdi_charge_types_violent",
    "jdi_charge_types_public_order",
    "jdi_charge_types_property",
    "jdi_charge_types_dui",
    "jdi_charge_types_drug",
    # "jdi_charge_types_criminal_traffic", Note: leave-one-out jdi_charge_types.
    # "jdi_length_of_stay", Note: excluded because contrary to temporal requirements for balance checking.
    "jdi_num_charges",
]


# Turnout co-variates.
turnout_co_variates = balance_co_variates + ["jdi_length_of_stay"]


# Heterogeneity co-variates.
turnout_heterogeneity_co_variates = list(set(turnout_co_variates).difference({"l2_race_White", "l2_race_Black"}))


# Co-variate key mapping for LaTeX table production.
param_map = {
    # Co-variates.
    "l2_age": "Age",
    "l2_gender_M": "Male",
    "l2_gender_F": "Female",
    "l2_race_White": "White",
    "l2_race_Black": "Black",
    "l2_race_Other": "Other Race",
    "l2_party_Republican": "Republican",
    "l2_party_Democratic": "Democrat",
    "l2_party_Non_Partisan_or_Other": "Non-Partisan or Other Party",
    "jdi_charge_types_violent": "Top Charge: Violent",
    "jdi_charge_types_public_order": "Top Charge: Public Order",
    "jdi_charge_types_property": "Top Charge: Property",
    "jdi_charge_types_dui": "Top Charge: DUI",
    "jdi_charge_types_drug": "Top Charge: Drug",
    "jdi_charge_types_criminal_traffic": "Top Charge: Criminal Traffic",
    "jdi_length_of_stay": "Length of Stay (days)",
    "jdi_num_charges": "Number of Charges",
    # Turnout.
    "treatment_no_co_variates": "Confined During Voting Days",
    "treatment_co_variates": "with covariates",
    "pct_votable_days_in_custody_no_co_variates": "Proportion of Voting Days Confined",
    "pct_votable_days_in_custody_co_variates": "with covariates",
    # Turnout heterogeneity.
    "treatment": "Confined During Voting Days",
    "treatment_x_black": "Confined $\\times$ Black",
    "pct_votable_days_in_custody": "Proportion of Voting Days Confined",
    "pct_votable_days_in_custody_x_black": "Proportion Confined $\\times$ Black",
    # Descriptive.
    "l2_voted_indicator": "Turnout",
    # Full bookings.
    "matched": "Registered (Matched)",
    "matched_registered": "Registered (Matched) by Election Day",
    "jdi_age": "Age",
    "jdi_gender_M": "Male",
    "jdi_gender_F": "Female",
    "jdi_race_White": "White",
    "jdi_race_Black": "Black",
    "jdi_race_Other": "Other Race",
}


# Global float formatting (i.e. rounding).
rounding = ".3f"


# Global table row (co-variate) sorting.
table_sorter = [
    "Age",
    "Black",
    "White",
    "Democrat",
    "Republican",
    "Male",
    "Number of Charges",
    "Top Charge: DUI",
    "Top Charge: Drug",
    "Top Charge: Property",
    "Top Charge: Public Order",
    "Top Charge: Violent",
    "Has L2 Match"
]


# JDI fields to collect for full booking data collection.
fields = [
    "_id",
    "meta",
    "Name",
    "Age_Standardized",
    "Sex_Gender_Standardized",
    "Race_Ethnicity_Standardized",
    "Charges.Charge_Standardized"
]


# Columns to check when merging full booking data with existing match records.
merge_check_fields = [
    "jail",
    "jdi_age",
    "jdi_charge_types",
    "jdi_date_admission",
    "jdi_date_release",
    "jdi_full_name",
    "jdi_gender",
    "jdi_num_charges",
    "jdi_race",
    "state"
]


# Balance co-variates for full JDI bookings.
full_bookings_balance_co_variates = [
    "jdi_age",
    "jdi_gender_M",
    # "jdi_gender_F", Note: leave-one-out jdi_gender.
    "jdi_race_White",
    "jdi_race_Black",
    # "jdi_race_Other", Note: leave-one-out jdi_race.
    "jdi_charge_types_violent",
    "jdi_charge_types_public_order",
    "jdi_charge_types_property",
    "jdi_charge_types_dui",
    "jdi_charge_types_drug",
    # "jdi_charge_types_criminal_traffic", Note: leave-one-out jdi_charge_types.
    # "jdi_length_of_stay", Note: excluded because contrary to temporal requirements for balance checking.
    "jdi_num_charges"
]


# Turnout co-variates for full JDI bookings.
full_bookings_turnout_co_variates = full_bookings_balance_co_variates + ["jdi_length_of_stay"]


# States where race is reported directly by the state (not L2-modeled).
race_reporting_states = ["AL", "FL", "GA", "LA", "NC", "SC", "TN", "TX"]
