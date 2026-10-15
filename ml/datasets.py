"""The transaction formats SecureBank can train on and score.

Each format ("profile") says which columns a file must have, how to clean
them, how to turn a row into the numbers the model reads, and how to
describe those numbers to a person. Datasets with different columns cannot
share a model, so each profile is trained and saved on its own, under
ml/artifacts/<key>/.

    creditcard  Kaggle credit-card fraud (Time, V1..V28, Amount, Class)
    sparkov     Simulated card payments (Kaggle "fraudTrain.csv"/"fraudTest.csv")
    paysim      Simulated mobile money transfers (Kaggle "PaySim")
"""

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

LABEL = "label"  # The cleaned label column, whatever the file called it.


class Field:
    """One raw input column: what a file must contain and the Check form asks for."""

    def __init__(self, name, label, kind="number", minimum=None, maximum=None, options=None, help=None):
        self.name, self.label, self.kind = name, label, kind
        self.minimum, self.maximum, self.options, self.help = minimum, maximum, options, help

    def to_dict(self):
        data = {"name": self.name, "label": self.label, "kind": self.kind}
        for key in ("minimum", "maximum", "help"):
            if getattr(self, key) is not None:
                data[key] = getattr(self, key)
        if self.options:
            data["options"] = [{"value": v, "label": label} for v, label in self.options.items()]
        return data


class Group:
    """Model inputs explained together, e.g. amount and log(amount) as "Amount".

    describe(values) turns the group's input values (a dict) into words.
    """

    def __init__(self, label, columns, describe, onehot=False):
        self.label, self.columns, self.describe = label, columns, describe
        # One-hot groups hold one option set to 1, so their "typical" value is the commonest option.
        self.onehot = onehot


class Profile:
    key = title = description = currency = None
    target = None              # label column in the raw file, optional when importing
    fields = []                # raw inputs: required in a file, stored with each transaction
    optional = []              # raw columns stored when a file has them, never model inputs
    readable = True            # False when the inputs have no plain meaning (anonymised)

    @property
    def required_columns(self):
        return [f.name for f in self.fields]

    @property
    def feature_names(self):
        return [c for g in self.groups for c in g.columns]

    def clean(self, df):
        """Coerce types and drop invalid rows. Returns (clean DataFrame, rows dropped).

        The result has the required columns, any optional ones present, and
        LABEL when the file has the target column (NaN where a row has none).
        """
        out = pd.DataFrame(index=df.index)
        valid = pd.Series(True, index=df.index)
        for field in self.fields:
            column, ok = _clean_column(df[field.name], field)
            out[field.name] = column
            valid &= ok
        for name in self.optional:
            if name in df.columns:
                out[name] = df[name].astype(str).str.strip()
        self.clean_extra(df, out)
        if self.target in df.columns:
            label = pd.to_numeric(df[self.target], errors="coerce")
            # A label is optional per row, but when present it must be 0 or 1.
            valid &= label.isna() | label.isin([0, 1])
            out[LABEL] = label.astype(float)
        out = out[valid]
        return out, int((~valid).sum())

    def clean_extra(self, raw, out):
        """Hook for profile-specific stored columns derived from the raw file."""

    def stored_columns(self, df):
        """Columns kept with each transaction and used to spot duplicates."""
        return [c for c in df.columns if c != LABEL]

    def parse_input(self, payload):
        """Validate one transaction typed into the Check form. Returns (row, errors)."""
        if not isinstance(payload, dict):
            return None, ["Body must be a JSON object with the fields " + ", ".join(self.required_columns)]
        row, errors = {}, []
        for field in self.fields:
            value, error = _parse_value(payload.get(field.name), field)
            if error:
                errors.append(error)
            else:
                row[field.name] = value
        return row, errors

    def features(self, df):
        raise NotImplementedError

    def amounts(self, df):
        raise NotImplementedError

    def timestamps(self, df, config):
        raise NotImplementedError

    def transaction_types(self, df):
        return np.full(len(df), "card", dtype=object)

    def to_dict(self):
        return {
            "key": self.key, "title": self.title, "description": self.description,
            "currency": self.currency, "readable": self.readable,
            "fields": [f.to_dict() for f in self.fields],
            "label_column": self.target, "optional_columns": self.optional,
        }


def _clean_column(series, field):
    """Return (cleaned column, valid mask) for one field."""
    if field.kind == "number":
        values = pd.to_numeric(series, errors="coerce").astype(float)
        ok = np.isfinite(values)
        if field.minimum is not None:
            ok &= values >= field.minimum
        if field.maximum is not None:
            ok &= values <= field.maximum
        return values, ok
    if field.kind == "choice":
        values = series.astype(str).str.strip()
        return values, values.isin(list(field.options))
    # Dates are stored as ISO text so they hash and serialise the same way every time.
    parsed = pd.to_datetime(series, errors="coerce")
    fmt = "%Y-%m-%d %H:%M:%S" if field.kind == "datetime" else "%Y-%m-%d"
    return parsed.dt.strftime(fmt), parsed.notna()


def _parse_value(value, field):
    if field.kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, f"{field.name} must be a number"
        if not math.isfinite(value):
            return None, f"{field.name} must be finite"
        if field.minimum is not None and value < field.minimum:
            return None, f"{field.name} cannot be below {field.minimum:g}" if field.minimum else f"{field.name} cannot be negative"
        if field.maximum is not None and value > field.maximum:
            return None, f"{field.name} cannot be above {field.maximum:g}"
        return float(value), None
    if field.kind == "choice":
        if value not in field.options:
            return None, f"{field.name} must be one of: {', '.join(field.options)}"
        return value, None
    try:
        parsed = pd.Timestamp(str(value)) if value not in (None, "") else None
    except ValueError:
        parsed = None
    if parsed is None or pd.isna(parsed):
        return None, f"{field.name} must be a date{' and time' if field.kind == 'datetime' else ''}"
    return parsed.strftime("%Y-%m-%d %H:%M:%S" if field.kind == "datetime" else "%Y-%m-%d"), None


def _money(symbol):
    return lambda v: f"{symbol}{v['amount' if 'amount' in v else 'Amount']:,.2f}"


def _onehot(prefix, labels):
    """Describe a one-hot group by the option whose column is set."""
    def describe(values):
        for key, label in labels.items():
            if values.get(f"{prefix}{key}", 0) >= 0.5:
                return label
        return "Other"
    return describe


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype=float)) for a in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# --- Kaggle credit-card dataset ------------------------------------------------

class CreditCard(Profile):
    key = "creditcard"
    title = "Kaggle credit card"
    description = ("European card payments from September 2013. V1 to V28 are anonymised, "
                   "so explanations name the column but not its meaning.")
    currency = "EUR"
    target = "Class"
    readable = False
    fields = (
        [Field("Time", "Time", minimum=0, help="Seconds since the first transaction")]
        + [Field(f"V{i}", f"V{i}") for i in range(1, 29)]
        + [Field("Amount", "Amount (€)", minimum=0)]
    )

    @property
    def groups(self):
        groups = [Group("Time", ["Time"], lambda v: f"{v['Time']:,.0f} s")]
        groups += [Group(f"V{i}", [f"V{i}"], lambda v, c=f"V{i}": f"{v[c]:.2f}") for i in range(1, 29)]
        groups.append(Group("Amount", ["Amount"], _money("€")))
        return groups

    def features(self, df):
        return df[self.required_columns].astype(float)

    def amounts(self, df):
        return df["Amount"].to_numpy(dtype=float)

    def timestamps(self, df, config):
        # Time counts seconds from the dataset's first transaction (see config.DATASET_START).
        start = config["DATASET_START"]
        return [start + timedelta(seconds=float(s)) for s in df["Time"]]


# --- Sparkov simulated card transactions ---------------------------------------

SPARKOV_CATEGORIES = {
    "entertainment": "Entertainment", "food_dining": "Food and dining",
    "gas_transport": "Fuel and transport", "grocery_net": "Groceries (online)",
    "grocery_pos": "Groceries (in store)", "health_fitness": "Health and fitness",
    "home": "Home", "kids_pets": "Kids and pets", "misc_net": "Other (online)",
    "misc_pos": "Other (in store)", "personal_care": "Personal care",
    "shopping_net": "Shopping (online)", "shopping_pos": "Shopping (in store)", "travel": "Travel",
}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class Sparkov(Profile):
    key = "sparkov"
    title = "Sparkov card payments"
    description = "Simulated US card payments from 2019 and 2020, with merchant, customer and location details."
    currency = "USD"
    target = "is_fraud"
    fields = [
        Field("trans_date_trans_time", "Transaction time", kind="datetime"),
        Field("category", "Merchant category", kind="choice", options=SPARKOV_CATEGORIES),
        Field("amt", "Amount ($)", minimum=0),
        Field("dob", "Customer date of birth", kind="date"),
        Field("city_pop", "Population of the customer's city", minimum=0),
        Field("lat", "Customer latitude", minimum=-90, maximum=90),
        Field("long", "Customer longitude", minimum=-180, maximum=180),
        Field("merch_lat", "Merchant latitude", minimum=-90, maximum=90),
        Field("merch_long", "Merchant longitude", minimum=-180, maximum=180),
    ]
    # Names, street, job, gender and the full card number are never stored. Gender is
    # left out of the model on purpose: a fraud score must not depend on it.
    optional = ["merchant", "city", "state", "trans_num"]

    groups = [
        Group("Amount", ["amount", "log_amount"], _money("$")),
        Group("Time of day", ["hour", "is_night"], lambda v: f"{int(v['hour']):02d}:00 to {int(v['hour']):02d}:59"),
        Group("Day of week", ["day_of_week"], lambda v: WEEKDAYS[int(v["day_of_week"]) % 7]),
        Group("Customer age", ["age"], lambda v: f"{v['age']:.0f} years"),
        Group("City population", ["log_city_pop"], lambda v: f"{math.expm1(v['log_city_pop']):,.0f}"),
        Group("Distance to merchant", ["distance_km"], lambda v: f"{v['distance_km']:,.0f} km"),
        Group("Merchant category", [f"cat_{c}" for c in SPARKOV_CATEGORIES], _onehot("cat_", SPARKOV_CATEGORIES), onehot=True),
    ]

    def clean_extra(self, raw, out):
        if "merchant" in out:
            # Every merchant name in the Sparkov files starts with "fraud_", fraud or not.
            out["merchant"] = out["merchant"].str.removeprefix("fraud_")
        if "cc_num" in raw.columns:
            out["card_last4"] = raw["cc_num"].astype(str).str.strip().str[-4:]

    def features(self, df):
        when = pd.to_datetime(df["trans_date_trans_time"])
        born = pd.to_datetime(df["dob"])
        amount = df["amt"].astype(float)
        hour = when.dt.hour
        out = pd.DataFrame({
            "amount": amount,
            "log_amount": np.log1p(amount),
            "hour": hour.astype(float),
            "is_night": ((hour >= 22) | (hour < 4)).astype(float),
            "day_of_week": when.dt.dayofweek.astype(float),
            "age": (when - born).dt.days / 365.25,
            "log_city_pop": np.log1p(df["city_pop"].astype(float)),
            "distance_km": _haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"]),
        }, index=df.index)
        for c in SPARKOV_CATEGORIES:
            out[f"cat_{c}"] = (df["category"] == c).astype(float)
        return out[self.feature_names]

    def amounts(self, df):
        return df["amt"].to_numpy(dtype=float)

    def timestamps(self, df, config):
        return list(pd.to_datetime(df["trans_date_trans_time"]).dt.to_pydatetime())


# --- PaySim simulated mobile money ---------------------------------------------

PAYSIM_TYPES = {
    "CASH_IN": "Cash in", "CASH_OUT": "Cash out", "DEBIT": "Debit",
    "PAYMENT": "Payment", "TRANSFER": "Transfer",
}
# PaySim counts hours ("step") from the start of a simulated month with no real date.
# SecureBank shows step 0 as midnight on 1 January 2016.
PAYSIM_START = datetime(2016, 1, 1)


class PaySim(Profile):
    key = "paysim"
    title = "PaySim mobile money"
    description = ("Simulated mobile money transfers over 30 days, modelled on a real service. "
                   "Amounts are in the simulation's own currency units.")
    currency = None
    target = "isFraud"
    fields = [
        Field("step", "Hour of the simulation", minimum=0, help="1 to 744; each step is one hour"),
        Field("type", "Transfer type", kind="choice", options=PAYSIM_TYPES),
        Field("amount", "Amount", minimum=0),
        Field("oldbalanceOrg", "Sender balance before", minimum=0),
        Field("newbalanceOrig", "Sender balance after", minimum=0),
        Field("oldbalanceDest", "Receiver balance before", minimum=0),
        Field("newbalanceDest", "Receiver balance after", minimum=0),
    ]
    optional = ["nameOrig", "nameDest"]

    groups = [
        Group("Amount", ["amount", "log_amount"], lambda v: f"{v['amount']:,.2f}"),
        Group("Transfer type", [f"type_{t}" for t in PAYSIM_TYPES], _onehot("type_", PAYSIM_TYPES), onehot=True),
        Group("Sender balance before", ["oldbalanceOrg"], lambda v: f"{v['oldbalanceOrg']:,.2f}"),
        Group("Sender balance after", ["newbalanceOrig", "empties_account"],
              lambda v: f"{v['newbalanceOrig']:,.2f}" + (" (account emptied)" if v["empties_account"] >= 0.5 else "")),
        Group("Receiver balance before", ["oldbalanceDest"], lambda v: f"{v['oldbalanceDest']:,.2f}"),
        Group("Receiver balance after", ["newbalanceDest"], lambda v: f"{v['newbalanceDest']:,.2f}"),
        Group("Sender balance mismatch", ["orig_balance_error"], lambda v: f"{v['orig_balance_error']:,.2f}"),
        Group("Receiver balance mismatch", ["dest_balance_error"], lambda v: f"{v['dest_balance_error']:,.2f}"),
        Group("Hour of day", ["hour"], lambda v: f"{int(v['hour']):02d}:00"),
    ]

    def features(self, df):
        amount = df["amount"].astype(float)
        out = pd.DataFrame({
            "amount": amount,
            "log_amount": np.log1p(amount),
            "oldbalanceOrg": df["oldbalanceOrg"],
            "newbalanceOrig": df["newbalanceOrig"],
            "empties_account": ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(float),
            "oldbalanceDest": df["oldbalanceDest"],
            "newbalanceDest": df["newbalanceDest"],
            # How far the balances fail to add up; the simulator leaves gaps around fraud.
            "orig_balance_error": df["oldbalanceOrg"] - amount - df["newbalanceOrig"],
            "dest_balance_error": df["oldbalanceDest"] + amount - df["newbalanceDest"],
            "hour": (df["step"] % 24).astype(float),
        }, index=df.index)
        for t in PAYSIM_TYPES:
            out[f"type_{t}"] = (df["type"] == t).astype(float)
        return out[self.feature_names].astype(float)

    def amounts(self, df):
        return df["amount"].to_numpy(dtype=float)

    def timestamps(self, df, config):
        return [PAYSIM_START + timedelta(hours=float(s)) for s in df["step"]]

    def transaction_types(self, df):
        return df["type"].map(PAYSIM_TYPES).to_numpy(dtype=object)


PROFILES = {p.key: p for p in (CreditCard(), Sparkov(), PaySim())}
DEFAULT_PROFILE = "creditcard"


def get_profile(key):
    try:
        return PROFILES[key]
    except KeyError:
        raise ValueError(f"Unknown dataset '{key}'. Choose one of: {', '.join(PROFILES)}") from None


def detect_profile(columns):
    """Pick the profile whose required columns best match a file's header.

    Returns (profile, missing columns). Missing is empty for a full match.
    """
    columns = set(columns)
    best = max(PROFILES.values(), key=lambda p: len(columns & set(p.required_columns)) / len(p.required_columns))
    return best, [c for c in best.required_columns if c not in columns]
