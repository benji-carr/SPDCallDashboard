import itertools

import pandas as pd

from spd_config import TARGET_IMPORTANCE_BINS


VIOLENT_OR_PERSON_CRIME_GROUPS = [
    "assault",
    "domestic disturbance/violence",
    "homicide",
    "kidnap",
    "rape",
    "robbery",
    "sex offenses (non-rape)",
]

DRUG_RELATED_GROUPS = [
    "narcotics",
]

PROPERTY_OR_NONVIOLENT_GROUPS = [
    "burglary",
    "theft",
    "car prowl",
    "fraud",
    "trespass",
]

LOWER_PUBLIC_SAFETY_GROUPS = [
    "traffic",
    "disturbance",
    "misc. misdemeanors & violations",
    "warrant services/order",
    "person down/injury",
    "suspicious circumstances",
]


def assign_event_importance_bin(event_group: object) -> str:
    if pd.isna(event_group):
        return "unknown / unclassified"

    event_group = str(event_group).strip().lower()

    if event_group in VIOLENT_OR_PERSON_CRIME_GROUPS:
        return "violent/person crime"

    if event_group in DRUG_RELATED_GROUPS:
        return "drug-related"

    if event_group in PROPERTY_OR_NONVIOLENT_GROUPS:
        return "property/nonviolent"

    if event_group in LOWER_PUBLIC_SAFETY_GROUPS:
        return "lower public-safety urgency"

    return "other / unclassified"


def ensure_event_importance_bin(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()

    if "event_group" not in out.columns:
        raise ValueError("DataFrame is missing required column: event_group")

    if "event_importance_bin" not in out.columns:
        out["event_group"] = (
            out["event_group"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        out["event_importance_bin"] = (
            out["event_group"]
            .apply(assign_event_importance_bin)
        )

    return out


def make_bin_combo_label(bin_combo: list[str]) -> str:
    if len(bin_combo) == len(TARGET_IMPORTANCE_BINS):
        return "All selected bins"

    return " + ".join(bin_combo)


def make_bin_combinations() -> list[list[str]]:
    bin_combinations = []

    for r in range(1, len(TARGET_IMPORTANCE_BINS) + 1):
        for combo in itertools.combinations(TARGET_IMPORTANCE_BINS, r):
            bin_combinations.append(list(combo))

    return [
        TARGET_IMPORTANCE_BINS,
        *[
            combo
            for combo in bin_combinations
            if combo != TARGET_IMPORTANCE_BINS
        ],
    ]


def encode_bin_combo(bin_combo: list[str]) -> str:
    return "||".join(bin_combo)


def decode_bin_combo(value: str | None) -> list[str]:
    if value is None:
        return TARGET_IMPORTANCE_BINS

    return value.split("||")


def make_bin_dropdown_options() -> list[dict[str, str]]:
    return [
        {
            "label": make_bin_combo_label(combo),
            "value": encode_bin_combo(combo),
        }
        for combo in make_bin_combinations()
    ]