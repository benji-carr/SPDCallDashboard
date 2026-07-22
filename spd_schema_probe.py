from typing import Any

import pandas as pd
import requests


SPD_CALL_ENDPOINT = (
    "https://data.seattle.gov/"
    "resource/33kz-ixgy.json"
)


def fetch_sample_records(
    limit: int = 25,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    params = {
        "$limit": limit,
    }

    response = requests.get(
        SPD_CALL_ENDPOINT,
        params=params,
        timeout=timeout,
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise ValueError("top-level response must be a list")

    if not all(isinstance(item, dict) for item in data):
        raise ValueError("every record must be a dictionary")

    return data


def inspect_records(records: list[dict[str, Any]]) -> None:
    print(f"Number of records: {len(records)}")

    all_keys = sorted(
        {
            key
            for record in records
            for key in record.keys()
        }
    )

    print("\nColumns found:")
    for key in all_keys:
        print(f"- {key}")

    df = pd.DataFrame.from_records(records)

    print("\nDataFrame preview:")
    print(df.head())

    print("\nDataFrame dtypes:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())


if __name__ == "__main__":
    records = fetch_sample_records(limit=25)
    inspect_records(records)