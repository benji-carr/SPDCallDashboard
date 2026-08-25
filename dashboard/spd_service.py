from typing import Any

import pandas as pd

from spd_client import fetch_spd_call_page
from spd_data import spd_calls_to_dataframe


def load_spd_call_dataset(
    start_date: str,
    page_size: int = 1000,
    max_pages: int | None = 3,
    timeout: float = 10.0,
) -> pd.DataFrame:
    
    if isinstance(page_size, bool) or not isinstance(page_size, int):
        raise ValueError("page_size must be an integer")
    if page_size < 1:
        raise ValueError("page_size cannot be less than 1")
    
    if max_pages is not None:
        if isinstance(max_pages, bool) or not isinstance(max_pages, int):
            raise ValueError("max_pages must be an integer")
        if max_pages < 1:
            raise ValueError("max_pages cannot be less than 1")

    all_records = []
    offset = 0 
    pages_fetched = 0

    while True:
        page = fetch_spd_call_page(
            start_date=start_date,
            limit=page_size,
            offset=offset,
            timeout=timeout,
        )
        all_records.extend(page)
        pages_fetched += 1

        if len(page) < page_size:
            break
        if pages_fetched == max_pages:
            break

        offset += page_size

    return spd_calls_to_dataframe(all_records)


if __name__ == "__main__":
    df = load_spd_call_dataset(
        start_date="2025-01-01",
        page_size=25,
        max_pages=2,
    )

    print(df.head())
    print(f"\nRows: {len(df)}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())


