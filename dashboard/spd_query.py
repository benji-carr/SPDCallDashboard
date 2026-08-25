from datetime import date


SPD_CALL_COLUMNS = [
    "cad_event_number",
    "cad_event_original_time_queued",
    "cad_event_arrived_time",
    "cad_event_clearance_description",
    "call_sign_dispatch_id",
    "call_type",
    "priority",
    "initial_call_type",
    "final_call_type",
    "cad_event_response_category",
    "dispatch_precinct",
    "dispatch_sector",
    "dispatch_beat",
    "dispatch_neighborhood",
    "dispatch_latitude",
    "dispatch_longitude",
    "count_of_officers",
    "event_group",
]


def build_spd_call_query_params(
    start_date: str,
    limit: int = 1000,
    offset: int = 0,
) -> dict[str, str | int]:
    if not isinstance(start_date, str):
        raise ValueError("date must be a string")
    
    try:
        parsed_date = date.fromisoformat(start_date)
    except ValueError as error:
        raise ValueError("start_date must be a valid date in YYYY-MM-DD format") from error
    
    if parsed_date.isoformat() != start_date:
        raise ValueError("start_date must be in YYYY-MM-DD format")

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    if limit < 1:
        raise ValueError("limit cannot be less than 1")
    
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError("offset must be an integer")
    if offset < 0:
        raise ValueError("offset cannot be negative")
    
    params = {
        "$select": ",".join(SPD_CALL_COLUMNS),
        "$where": (f"cad_event_original_time_queued >= '{start_date}T00:00:00.000'"),
        "$order": "cad_event_original_time_queued DESC",
        "$limit": limit, 
        "$offset": offset,
    }

    return params

if __name__ == "__main__":
    params = build_spd_call_query_params(
        start_date="2025-01-01",
        limit=500,
        offset=1000,
    )

    print(params)

