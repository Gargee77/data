"""Rating and organization type analytics for the organization dashboard."""

import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    import psycopg2
except ImportError:  # Local CSV testing does not require the Postgres driver.
    psycopg2 = None


FIXED_BUCKETS = ("7D", "30D", "1Y", "All")
EMPTY_MIX = {"non_profit": [], "for_profit": []}
REQUIRED_ORGANIZATION_COLUMNS = {
    "org_id",
    "org_rating",
    "org_type",
    "state_id",
    "created_at",
}
REQUIRED_STATE_COLUMNS = {"state_id", "country_id"}
REQUIRED_COUNTRY_COLUMNS = {"country_id"}
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class RequestError(ValueError):
    """Raised when the request contains an invalid filter."""


def empty_bucket():
    """Return a new empty chart payload."""
    return {
        "rating_distribution": [],
        "organization_mix_trend": {"non_profit": [], "for_profit": []},
    }


def build_response(status_code, payload):
    """Create an API Gateway compatible response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(payload),
    }


def parse_event_body(event):
    """Return the request body as a dictionary."""
    if not event:
        return {}

    body = event.get("body") if isinstance(event, dict) else None
    if body is None:
        body = event
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RequestError("Request body must be valid JSON.") from exc
    if not isinstance(body, dict):
        raise RequestError("Request body must be a JSON object.")
    return body


def parse_date_pair(body, prefix):
    """Validate and return one optional inclusive date range."""
    start_key = f"{prefix}_start_date"
    end_key = f"{prefix}_end_date"
    start_value = body.get(start_key)
    end_value = body.get(end_key)

    start_present = start_key in body
    end_present = end_key in body
    if not start_present and not end_present:
        return None
    if start_present != end_present or not start_value or not end_value:
        raise RequestError(
            f"{start_key} and {end_key} must be supplied together."
        )
    if not isinstance(start_value, str) or not isinstance(end_value, str):
        raise RequestError(f"{start_key} and {end_key} must use YYYY-MM-DD.")

    start_value = start_value.strip()
    end_value = end_value.strip()
    if not DATE_PATTERN.fullmatch(start_value) or not DATE_PATTERN.fullmatch(
        end_value
    ):
        raise RequestError(f"{start_key} and {end_key} must use YYYY-MM-DD.")

    try:
        start = date.fromisoformat(start_value)
        end = date.fromisoformat(end_value)
    except ValueError as exc:
        raise RequestError(
            f"{start_key} and {end_key} must contain valid dates."
        ) from exc
    if start > end:
        raise RequestError(f"{start_key} cannot be after {end_key}.")
    return start, end


def validate_columns(frame, required, filename):
    """Ensure a loaded table contains the columns used by this function."""
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{filename} is missing columns: {', '.join(missing)}")


def load_mock_data(data_dir=None):
    """Load the three local CSV files required by the analytics function."""
    directory = Path(data_dir or os.environ.get("MOCK_DATA_DIR", "."))
    organizations = pd.read_csv(directory / "organizations.csv")
    states = pd.read_csv(directory / "states.csv")
    countries = pd.read_csv(directory / "countries.csv")
    return organizations, states, countries


def checked_identifier(value, label):
    """Validate a schema or table identifier before using it in SQL."""
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid {label} identifier.")
    return value


def get_db_connection():
    """Create a Postgres connection from environment variables."""
    if psycopg2 is None:
        raise RuntimeError(
            "psycopg2 is required when USE_MOCK_DATA is not set to true."
        )

    required = ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing database settings: {', '.join(missing)}")

    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        database=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        port=os.environ.get("DB_PORT", "5432"),
        sslmode=os.environ.get("DB_SSLMODE", "require"),
    )


def query_frame(connection, query):
    """Run a query and return its rows as a DataFrame."""
    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=columns)


def load_postgres_data():
    """Load organization, state, and country fields from Postgres."""
    schema = checked_identifier(
        os.environ.get("DB_SCHEMA", "virginia_dev_saayam_rdbms"), "schema"
    )
    organization_table = checked_identifier(
        os.environ.get("ORGANIZATION_TABLE", "organization"), "organization table"
    )
    state_table = checked_identifier(
        os.environ.get("STATE_TABLE", "state"), "state table"
    )
    country_table = checked_identifier(
        os.environ.get("COUNTRY_TABLE", "country"), "country table"
    )

    connection = get_db_connection()
    try:
        organizations = query_frame(
            connection,
            f"SELECT org_id, org_rating, org_type, state_id, created_at "
            f"FROM {schema}.{organization_table}",
        )
        states = query_frame(
            connection,
            f"SELECT state_id, country_id FROM {schema}.{state_table}",
        )
        countries = query_frame(
            connection,
            f"SELECT country_id, country_code, country_name "
            f"FROM {schema}.{country_table}",
        )
    finally:
        connection.close()
    return organizations, states, countries


def prepare_data(organizations, states, countries):
    """Validate, normalize, and join the three source tables."""
    validate_columns(
        organizations, REQUIRED_ORGANIZATION_COLUMNS, "organizations.csv"
    )
    validate_columns(states, REQUIRED_STATE_COLUMNS, "states.csv")
    validate_columns(countries, REQUIRED_COUNTRY_COLUMNS, "countries.csv")
    if not {"country_code", "country_name"}.intersection(countries.columns):
        raise ValueError(
            "countries.csv must contain country_code or country_name."
        )

    country_fields = ["country_id"]
    country_fields.extend(
        name
        for name in ("country_code", "country_name")
        if name in countries.columns
    )
    joined = organizations.merge(
        states[["state_id", "country_id"]].drop_duplicates("state_id"),
        on="state_id",
        how="left",
    ).merge(
        countries[country_fields].drop_duplicates("country_id"),
        on="country_id",
        how="left",
    )

    joined["created_at"] = pd.to_datetime(
        joined["created_at"], errors="coerce", utc=True
    ).dt.date
    joined["org_rating"] = pd.to_numeric(
        joined["org_rating"], errors="coerce"
    )
    joined["org_type"] = (
        joined["org_type"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    joined["org_type"] = joined["org_type"].replace(
        {"nonprofit": "non_profit", "forprofit": "for_profit"}
    )
    valid_countries = set()
    for column in ("country_code", "country_name"):
        if column in countries.columns:
            valid_countries.update(
                countries[column].dropna().astype("string").map(normalize_country)
            )
    joined.attrs["valid_countries"] = valid_countries
    return joined


def load_data():
    """Load and prepare either local mock data or Postgres data."""
    use_mock_data = os.environ.get("USE_MOCK_DATA", "false").lower() == "true"
    tables = load_mock_data() if use_mock_data else load_postgres_data()
    return prepare_data(*tables)


def normalize_country(value):
    """Normalize a country name or code for a case-insensitive comparison."""
    return " ".join(str(value).strip().upper().replace("_", " ").split())


def filter_country(data, country):
    """Apply a country code or country name filter."""
    requested = normalize_country(country or "ALL")
    if requested in {"ALL", "ALL COUNTRIES"}:
        return data.copy()

    matches = pd.Series(False, index=data.index)
    for column in ("country_code", "country_name"):
        if column in data.columns:
            normalized = data[column].astype("string").map(normalize_country)
            matches = matches | normalized.eq(requested)
    if not matches.any():
        if requested in data.attrs.get("valid_countries", set()):
            return data.iloc[0:0].copy()
        raise RequestError(f"Unknown country: {country}")
    return data.loc[matches].copy()


def rating_distribution(data, start, end):
    """Count organizations by literal rating inside an inclusive window."""
    window = data.loc[
        data["created_at"].notna()
        & data["created_at"].between(start, end, inclusive="both")
        & data["org_rating"].between(1, 5, inclusive="both")
        & data["org_rating"].mod(1).eq(0)
    ]
    if window.empty:
        return []

    counts = window.groupby("org_rating", sort=True).size()
    return [
        {"rating": int(rating), "count": int(count)}
        for rating, count in counts.items()
    ]


def organization_mix_trend(data, start, end, monthly=False):
    """Build sparse cumulative organization counts for each organization type."""
    valid = data.loc[
        data["created_at"].notna()
        & data["org_type"].isin(("non_profit", "for_profit"))
        & data["created_at"].le(end)
    ].copy()
    if valid.empty:
        return {"non_profit": [], "for_profit": []}

    if start is None:
        base_counts = {"non_profit": 0, "for_profit": 0}
        window = valid
    else:
        base_counts = (
            valid.loc[valid["created_at"].lt(start)]
            .groupby("org_type")
            .size()
            .to_dict()
        )
        window = valid.loc[valid["created_at"].ge(start)].copy()

    if window.empty:
        return {"non_profit": [], "for_profit": []}

    date_values = pd.to_datetime(window["created_at"])
    window["period"] = (
        date_values.dt.strftime("%Y-%m")
        if monthly
        else date_values.dt.strftime("%Y-%m-%d")
    )
    grouped = window.groupby(["org_type", "period"], sort=True).size()

    result = {"non_profit": [], "for_profit": []}
    for org_type in result:
        running_total = int(base_counts.get(org_type, 0))
        try:
            period_counts = grouped.loc[org_type]
        except KeyError:
            continue
        for period, count in period_counts.items():
            running_total += int(count)
            result[org_type].append(
                {"period": str(period), "count": running_total}
            )
    return result


def one_year_start(today):
    """Return the same date one year earlier, handling leap day."""
    try:
        return today.replace(year=today.year - 1)
    except ValueError:
        return today.replace(year=today.year - 1, day=28)


def fixed_window(bucket, today):
    """Return the inclusive start and end date for a fixed bucket."""
    starts = {
        "7D": today - timedelta(days=6),
        "30D": today - timedelta(days=29),
        "1Y": one_year_start(today),
        "All": None,
    }
    return starts[bucket], today


def build_analytics(data, body, today=None):
    """Build the fixed-bucket or Custom-only response payload."""
    today = today or datetime.now(timezone.utc).date()
    country_data = filter_country(data, body.get("country", "ALL"))
    rating_range = parse_date_pair(body, "rating")
    type_range = parse_date_pair(body, "type")

    if rating_range or type_range:
        custom = empty_bucket()
        if rating_range:
            custom["rating_distribution"] = rating_distribution(
                country_data, *rating_range
            )
        if type_range:
            custom["organization_mix_trend"] = organization_mix_trend(
                country_data, *type_range, monthly=False
            )
        return {"Custom": custom}

    response = {}
    for bucket in FIXED_BUCKETS:
        start, end = fixed_window(bucket, today)
        rating_start = start or date.min
        response[bucket] = {
            "rating_distribution": rating_distribution(
                country_data, rating_start, end
            ),
            "organization_mix_trend": organization_mix_trend(
                country_data, start, end, monthly=bucket in {"1Y", "All"}
            ),
        }
    response["Custom"] = empty_bucket()
    return response


def lambda_handler(event, context):
    """AWS Lambda entry point for rating and type analytics."""
    try:
        body = parse_event_body(event)
        parse_date_pair(body, "rating")
        parse_date_pair(body, "type")
        return build_response(200, build_analytics(load_data(), body))
    except RequestError as exc:
        return build_response(400, {"error": str(exc)})
    except (OSError, RuntimeError, ValueError, pd.errors.ParserError) as exc:
        return build_response(500, {"error": f"Unable to load analytics data: {exc}"})


if __name__ == "__main__":
    samples = {
        "no body": {},
        "country": {"body": {"country": "USA"}},
        "rating custom": {
            "body": {
                "rating_start_date": "2026-01-01",
                "rating_end_date": "2026-06-30",
            }
        },
        "type custom": {
            "body": {
                "type_start_date": "2025-01-01",
                "type_end_date": "2025-12-31",
            }
        },
        "both custom": {
            "body": {
                "rating_start_date": "2026-01-01",
                "rating_end_date": "2026-06-30",
                "type_start_date": "2025-01-01",
                "type_end_date": "2025-12-31",
            }
        },
    }
    for sample_name, sample_event in samples.items():
        print(f"\n{sample_name}")
        print(json.dumps(lambda_handler(sample_event, None), indent=2))
