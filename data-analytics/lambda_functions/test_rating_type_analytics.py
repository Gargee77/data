"""Tests for the Rating & Type organization analytics Lambda."""

import json
from datetime import date

import pandas as pd
import pytest

import rating_type_analytics as analytics


TODAY = date(2026, 9, 29)


def prepared_data(rows=None):
    """Build a small joined dataset with two countries."""
    rows = rows if rows is not None else [
        {
            "org_id": "old-us",
            "org_rating": 1,
            "org_type": "non_profit",
            "state_id": "NY",
            "created_at": "2024-01-10",
        },
        {
            "org_id": "year-us",
            "org_rating": 2,
            "org_type": "Non-Profit",
            "state_id": "NY",
            "created_at": "2025-10-01",
        },
        {
            "org_id": "month-us",
            "org_rating": 4,
            "org_type": "for_profit",
            "state_id": "NY",
            "created_at": "2026-09-01",
        },
        {
            "org_id": "week-us",
            "org_rating": 5,
            "org_type": "non_profit",
            "state_id": "NY",
            "created_at": "2026-09-25",
        },
        {
            "org_id": "canada",
            "org_rating": 3,
            "org_type": "for_profit",
            "state_id": "ON",
            "created_at": "2026-09-28",
        },
        {
            "org_id": "today-us",
            "org_rating": 4,
            "org_type": "For-profit",
            "state_id": "NY",
            "created_at": "2026-09-29",
        },
    ]
    organizations = pd.DataFrame(
        rows,
        columns=["org_id", "org_rating", "org_type", "state_id", "created_at"],
    )
    states = pd.DataFrame(
        [
            {"state_id": "NY", "country_id": 1},
            {"state_id": "ON", "country_id": 2},
        ]
    )
    countries = pd.DataFrame(
        [
            {"country_id": 1, "country_code": "USA", "country_name": "United States"},
            {"country_id": 2, "country_code": "CAN", "country_name": "Canada"},
        ]
    )
    return analytics.prepare_data(organizations, states, countries)


def decode(response):
    """Decode a Lambda response body."""
    return json.loads(response["body"])


def test_no_custom_range_returns_exactly_five_buckets():
    result = analytics.build_analytics(prepared_data(), {}, today=TODAY)

    assert list(result) == ["7D", "30D", "1Y", "All", "Custom"]
    assert result["Custom"] == analytics.empty_bucket()
    assert all(
        set(bucket) == {"rating_distribution", "organization_mix_trend"}
        for bucket in result.values()
    )


def test_country_filter_accepts_code_or_name():
    by_code = analytics.build_analytics(
        prepared_data(), {"country": " can "}, today=TODAY
    )
    by_name = analytics.build_analytics(
        prepared_data(), {"country": "Canada"}, today=TODAY
    )

    assert by_code == by_name
    assert by_code["All"]["rating_distribution"] == [
        {"rating": 3, "count": 1}
    ]
    assert by_code["All"]["organization_mix_trend"]["non_profit"] == []


def test_rating_custom_range_only_populates_rating_chart():
    result = analytics.build_analytics(
        prepared_data(),
        {
            "country": "USA",
            "rating_start_date": "2026-09-25",
            "rating_end_date": "2026-09-29",
        },
        today=TODAY,
    )

    assert list(result) == ["Custom"]
    assert result["Custom"]["rating_distribution"] == [
        {"rating": 4, "count": 1},
        {"rating": 5, "count": 1},
    ]
    assert result["Custom"]["organization_mix_trend"] == analytics.EMPTY_MIX


def test_type_custom_range_only_populates_cumulative_type_chart():
    result = analytics.build_analytics(
        prepared_data(),
        {
            "country": "USA",
            "type_start_date": "2026-09-25",
            "type_end_date": "2026-09-29",
        },
        today=TODAY,
    )

    assert result["Custom"]["rating_distribution"] == []
    assert result["Custom"]["organization_mix_trend"] == {
        "non_profit": [{"period": "2026-09-25", "count": 3}],
        "for_profit": [{"period": "2026-09-29", "count": 2}],
    }


def test_both_custom_ranges_populate_independently():
    result = analytics.build_analytics(
        prepared_data(),
        {
            "rating_start_date": "2026-09-28",
            "rating_end_date": "2026-09-29",
            "type_start_date": "2026-09-01",
            "type_end_date": "2026-09-25",
        },
        today=TODAY,
    )

    assert list(result) == ["Custom"]
    assert result["Custom"]["rating_distribution"] == [
        {"rating": 3, "count": 1},
        {"rating": 4, "count": 1},
    ]
    assert result["Custom"]["organization_mix_trend"]["for_profit"] == [
        {"period": "2026-09-01", "count": 1}
    ]


@pytest.mark.parametrize(
    "body, message",
    [
        ({"rating_start_date": "2026-01-01"}, "must be supplied together"),
        ({"rating_start_date": "", "rating_end_date": ""}, "must be supplied together"),
        (
            {"type_start_date": "2026/01/01", "type_end_date": "2026-02-01"},
            "must use YYYY-MM-DD",
        ),
        (
            {"rating_start_date": "2026-03-01", "rating_end_date": "2026-02-01"},
            "cannot be after",
        ),
    ],
)
def test_bad_date_pairs_return_clear_400(monkeypatch, body, message):
    monkeypatch.setattr(analytics, "load_data", prepared_data)

    response = analytics.lambda_handler({"body": body}, None)

    assert response["statusCode"] == 400
    assert message in decode(response)["error"]


def test_invalid_json_returns_400():
    response = analytics.lambda_handler({"body": "{"}, None)

    assert response["statusCode"] == 400
    assert decode(response) == {"error": "Request body must be valid JSON."}


def test_unknown_country_returns_400(monkeypatch):
    monkeypatch.setattr(analytics, "load_data", prepared_data)

    response = analytics.lambda_handler({"body": {"country": "XYZ"}}, None)

    assert response["statusCode"] == 400
    assert decode(response) == {"error": "Unknown country: XYZ"}


def test_fixed_type_series_are_sparse_cumulative_and_non_decreasing():
    result = analytics.build_analytics(
        prepared_data(), {"country": "USA"}, today=TODAY
    )

    for bucket in ("7D", "30D", "1Y", "All"):
        trend = result[bucket]["organization_mix_trend"]
        for series in trend.values():
            counts = [point["count"] for point in series]
            assert counts == sorted(counts)
    assert result["30D"]["organization_mix_trend"]["for_profit"] == [
        {"period": "2026-09-01", "count": 1},
        {"period": "2026-09-29", "count": 2},
    ]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [
            {
                "org_id": "only-row",
                "org_rating": 5,
                "org_type": "non_profit",
                "state_id": "NY",
                "created_at": "2026-09-29",
            }
        ],
    ],
)
def test_empty_and_single_row_organization_data_are_safe(rows):
    result = analytics.build_analytics(prepared_data(rows), {}, today=TODAY)

    assert list(result) == ["7D", "30D", "1Y", "All", "Custom"]


def test_real_country_with_no_organizations_returns_empty_charts():
    result = analytics.build_analytics(
        prepared_data([]), {"country": "USA"}, today=TODAY
    )

    assert result["All"] == analytics.empty_bucket()


def test_mock_csv_path_runs_through_lambda(tmp_path, monkeypatch):
    data = prepared_data()
    data[["org_id", "org_rating", "org_type", "state_id", "created_at"]].to_csv(
        tmp_path / "organizations.csv", index=False
    )
    pd.DataFrame(
        [
            {"state_id": "NY", "country_id": 1},
            {"state_id": "ON", "country_id": 2},
        ]
    ).to_csv(tmp_path / "states.csv", index=False)
    pd.DataFrame(
        [
            {"country_id": 1, "country_code": "USA", "country_name": "United States"},
            {"country_id": 2, "country_code": "CAN", "country_name": "Canada"},
        ]
    ).to_csv(tmp_path / "countries.csv", index=False)
    monkeypatch.setenv("USE_MOCK_DATA", "true")
    monkeypatch.setenv("MOCK_DATA_DIR", str(tmp_path))

    response = analytics.lambda_handler(
        {
            "body": {
                "country": "USA",
                "rating_start_date": "2026-09-25",
                "rating_end_date": "2026-09-29",
            }
        },
        None,
    )

    assert response["statusCode"] == 200
    assert list(decode(response)) == ["Custom"]
