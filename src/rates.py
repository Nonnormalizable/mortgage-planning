"""Historical interest rate data provider using FRED API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

# FRED API configuration
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED Series IDs
SERIES_SOFR = "SOFR"  # Daily SOFR (2018-04-03 to present)
SERIES_FED_FUNDS = "FEDFUNDS"  # Monthly Fed Funds Rate (1954-07 to present)
SERIES_MORTGAGE_30Y = "MORTGAGE30US"  # Weekly 30-Year Fixed Mortgage Rate (1971-04 to present)

# Cache configuration
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mortgage-planning"
CACHE_EXPIRY_CURRENT = timedelta(hours=24)  # Current rates expire after 24 hours
CACHE_EXPIRY_HISTORICAL = timedelta(days=7)  # Historical data expires after 7 days

# Bundled data path (fallback when API unavailable)
BUNDLED_DATA_PATH = Path(__file__).parent.parent / "data" / "historical_rates.json"


@dataclass
class RateDataPoint:
    """A single rate data point."""

    date: date
    rate: float
    source: str  # "SOFR", "FED_FUNDS", "MORTGAGE30"


@dataclass
class CacheMetadata:
    """Metadata for cached data."""

    fetch_date: datetime
    series_id: str
    start_date: date
    end_date: date


class RateDataProvider:
    """Fetches and caches historical rate data from FRED."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        api_key: str | None = None,
    ):
        """Initialize the rate data provider.

        Args:
            cache_dir: Directory for caching data. Defaults to ~/.cache/mortgage-planning/
            api_key: Optional FRED API key for higher rate limits
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self._cache: dict[str, pd.DataFrame] = {}

    def _get_cache_path(self, series_id: str) -> Path:
        """Get the cache file path for a series."""
        return self.cache_dir / f"{series_id.lower()}_cache.json"

    def _load_from_cache(self, series_id: str) -> tuple[pd.DataFrame | None, CacheMetadata | None]:
        """Load data from cache if available and not expired.

        Returns:
            Tuple of (DataFrame, metadata) or (None, None) if cache miss
        """
        cache_path = self._get_cache_path(series_id)

        if not cache_path.exists():
            return None, None

        try:
            with open(cache_path) as f:
                cache_data = json.load(f)

            metadata = CacheMetadata(
                fetch_date=datetime.fromisoformat(cache_data["fetch_date"]),
                series_id=cache_data["series_id"],
                start_date=date.fromisoformat(cache_data["start_date"]),
                end_date=date.fromisoformat(cache_data["end_date"]),
            )

            # Check if cache is expired
            age = datetime.now() - metadata.fetch_date
            if age > CACHE_EXPIRY_HISTORICAL:
                return None, None

            # Parse the data
            df = pd.DataFrame(cache_data["data"])
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df, metadata

        except (json.JSONDecodeError, KeyError, ValueError):
            # Invalid cache file
            return None, None

    def _save_to_cache(self, series_id: str, df: pd.DataFrame) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(series_id)

        cache_data = {
            "fetch_date": datetime.now().isoformat(),
            "series_id": series_id,
            "start_date": df["date"].min().isoformat(),
            "end_date": df["date"].max().isoformat(),
            "data": [
                {"date": row["date"].isoformat(), "rate": row["rate"]}
                for _, row in df.iterrows()
            ],
        }

        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

    def _fetch_from_fred(
        self,
        series_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Fetch data from FRED API.

        Args:
            series_id: FRED series ID
            start_date: Start date for data (optional)
            end_date: End date for data (optional)

        Returns:
            DataFrame with 'date' and 'rate' columns

        Raises:
            RuntimeError: If API request fails
        """
        import requests

        params = {
            "series_id": series_id,
            "file_type": "json",
        }

        if self.api_key:
            params["api_key"] = self.api_key

        if start_date:
            params["observation_start"] = start_date.isoformat()
        if end_date:
            params["observation_end"] = end_date.isoformat()

        try:
            response = requests.get(FRED_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            observations = data.get("observations", [])

            if not observations:
                raise RuntimeError(f"No data returned for series {series_id}")

            # Parse observations
            records = []
            for obs in observations:
                try:
                    rate_value = float(obs["value"])
                    # FRED returns rates as percentages (e.g., 4.5 for 4.5%)
                    # Convert to decimal form
                    rate_decimal = rate_value / 100.0
                    records.append({
                        "date": datetime.strptime(obs["date"], "%Y-%m-%d").date(),
                        "rate": rate_decimal,
                    })
                except (ValueError, KeyError):
                    # Skip invalid observations (e.g., "." for missing data)
                    continue

            if not records:
                raise RuntimeError(f"No valid observations for series {series_id}")

            return pd.DataFrame(records)

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch data from FRED: {e}") from e

    def _get_series_data(
        self,
        series_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Get data for a series, using cache if available.

        Args:
            series_id: FRED series ID
            start_date: Start date for data
            end_date: End date for data
            use_cache: Whether to use cached data

        Returns:
            DataFrame with 'date' and 'rate' columns
        """
        if use_cache:
            cached_df, metadata = self._load_from_cache(series_id)
            if cached_df is not None:
                # Filter to requested date range
                df = cached_df.copy()
                if start_date:
                    df = df[df["date"] >= start_date]
                if end_date:
                    df = df[df["date"] <= end_date]
                if len(df) > 0:
                    return df

        # Fetch fresh data
        df = self._fetch_from_fred(series_id, start_date, end_date)

        if use_cache:
            self._save_to_cache(series_id, df)

        return df

    def get_current_sofr(self) -> tuple[float, date]:
        """Get the current SOFR rate.

        Returns:
            Tuple of (rate as decimal, observation date)

        Raises:
            RuntimeError: If unable to fetch rate
        """
        # Get recent data
        end_date = date.today()
        start_date = end_date - timedelta(days=14)  # Look back 2 weeks

        try:
            df = self._get_series_data(SERIES_SOFR, start_date, end_date)
            latest = df.loc[df["date"].idxmax()]
            return float(latest["rate"]), latest["date"]
        except RuntimeError:
            # Try cache even if expired
            cached_df, _ = self._load_from_cache(SERIES_SOFR)
            if cached_df is not None and len(cached_df) > 0:
                latest = cached_df.loc[cached_df["date"].idxmax()]
                return float(latest["rate"]), latest["date"]
            raise

    def get_current_mortgage_rate(self) -> tuple[float, date]:
        """Get the current 30-year fixed mortgage rate.

        Returns:
            Tuple of (rate as decimal, observation date)

        Raises:
            RuntimeError: If unable to fetch rate
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=30)  # Look back 1 month

        try:
            df = self._get_series_data(SERIES_MORTGAGE_30Y, start_date, end_date)
            latest = df.loc[df["date"].idxmax()]
            return float(latest["rate"]), latest["date"]
        except RuntimeError:
            cached_df, _ = self._load_from_cache(SERIES_MORTGAGE_30Y)
            if cached_df is not None and len(cached_df) > 0:
                latest = cached_df.loc[cached_df["date"].idxmax()]
                return float(latest["rate"]), latest["date"]
            raise

    def get_historical_index_rates(
        self,
        start_year: int = 1975,
        end_year: int | None = None,
    ) -> pd.DataFrame:
        """Get historical index rates (SOFR + Fed Funds for earlier data).

        Uses bundled historical data as fallback when FRED API is unavailable.

        Args:
            start_year: Start year (default 1975)
            end_year: End year (default current year)

        Returns:
            DataFrame with 'date' and 'rate' columns, monthly data
        """
        if end_year is None:
            end_year = date.today().year

        # Try to load from bundled data first (always available, no API needed)
        try:
            return self._load_bundled_historical_data(start_year, end_year)
        except Exception:
            pass  # Fall through to API if bundled data fails

        # Try FRED API
        try:
            return self._fetch_historical_from_fred(start_year, end_year)
        except RuntimeError:
            # Last resort: try bundled data again with less strict error handling
            return self._load_bundled_historical_data(start_year, end_year)

    def _load_bundled_historical_data(
        self,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """Load historical data from bundled JSON file.

        Args:
            start_year: Start year to filter
            end_year: End year to filter

        Returns:
            DataFrame with 'date', 'rate', and 'source' columns
        """
        if not BUNDLED_DATA_PATH.exists():
            raise RuntimeError(f"Bundled data file not found: {BUNDLED_DATA_PATH}")

        with open(BUNDLED_DATA_PATH) as f:
            data = json.load(f)

        records = []
        for entry in data["rates"]:
            # Parse date from "YYYY-MM" format
            year, month = entry["date"].split("-")
            entry_date = date(int(year), int(month), 1)
            records.append({
                "date": entry_date,
                "rate": entry["rate"],
                "source": "BUNDLED",
            })

        df = pd.DataFrame(records)

        # Filter to requested date range
        start_date = date(start_year, 1, 1)
        end_date = date(end_year, 12, 31)
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

        return df.sort_values("date").reset_index(drop=True)

    def _fetch_historical_from_fred(
        self,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """Fetch historical rates from FRED API.

        Args:
            start_year: Start year
            end_year: End year

        Returns:
            DataFrame with 'date', 'rate', and 'source' columns
        """
        start_date = date(start_year, 1, 1)
        end_date = date(end_year, 12, 31)

        # SOFR available from April 2018
        sofr_start = date(2018, 4, 3)

        # Get Fed Funds data for pre-SOFR era
        fed_funds_df = self._get_series_data(
            SERIES_FED_FUNDS,
            start_date,
            min(end_date, sofr_start - timedelta(days=1)),
        )
        fed_funds_df["source"] = "FED_FUNDS"

        # Get SOFR data for 2018+
        if end_date >= sofr_start:
            try:
                sofr_df = self._get_series_data(
                    SERIES_SOFR,
                    max(start_date, sofr_start),
                    end_date,
                )
                sofr_df["source"] = "SOFR"

                # Resample SOFR to monthly (last observation of each month)
                sofr_df["date"] = pd.to_datetime(sofr_df["date"])
                sofr_monthly = sofr_df.set_index("date").resample("ME").last().reset_index()
                sofr_monthly["date"] = sofr_monthly["date"].dt.date

                # Combine datasets
                combined = pd.concat([fed_funds_df, sofr_monthly], ignore_index=True)
            except RuntimeError:
                # Fall back to Fed Funds only if SOFR unavailable
                combined = fed_funds_df
        else:
            combined = fed_funds_df

        # Sort by date
        combined = combined.sort_values("date").reset_index(drop=True)

        # Resample to ensure monthly frequency (forward fill gaps)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.set_index("date")
        combined = combined.resample("ME").last().ffill().reset_index()
        combined["date"] = combined["date"].dt.date

        return combined[["date", "rate", "source"]]

    def get_historical_mortgage_rates(
        self,
        start_year: int = 1975,
        end_year: int | None = None,
    ) -> pd.DataFrame:
        """Get historical 30-year fixed mortgage rates.

        Args:
            start_year: Start year (default 1975)
            end_year: End year (default current year)

        Returns:
            DataFrame with 'date' and 'rate' columns, monthly data
        """
        if end_year is None:
            end_year = date.today().year

        start_date = date(start_year, 1, 1)
        end_date = date(end_year, 12, 31)

        # MORTGAGE30US available from April 1971
        df = self._get_series_data(SERIES_MORTGAGE_30Y, start_date, end_date)
        df["source"] = "MORTGAGE30"

        # Resample to monthly (average of weekly observations)
        df["date"] = pd.to_datetime(df["date"])
        monthly = df.set_index("date").resample("ME").mean().reset_index()
        monthly["date"] = monthly["date"].dt.date
        monthly["source"] = "MORTGAGE30"

        return monthly[["date", "rate", "source"]]

    def get_rate_statistics(
        self,
        start_year: int = 1975,
        end_year: int | None = None,
    ) -> dict:
        """Get summary statistics for historical rates.

        Returns:
            Dictionary with rate statistics
        """
        df = self.get_historical_index_rates(start_year, end_year)

        rates = df["rate"].values
        return {
            "min": float(np.min(rates)),
            "max": float(np.max(rates)),
            "mean": float(np.mean(rates)),
            "median": float(np.median(rates)),
            "std": float(np.std(rates)),
            "p25": float(np.percentile(rates, 25)),
            "p75": float(np.percentile(rates, 75)),
        }

    def get_current_rate_percentile(self, current_rate: float) -> float:
        """Calculate what percentile the current rate falls at historically.

        Args:
            current_rate: Current rate as decimal (e.g., 0.045 for 4.5%)

        Returns:
            Percentile (0-100)
        """
        df = self.get_historical_index_rates()
        rates = df["rate"].values
        percentile = (rates < current_rate).sum() / len(rates) * 100
        return float(percentile)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        for series_id in [SERIES_SOFR, SERIES_FED_FUNDS, SERIES_MORTGAGE_30Y]:
            cache_path = self._get_cache_path(series_id)
            if cache_path.exists():
                cache_path.unlink()
        self._cache.clear()


def resample_to_monthly(df: pd.DataFrame, method: str = "last") -> np.ndarray:
    """Resample rate data to monthly frequency.

    Args:
        df: DataFrame with 'date' and 'rate' columns
        method: Resampling method ('last', 'mean', 'first')

    Returns:
        Array of monthly rates
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    if method == "last":
        monthly = df.resample("ME").last()
    elif method == "mean":
        monthly = df.resample("ME").mean()
    elif method == "first":
        monthly = df.resample("ME").first()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Forward fill any gaps
    monthly = monthly.ffill()

    return monthly["rate"].values


def generate_historical_rate_paths(
    historical_rates: np.ndarray,
    time_horizon_months: int,
) -> np.ndarray:
    """Generate rate paths from historical data using rolling windows.

    Each "simulation" is a different starting year from history.

    Args:
        historical_rates: Array of monthly historical rates
        time_horizon_months: Length of each path in months

    Returns:
        Array of shape (n_paths, time_horizon_months) where n_paths
        is determined by available data
    """
    n_available = len(historical_rates) - time_horizon_months

    if n_available <= 0:
        raise ValueError(
            f"Insufficient historical data: have {len(historical_rates)} months, "
            f"need at least {time_horizon_months + 1} for the requested horizon"
        )

    paths = np.zeros((n_available, time_horizon_months))

    for i in range(n_available):
        paths[i, :] = historical_rates[i : i + time_horizon_months]

    return paths


# Singleton instance for convenience
_default_provider: RateDataProvider | None = None


def get_rate_provider() -> RateDataProvider:
    """Get the default rate provider instance."""
    global _default_provider
    if _default_provider is None:
        _default_provider = RateDataProvider()
    return _default_provider
