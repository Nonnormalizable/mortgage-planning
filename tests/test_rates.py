"""Tests for rate data module."""

import json
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.rates import (
    CacheMetadata,
    RateDataPoint,
    RateDataProvider,
    generate_historical_rate_paths,
    resample_to_monthly,
)


class TestRateDataPoint:
    """Tests for RateDataPoint dataclass."""

    def test_create_rate_data_point(self):
        """Test creating a rate data point."""
        point = RateDataPoint(
            date=date(2024, 1, 15),
            rate=0.045,
            source="SOFR",
        )

        assert point.date == date(2024, 1, 15)
        assert point.rate == 0.045
        assert point.source == "SOFR"


class TestCacheMetadata:
    """Tests for CacheMetadata dataclass."""

    def test_create_cache_metadata(self):
        """Test creating cache metadata."""
        metadata = CacheMetadata(
            fetch_date=datetime(2024, 1, 15, 10, 30),
            series_id="SOFR",
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 15),
        )

        assert metadata.series_id == "SOFR"
        assert metadata.start_date == date(2023, 1, 1)


class TestRateDataProviderCaching:
    """Tests for RateDataProvider caching behavior."""

    def test_cache_directory_created(self):
        """Test that cache directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            RateDataProvider(cache_dir=cache_dir)

            assert cache_dir.exists()

    def test_save_and_load_from_cache(self):
        """Test saving and loading data from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            provider = RateDataProvider(cache_dir=cache_dir)

            # Create test data
            df = pd.DataFrame({
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "rate": [0.045, 0.046, 0.044],
            })

            # Save to cache
            provider._save_to_cache("TEST_SERIES", df)

            # Verify cache file exists
            cache_path = cache_dir / "test_series_cache.json"
            assert cache_path.exists()

            # Load from cache
            loaded_df, metadata = provider._load_from_cache("TEST_SERIES")

            assert loaded_df is not None
            assert len(loaded_df) == 3
            assert metadata is not None
            assert metadata.series_id == "TEST_SERIES"

    def test_expired_cache_returns_none(self):
        """Test that expired cache returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            provider = RateDataProvider(cache_dir=cache_dir)

            # Create expired cache data
            cache_data = {
                "fetch_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "series_id": "TEST_SERIES",
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "data": [
                    {"date": "2024-01-01", "rate": 0.045},
                ],
            }

            cache_path = cache_dir / "test_series_cache.json"
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)

            # Load should return None for expired cache
            loaded_df, metadata = provider._load_from_cache("TEST_SERIES")

            assert loaded_df is None
            assert metadata is None

    def test_invalid_cache_returns_none(self):
        """Test that invalid cache file returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            provider = RateDataProvider(cache_dir=cache_dir)

            # Create invalid cache file
            cache_path = cache_dir / "test_series_cache.json"
            with open(cache_path, "w") as f:
                f.write("invalid json{")

            loaded_df, metadata = provider._load_from_cache("TEST_SERIES")

            assert loaded_df is None
            assert metadata is None

    def test_clear_cache(self):
        """Test clearing the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            provider = RateDataProvider(cache_dir=cache_dir)

            # Create some cache files
            df = pd.DataFrame({
                "date": [date(2024, 1, 1)],
                "rate": [0.045],
            })
            provider._save_to_cache("SOFR", df)
            provider._save_to_cache("FEDFUNDS", df)

            # Verify files exist
            assert (cache_dir / "sofr_cache.json").exists()
            assert (cache_dir / "fedfunds_cache.json").exists()

            # Clear cache
            provider.clear_cache()

            # Verify files are gone
            assert not (cache_dir / "sofr_cache.json").exists()
            assert not (cache_dir / "fedfunds_cache.json").exists()


class TestRateDataProviderFRED:
    """Tests for FRED API integration."""

    def test_fetch_from_fred_success(self):
        """Test successful FRED API fetch."""
        import requests

        with patch.object(requests, "get") as mock_get:
            # Mock response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "observations": [
                    {"date": "2024-01-01", "value": "4.5"},
                    {"date": "2024-01-02", "value": "4.6"},
                ]
            }
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            with tempfile.TemporaryDirectory() as tmpdir:
                provider = RateDataProvider(cache_dir=Path(tmpdir))
                df = provider._fetch_from_fred("SOFR")

                assert len(df) == 2
                # FRED returns percentages, provider converts to decimal
                assert df["rate"].iloc[0] == 0.045
                assert df["rate"].iloc[1] == 0.046

    def test_fetch_from_fred_handles_missing_values(self):
        """Test that FRED fetch handles missing values."""
        import requests

        with patch.object(requests, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "observations": [
                    {"date": "2024-01-01", "value": "4.5"},
                    {"date": "2024-01-02", "value": "."},  # Missing value
                    {"date": "2024-01-03", "value": "4.6"},
                ]
            }
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            with tempfile.TemporaryDirectory() as tmpdir:
                provider = RateDataProvider(cache_dir=Path(tmpdir))
                df = provider._fetch_from_fred("SOFR")

                # Should skip the missing value
                assert len(df) == 2

    def test_fetch_from_fred_error(self):
        """Test FRED fetch error handling."""
        import requests

        with patch.object(requests, "get") as mock_get:
            mock_get.side_effect = requests.RequestException("Network error")

            with tempfile.TemporaryDirectory() as tmpdir:
                provider = RateDataProvider(cache_dir=Path(tmpdir))

                with pytest.raises(RuntimeError, match="Failed to fetch data from FRED"):
                    provider._fetch_from_fred("SOFR")


class TestRateDataProviderGetters:
    """Tests for rate data getter methods."""

    @patch.object(RateDataProvider, "_get_series_data")
    def test_get_current_sofr(self, mock_get_series):
        """Test getting current SOFR rate."""
        mock_get_series.return_value = pd.DataFrame({
            "date": [date(2024, 1, 10), date(2024, 1, 11), date(2024, 1, 12)],
            "rate": [0.045, 0.046, 0.047],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = RateDataProvider(cache_dir=Path(tmpdir))
            rate, obs_date = provider.get_current_sofr()

            assert rate == 0.047  # Most recent
            assert obs_date == date(2024, 1, 12)

    @patch.object(RateDataProvider, "_get_series_data")
    def test_get_current_mortgage_rate(self, mock_get_series):
        """Test getting current mortgage rate."""
        mock_get_series.return_value = pd.DataFrame({
            "date": [date(2024, 1, 5), date(2024, 1, 12)],
            "rate": [0.068, 0.069],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = RateDataProvider(cache_dir=Path(tmpdir))
            rate, obs_date = provider.get_current_mortgage_rate()

            assert rate == 0.069
            assert obs_date == date(2024, 1, 12)


class TestRateDataProviderHistorical:
    """Tests for historical data methods."""

    @patch.object(RateDataProvider, "_get_series_data")
    def test_get_historical_index_rates(self, mock_get_series):
        """Test getting historical index rates."""
        # Mock Fed Funds data (pre-2018)
        fed_funds_df = pd.DataFrame({
            "date": pd.date_range("2017-01-01", periods=12, freq="ME").date,
            "rate": [0.04] * 12,
        })

        # Mock SOFR data (2018+)
        sofr_df = pd.DataFrame({
            "date": pd.date_range("2018-04-03", periods=24, freq="D").date,
            "rate": [0.045] * 24,
        })

        def side_effect(series_id, start_date, end_date):
            if series_id == "FEDFUNDS":
                return fed_funds_df
            else:  # SOFR
                return sofr_df

        mock_get_series.side_effect = side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = RateDataProvider(cache_dir=Path(tmpdir))
            df = provider.get_historical_index_rates(2017, 2018)

            assert len(df) > 0
            assert "date" in df.columns
            assert "rate" in df.columns
            assert "source" in df.columns

    @patch.object(RateDataProvider, "get_historical_index_rates")
    def test_get_rate_statistics(self, mock_get_historical):
        """Test getting rate statistics."""
        mock_get_historical.return_value = pd.DataFrame({
            "date": [date(2024, 1, i) for i in range(1, 11)],
            "rate": [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11],
            "source": ["TEST"] * 10,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = RateDataProvider(cache_dir=Path(tmpdir))
            stats = provider.get_rate_statistics()

            assert "min" in stats
            assert "max" in stats
            assert "mean" in stats
            assert "median" in stats
            assert stats["min"] == 0.02
            assert stats["max"] == 0.11

    @patch.object(RateDataProvider, "get_historical_index_rates")
    def test_get_current_rate_percentile(self, mock_get_historical):
        """Test calculating current rate percentile."""
        mock_get_historical.return_value = pd.DataFrame({
            "date": [date(2024, 1, i) for i in range(1, 11)],
            "rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
            "source": ["TEST"] * 10,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = RateDataProvider(cache_dir=Path(tmpdir))

            # 0.05 is at the 50th percentile (5 values below it)
            percentile = provider.get_current_rate_percentile(0.05)
            assert percentile == 40.0  # 4 values below 0.05

            # 0.10 should be near top
            percentile = provider.get_current_rate_percentile(0.10)
            assert percentile == 90.0


class TestResampleToMonthly:
    """Tests for resample_to_monthly function."""

    def test_resample_daily_to_monthly_last(self):
        """Test resampling daily data to monthly using last value."""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "rate": range(60),
        })

        monthly = resample_to_monthly(df, method="last")

        # Should have 2 months (Jan and Feb)
        assert len(monthly) == 2

    def test_resample_daily_to_monthly_mean(self):
        """Test resampling daily data to monthly using mean."""
        dates = pd.date_range("2024-01-01", periods=31, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "rate": [0.05] * 31,
        })

        monthly = resample_to_monthly(df, method="mean")

        assert len(monthly) >= 1
        assert monthly[0] == pytest.approx(0.05)

    def test_resample_invalid_method(self):
        """Test that invalid method raises error."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "rate": [0.05],
        })

        with pytest.raises(ValueError, match="Unknown method"):
            resample_to_monthly(df, method="invalid")


class TestGenerateHistoricalRatePaths:
    """Tests for generate_historical_rate_paths function."""

    def test_generate_paths_correct_shape(self):
        """Test that generated paths have correct shape."""
        # 100 months of historical data, 30-month horizon
        historical_rates = np.linspace(0.02, 0.06, 100)

        paths = generate_historical_rate_paths(historical_rates, time_horizon_months=30)

        # Should have 70 paths (100 - 30)
        assert paths.shape == (70, 30)

    def test_generate_paths_content(self):
        """Test that paths contain correct historical windows."""
        historical_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])

        paths = generate_historical_rate_paths(historical_rates, time_horizon_months=3)

        # Should have 5 paths (8 - 3)
        assert paths.shape == (5, 3)

        # First path should be [0.01, 0.02, 0.03]
        np.testing.assert_array_equal(paths[0], [0.01, 0.02, 0.03])

        # Second path should be [0.02, 0.03, 0.04]
        np.testing.assert_array_equal(paths[1], [0.02, 0.03, 0.04])

        # Last path should be [0.05, 0.06, 0.07]
        np.testing.assert_array_equal(paths[4], [0.05, 0.06, 0.07])

    def test_generate_paths_insufficient_data(self):
        """Test error when insufficient historical data."""
        historical_rates = np.array([0.01, 0.02, 0.03])

        with pytest.raises(ValueError, match="Insufficient historical data"):
            generate_historical_rate_paths(historical_rates, time_horizon_months=5)

    def test_generate_paths_exact_length(self):
        """Test when historical data is exactly one more than horizon."""
        historical_rates = np.array([0.01, 0.02, 0.03, 0.04])

        paths = generate_historical_rate_paths(historical_rates, time_horizon_months=3)

        # Should have exactly 1 path
        assert paths.shape == (1, 3)
        np.testing.assert_array_equal(paths[0], [0.01, 0.02, 0.03])
