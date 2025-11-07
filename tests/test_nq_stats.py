"""
Unit tests for NQ Stats Analyzer

Tests core functionality of NQ Stats probability-based analysis
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Direct import to avoid dependency issues
import importlib.util

# Load NQ Stats analyzer
analyzer_path = src_path / 'spectra_killer_ai' / 'strategies' / 'advanced' / 'nq_stats_analyzer.py'
spec = importlib.util.spec_from_file_location("nq_stats_analyzer", analyzer_path)
nq_stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nq_stats)

NQStatsAnalyzer = nq_stats.NQStatsAnalyzer
SDEVLevels = nq_stats.SDEVLevels


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'sdev_values': {
            'daily': 1.376,
            'hourly': 0.34
        },
        'high_probability_threshold': 0.75,
        'medium_probability_threshold': 0.51
    }


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data"""
    end_time = datetime.now(pytz.timezone('US/Eastern'))
    start_time = end_time - timedelta(days=3)
    date_range = pd.date_range(start=start_time, end=end_time, freq='5min')

    np.random.seed(42)
    base_price = 15000
    returns = np.random.normal(0, 0.001, len(date_range))
    prices = base_price * (1 + returns).cumprod()

    data = pd.DataFrame(index=date_range)
    data['open'] = prices
    data['high'] = prices * 1.001
    data['low'] = prices * 0.999
    data['close'] = prices
    data['volume'] = 1000

    return data


class TestSDEVLevels:
    """Test SDEV level calculations"""

    def test_sdev_levels_calculation(self):
        """Test SDEV level calculation"""
        sdev = SDEVLevels(
            timeframe='daily',
            open_price=15000,
            sdev_value=1.376
        )

        # Check levels are calculated
        assert sdev.plus_1_0 > sdev.open_price
        assert sdev.minus_1_0 < sdev.open_price

        # Check relationships
        assert sdev.plus_1_0 > sdev.plus_0_5
        assert sdev.minus_1_0 < sdev.minus_0_5

    def test_sdev_position_calculation(self):
        """Test current position calculation"""
        sdev = SDEVLevels(
            timeframe='daily',
            open_price=15000,
            sdev_value=1.376
        )

        # Test at mean
        distance, zone = sdev.get_current_position(15000)
        assert distance == 0
        assert zone == "mean_zone"

        # Test above mean
        distance, zone = sdev.get_current_position(15200)
        assert distance > 0
        assert zone in ["normal_zone", "extended_zone"]

        # Test below mean
        distance, zone = sdev.get_current_position(14800)
        assert distance < 0

    def test_reversion_probability(self):
        """Test reversion probability calculation"""
        sdev = SDEVLevels(
            timeframe='daily',
            open_price=15000,
            sdev_value=1.376
        )

        # At mean - low reversion
        prob = sdev.get_reversion_probability(15000)
        assert 0.2 <= prob <= 0.4

        # At extremes - high reversion
        prob = sdev.get_reversion_probability(15500)
        assert prob >= 0.7


class TestNQStatsAnalyzer:
    """Test NQ Stats Analyzer"""

    def test_analyzer_initialization(self, sample_config):
        """Test analyzer initializes correctly"""
        analyzer = NQStatsAnalyzer(sample_config)

        assert analyzer.config == sample_config
        assert analyzer.sdev_values == sample_config['sdev_values']

    def test_analyze_returns_results(self, sample_config, sample_data):
        """Test analyze returns valid results"""
        analyzer = NQStatsAnalyzer(sample_config)
        results = analyzer.analyze(sample_data)

        # Check basic structure
        assert 'timestamp' in results
        assert 'sdev_analysis' in results
        assert 'hour_stats' in results
        assert 'signals' in results
        assert 'overall_confidence' in results

    def test_sdev_analysis_available(self, sample_config, sample_data):
        """Test SDEV analysis is available"""
        analyzer = NQStatsAnalyzer(sample_config)
        results = analyzer.analyze(sample_data)

        sdev = results['sdev_analysis']
        assert sdev['available'] == True
        assert 'sdev_levels' in sdev

    def test_hour_stats_available(self, sample_config, sample_data):
        """Test hour stats analysis"""
        analyzer = NQStatsAnalyzer(sample_config)
        results = analyzer.analyze(sample_data)

        hour_stats = results['hour_stats']
        assert hour_stats['available'] == True
        assert 'segment' in hour_stats
        assert hour_stats['segment'] in [1, 2, 3]

    def test_signals_generated(self, sample_config, sample_data):
        """Test signals are generated"""
        analyzer = NQStatsAnalyzer(sample_config)
        results = analyzer.analyze(sample_data)

        signals = results['signals']
        assert isinstance(signals, list)

        # If signals exist, check structure
        if signals:
            signal = signals[0]
            assert hasattr(signal, 'signal_type')
            assert hasattr(signal, 'probability')
            assert hasattr(signal, 'confidence')

    def test_overall_confidence_calculated(self, sample_config, sample_data):
        """Test overall confidence calculation"""
        analyzer = NQStatsAnalyzer(sample_config)
        results = analyzer.analyze(sample_data)

        confidence = results['overall_confidence']
        assert 'score' in confidence
        assert 0 <= confidence['score'] <= 1
        assert 'level' in confidence
        assert 'recommendation' in confidence

    def test_empty_data_handling(self, sample_config):
        """Test handling of empty data"""
        analyzer = NQStatsAnalyzer(sample_config)
        empty_data = pd.DataFrame()

        results = analyzer.analyze(empty_data)

        # Should return error or empty result
        assert 'error' in results or results.get('available') == False

    def test_different_timeframes(self, sample_config, sample_data):
        """Test analysis at different times of day"""
        analyzer = NQStatsAnalyzer(sample_config)

        # Morning
        morning = datetime.now(pytz.timezone('US/Eastern')).replace(hour=9, minute=30)
        results_morning = analyzer.analyze(sample_data, morning)

        # Afternoon
        afternoon = datetime.now(pytz.timezone('US/Eastern')).replace(hour=14, minute=30)
        results_afternoon = analyzer.analyze(sample_data, afternoon)

        # Both should return results
        assert 'timestamp' in results_morning
        assert 'timestamp' in results_afternoon


class TestIntegration:
    """Integration tests"""

    def test_full_analysis_workflow(self, sample_config, sample_data):
        """Test complete analysis workflow"""
        analyzer = NQStatsAnalyzer(sample_config)

        # Run analysis
        results = analyzer.analyze(sample_data)

        # Verify all components
        assert results['sdev_analysis']['available']
        assert results['hour_stats']['available']
        assert isinstance(results['signals'], list)
        assert 'overall_confidence' in results

        # Check confidence makes sense
        confidence = results['overall_confidence']
        assert confidence['recommendation'] in ['LONG', 'SHORT', 'WAIT', 'LONG_SMALL', 'SHORT_SMALL']

    def test_probability_values_valid(self, sample_config, sample_data):
        """Test all probabilities are in valid range"""
        analyzer = NQStatsAnalyzer(sample_config)
        results = analyzer.analyze(sample_data)

        # Check SDEV probabilities
        sdev = results['sdev_analysis']
        if 'sdev_levels' in sdev and 'daily' in sdev['sdev_levels']:
            daily = sdev['sdev_levels']['daily']
            if 'reversion_probability' in daily:
                assert 0 <= daily['reversion_probability'] <= 1

        # Check signal probabilities
        for signal in results['signals']:
            assert 0 <= signal.probability <= 1
            assert 0 <= signal.confidence <= 1

        # Check overall confidence
        confidence = results['overall_confidence']
        assert 0 <= confidence['score'] <= 1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])
