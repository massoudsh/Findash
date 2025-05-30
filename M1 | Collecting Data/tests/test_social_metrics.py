import pytest
from ..Scraping import SocialMetricsAnalyzer

def test_social_metrics_analyzer_init():
    """Test SocialMetricsAnalyzer initialization"""
    analyzer = SocialMetricsAnalyzer()
    assert hasattr(analyzer, 'sia')

def test_safe_division():
    """Test safe division utility method"""
    analyzer = SocialMetricsAnalyzer()
    
    assert analyzer.safe_division(10, 2) == 5.0
    assert analyzer.safe_division(10, 0) == 0
    assert analyzer.safe_division(0, 5) == 0

@pytest.mark.parametrize("n,d,expected", [
    (10, 2, 5.0),
    (10, 0, 0),
    (0, 5, 0),
    (-10, 2, -5.0),
])
def test_safe_division_parametrized(n, d, expected):
    """Test safe division with various inputs"""
    analyzer = SocialMetricsAnalyzer()
    assert analyzer.safe_division(n, d) == expected 