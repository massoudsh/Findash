import pytest
from unittest.mock import Mock
from bs4 import BeautifulSoup

@pytest.fixture
def mock_alpaca_api():
    """Fixture for mocked Alpaca API"""
    mock_api = Mock()
    mock_api.submit_order.return_value = Mock(id='order123')
    mock_api.list_positions.return_value = [
        Mock(symbol='AAPL', qty=10, current_price=150.0)
    ]
    return mock_api

@pytest.fixture
def sample_html_content():
    """Fixture for sample HTML content"""
    return """
    <div class="item">
        <a href="#">Test News Title</a>
        <div class="summary">Test News Summary</div>
    </div>
    """

@pytest.fixture
def sample_soup(sample_html_content):
    """Fixture for BeautifulSoup object"""
    return BeautifulSoup(sample_html_content, 'html.parser') 