"""Tests for the AI Asset Allocation Copilot endpoint (concentration/diversification analysis)."""


def test_allocation_analysis_empty_holdings(client):
    resp = client.post("/api/copilot/allocation-analysis", json={"holdings": []})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_value"] == 0
    assert data["category_breakdown"] == []
    assert "disclaimer" in data and data["disclaimer"]


def test_allocation_analysis_concentrated_portfolio(client):
    resp = client.post(
        "/api/copilot/allocation-analysis",
        json={
            "holdings": [
                {"code": "XAU18", "name": "طلای 18 عیار", "type": "gold", "value": 90_000_000},
                {"code": "USD", "name": "دلار", "type": "currency", "value": 10_000_000},
            ]
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_value"] == 100_000_000
    gold = next(c for c in data["category_breakdown"] if c["type"] == "gold")
    assert gold["pct"] == 90.0
    assert data["concentration_level"] == "بالا"
    assert data["top_holding_pct"] == 90.0
    assert any("insights" != "" for _ in data["insights"])  # non-empty insights list
    assert len(data["insights"]) > 0


def test_allocation_analysis_diversified_portfolio(client):
    resp = client.post(
        "/api/copilot/allocation-analysis",
        json={
            "holdings": [
                {"code": "XAU18", "name": "طلا", "type": "gold", "value": 25_000_000},
                {"code": "USD", "name": "دلار", "type": "currency", "value": 25_000_000},
                {"code": "BTC", "name": "بیت‌کوین", "type": "crypto", "value": 25_000_000},
                {"code": "RE_TEHRAN", "name": "مسکن", "type": "real_estate", "value": 25_000_000},
            ]
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    # 4 evenly-split categories -> HHI = 2500 (moderate concentration boundary)
    assert data["concentration_level"] == "متوسط"
    assert data["diversification_score"] > 50
