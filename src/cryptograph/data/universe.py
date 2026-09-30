"""CryptoGraph Universe Definition, Sector Taxonomy, and Selection Auditing.

Defines the 30-coin primary research universe, event coins (LUNC, FTT),
sector taxonomies, Binance spot symbol mappings, and exclusion audit records.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

TARGET_UNIVERSE_30: List[str] = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "DOGE", "ADA", "AVAX", "LINK", "SUI",
    "SHIB", "NEAR", "DOT", "BCH", "LTC",
    "UNI", "PEPE", "APT", "ICP", "TRX",
    "XLM", "ETC", "ATOM", "FIL", "ARB",
    "OP", "INJ", "RENDER", "FET", "AAVE"
]

EVENT_COINS: List[str] = ["LUNC", "FTT"]

SECTOR_TAXONOMY: Dict[str, str] = {
    "BTC": "Layer 1",
    "ETH": "Layer 1",
    "BNB": "Layer 1",
    "SOL": "Layer 1",
    "XRP": "Payments",
    "DOGE": "Meme",
    "ADA": "Layer 1",
    "AVAX": "Layer 1",
    "LINK": "Infrastructure",
    "SUI": "Layer 1",
    "SHIB": "Meme",
    "NEAR": "Layer 1",
    "DOT": "Interoperability",
    "BCH": "Payments",
    "LTC": "Payments",
    "UNI": "DeFi",
    "PEPE": "Meme",
    "APT": "Layer 1",
    "ICP": "Layer 1",
    "TRX": "Layer 1",
    "XLM": "Payments",
    "ETC": "Layer 1",
    "ATOM": "Interoperability",
    "FIL": "DePIN",
    "ARB": "Layer 2",
    "OP": "Layer 2",
    "INJ": "DeFi",
    "RENDER": "DePIN",
    "FET": "AI",
    "AAVE": "DeFi",
    "LUNC": "Event Coin",
    "FTT": "Event Coin"
}

BINANCE_SPOT_SYMBOLS: Dict[str, str] = {
    coin: f"{coin}USDT" for coin in TARGET_UNIVERSE_30
}
BINANCE_SPOT_SYMBOLS["LUNC"] = "LUNCUSDT"
BINANCE_SPOT_SYMBOLS["FTT"] = "FTTUSDT"

CANDIDATE_AUDIT_CATALOGUE: List[Dict[str, Any]] = [
    # Top 30 Selected Coins
    {"coin": "BTC", "status": "Included", "category": "Layer 1", "spot_market": "BTCUSDT", "history_years": ">7y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "ETH", "status": "Included", "category": "Layer 1", "spot_market": "ETHUSDT", "history_years": ">7y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "BNB", "status": "Included", "category": "Layer 1", "spot_market": "BNBUSDT", "history_years": ">6y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "SOL", "status": "Included", "category": "Layer 1", "spot_market": "SOLUSDT", "history_years": ">4y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "XRP", "status": "Included", "category": "Payments", "spot_market": "XRPUSDT", "history_years": ">6y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "DOGE", "status": "Included", "category": "Meme", "spot_market": "DOGEUSDT", "history_years": ">5y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "ADA", "status": "Included", "category": "Layer 1", "spot_market": "ADAUSDT", "history_years": ">6y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "AVAX", "status": "Included", "category": "Layer 1", "spot_market": "AVAXUSDT", "history_years": ">4y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "LINK", "status": "Included", "category": "Infrastructure", "spot_market": "LINKUSDT", "history_years": ">5y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "SUI", "status": "Included", "category": "Layer 1", "spot_market": "SUIUSDT", "history_years": ">2y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "SHIB", "status": "Included", "category": "Meme", "spot_market": "SHIBUSDT", "history_years": ">3y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "NEAR", "status": "Included", "category": "Layer 1", "spot_market": "NEARUSDT", "history_years": ">4y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "DOT", "status": "Included", "category": "Interoperability", "spot_market": "DOTUSDT", "history_years": ">4y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "BCH", "status": "Included", "category": "Payments", "spot_market": "BCHUSDT", "history_years": ">4y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "LTC", "status": "Included", "category": "Payments", "spot_market": "LTCUSDT", "history_years": ">6y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "UNI", "status": "Included", "category": "DeFi", "spot_market": "UNIUSDT", "history_years": ">4y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "PEPE", "status": "Included", "category": "Meme", "spot_market": "PEPEUSDT", "history_years": ">2y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "APT", "status": "Included", "category": "Layer 1", "spot_market": "APTUSDT", "history_years": ">2.5y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "ICP", "status": "Included", "category": "Layer 1", "spot_market": "ICPUSDT", "history_years": ">3y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "TRX", "status": "Included", "category": "Layer 1", "spot_market": "TRXUSDT", "history_years": ">6y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "XLM", "status": "Included", "category": "Payments", "spot_market": "XLMUSDT", "history_years": ">6y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "ETC", "status": "Included", "category": "Layer 1", "spot_market": "ETCUSDT", "history_years": ">6y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "ATOM", "status": "Included", "category": "Interoperability", "spot_market": "ATOMUSDT", "history_years": ">5y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "FIL", "status": "Included", "category": "DePIN", "spot_market": "FILUSDT", "history_years": ">4y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "ARB", "status": "Included", "category": "Layer 2", "spot_market": "ARBUSDT", "history_years": ">2.5y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "OP", "status": "Included", "category": "Layer 2", "spot_market": "OPUSDT", "history_years": ">3y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "INJ", "status": "Included", "category": "DeFi", "spot_market": "INJUSDT", "history_years": ">4y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "RENDER", "status": "Included", "category": "DePIN", "spot_market": "RENDERUSDT", "history_years": ">3y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria; formerly RNDR)"},
    {"coin": "FET", "status": "Included", "category": "AI", "spot_market": "FETUSDT", "history_years": ">5y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},
    {"coin": "AAVE", "status": "Included", "category": "DeFi", "spot_market": "AAVEUSDT", "history_years": ">4y", "test_coverage": "100%", "reason": "Target 30 Universe (Passed all criteria)"},

    # Exclusions - Delisted / Insufficient Coverage
    {"coin": "XMR", "status": "Excluded", "category": "Privacy", "spot_market": "XMRUSDT", "history_years": ">7y", "test_coverage": "6%", "reason": "Binance spot delisting on 2024-02-20 (<90% test period coverage)"},

    # Exclusions - Stablecoins
    {"coin": "USDT", "status": "Excluded", "category": "Stablecoin", "spot_market": "N/A", "history_years": "N/A", "test_coverage": "N/A", "reason": "Excluded fiat-pegged stablecoin"},
    {"coin": "USDC", "status": "Excluded", "category": "Stablecoin", "spot_market": "USDCUSDT", "history_years": ">5y", "test_coverage": "100%", "reason": "Excluded fiat-pegged stablecoin"},
    {"coin": "DAI", "status": "Excluded", "category": "Stablecoin", "spot_market": "DAIUSDT", "history_years": ">4y", "test_coverage": "100%", "reason": "Excluded algorithmic/fiat stablecoin"},
    {"coin": "FDUSD", "status": "Excluded", "category": "Stablecoin", "spot_market": "FDUSDUSDT", "history_years": "<2y", "test_coverage": "100%", "reason": "Excluded fiat-pegged stablecoin"},
    {"coin": "USDe", "status": "Excluded", "category": "Stablecoin", "spot_market": "N/A", "history_years": "<1y", "test_coverage": "N/A", "reason": "Excluded synthetic dollar stablecoin"},

    # Exclusions - Wrapped / Staked Assets
    {"coin": "WBTC", "status": "Excluded", "category": "Wrapped", "spot_market": "WBTCBTC", "history_years": ">4y", "test_coverage": "100%", "reason": "Excluded wrapped token (duplicates BTC)"},
    {"coin": "STETH", "status": "Excluded", "category": "Liquid Staking", "spot_market": "N/A", "history_years": ">3y", "test_coverage": "N/A", "reason": "Excluded liquid staking receipt (duplicates ETH)"},
    {"coin": "WSTETH", "status": "Excluded", "category": "Liquid Staking", "spot_market": "WSTETHUSDT", "history_years": "<2y", "test_coverage": "100%", "reason": "Excluded liquid staking wrapper"},

    # Exclusions - Exchange Specific / Not on Binance Spot
    {"coin": "LEO", "status": "Excluded", "category": "Exchange", "spot_market": "None", "history_years": "N/A", "test_coverage": "0%", "reason": "Bitfinex proprietary token, not listed on Binance spot"},
    {"coin": "WBT", "status": "Excluded", "category": "Exchange", "spot_market": "None", "history_years": "N/A", "test_coverage": "0%", "reason": "WhiteBIT exchange token, not listed on Binance spot"},
    {"coin": "MNT", "status": "Excluded", "category": "Layer 2", "spot_market": "None", "history_years": "N/A", "test_coverage": "0%", "reason": "Bybit Mantle ecosystem, not listed on Binance spot"},
    {"coin": "OKB", "status": "Excluded", "category": "Exchange", "spot_market": "None", "history_years": "N/A", "test_coverage": "0%", "reason": "OKX exchange token, not listed on Binance spot"},

    # Event Coins
    {"coin": "LUNC", "status": "Event Coin", "category": "Layer 1", "spot_market": "LUNCUSDT", "history_years": ">4y", "test_coverage": "100%", "reason": "Preserved for Terra collapse case studies (May 2022)"},
    {"coin": "FTT", "status": "Event Coin", "category": "Exchange", "spot_market": "FTTUSDT", "history_years": ">3y", "test_coverage": "100%", "reason": "Preserved for FTX collapse case studies (Nov 2022)"}
]

def get_universe(resolution: str = "1h") -> List[str]:
    """Returns primary research coin universe."""
    return list(TARGET_UNIVERSE_30)

def get_binance_symbol(coin: str) -> str:
    """Returns Binance USDT spot pair for coin."""
    return BINANCE_SPOT_SYMBOLS.get(coin, f"{coin}USDT")

def get_sector(coin: str) -> str:
    """Returns economic sector taxonomy for coin."""
    return SECTOR_TAXONOMY.get(coin, "Other")

def generate_universe_audit_report(reports_dir: str = "reports") -> Path:
    """Generates and saves universe_audit_v2.csv."""
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "universe_audit_v2.csv"

    df = pd.DataFrame(CANDIDATE_AUDIT_CATALOGUE)
    df.to_csv(out_path, index=False)
    return out_path
