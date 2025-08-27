# server.py
from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, List

import ta
from fastmcp import FastMCP
from dotenv import load_dotenv
from bitmart_perp import PerpBitmart, TransactionType
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

load_dotenv()

mcp = FastMCP(
    name="Bitmart Assistant",
    instructions="""
        This server provides data analysis tools about Bitmart Crypto Exchange.
        You can use the tools to get the data of the crypto market.
    """,
)

API_KEY = os.getenv("BITMART_API_KEY")
API_SECRET = os.getenv("BITMART_API_SECRET")
UID = os.getenv("BITMART_MEMO")

# Create a single instance of PerpBitmart
client = PerpBitmart(public_api=API_KEY, secret_api=API_SECRET, uid=UID)

# Flag to track if markets are loaded
markets_loaded = False


async def ensure_markets_loaded():
    global markets_loaded
    if not markets_loaded:
        await client.load_markets()
        markets_loaded = True


@mcp.tool
async def get_crypto_market_data(crypto: str) -> dict:
    """
    Get market data for a cryptocurrency from Bitmart exchange.

    Args:
        crypto: The cryptocurrency symbol (e.g., 'BTC', 'ETH')

    Returns:
        Market information for the crypto/USDT pair or error message
    """
    try:
        await ensure_markets_loaded()

        # Check if the pair exists
        pair_info = client.get_pair_info(f"{crypto.upper()}/USDT")
        if pair_info is None:
            return {
                "error": f"The cryptocurrency {crypto.upper()} is not available on Bitmart"
            }

        # Get ticker data
        ticker = await client._session.fetch_ticker(f"{crypto.upper()}/USDT:USDT")

        return {
            "symbol": f"{crypto.upper()}/USDT",
            "market_info": pair_info,
            "ticker_data": ticker,
        }

    except Exception as e:
        return {"error": f"Error fetching data: {str(e)}"}


@mcp.tool
async def get_open_positions() -> dict:
    """
    Get all open positions from Bitmart exchange.

    Returns:
        List of open positions or error message
    """
    try:
        await ensure_markets_loaded()
        # Get positions
        positions = await client.get_open_positions()

        if not positions:
            return {"message": "No open positions found", "positions": []}

        return {
            "message": f"Found {len(positions)} open position(s)",
            "positions": positions,
        }

    except Exception as e:
        return {"error": f"Error fetching positions: {str(e)}"}


@mcp.tool
async def get_balance() -> dict:
    """
    Get USDT balance from Bitmart exchange.

    Returns:
        USDT balance information or error message
    """
    try:
        await ensure_markets_loaded()

        balance = await client.get_balance()

        return {"total": balance.total, "free": balance.free, "used": balance.used}

    except Exception as e:
        return {"error": f"Error fetching balance: {str(e)}"}


@mcp.tool
async def transactions_analysis(
    pair: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze transactions and return a summary dict with:
    - funding_fees: sum of all funding fees
    - commission_fees: sum of all commission fees
    - realized_pnl: sum of all realized PnL
    - transactions: list of all realized PnL transactions

    Args:
        pair: Optional pair to filter (e.g., 'BTC/USDT')
        start_date: Optional start date as string 'DD-MM-YYYY' (default: 7 days ago)
        end_date: Optional end date as string 'DD-MM-YYYY' (default: today)
    """
    await ensure_markets_loaded()

    # Convert dates to timestamps
    if end_date:
        end_dt = datetime.strptime(end_date, "%d-%m-%Y")
        # Set to end of day
        end_dt = end_dt.replace(hour=23, minute=59, second=59)
        end_timestamp = int(end_dt.timestamp() * 1000)
    else:
        end_timestamp = None

    if start_date:
        start_dt = datetime.strptime(start_date, "%d-%m-%Y")
        # Set to start of day
        start_dt = start_dt.replace(hour=0, minute=0, second=0)
        start_timestamp = int(start_dt.timestamp() * 1000)
    else:
        # Default to 7 days ago
        start_dt = datetime.now() - timedelta(days=7)
        start_dt = start_dt.replace(hour=0, minute=0, second=0)
        start_timestamp = int(start_dt.timestamp() * 1000)

    # Get all transactions
    all_transactions = await client.get_transactions(
        pair=pair, start_time=start_timestamp, end_time=end_timestamp
    )

    # Initialize sums
    funding_fees = 0.0
    commission_fees = 0.0
    realized_pnl = 0.0
    pnl_transactions = []

    # Initialize pair analysis
    pair_analysis = {}

    # Process each transaction
    for tx in all_transactions:
        # Get or create pair entry
        if tx.symbol:
            if tx.symbol not in pair_analysis:
                pair_analysis[tx.symbol] = {
                    "funding_fees": 0.0,
                    "commission_fees": 0.0,
                    "realized_pnl": 0.0,
                    "total_transactions": 0,
                }

            pair_analysis[tx.symbol]["total_transactions"] += 1

        # Update totals and pair-specific stats
        if tx.type == TransactionType.FUNDING_FEE:
            funding_fees += tx.amount
            if tx.symbol:
                pair_analysis[tx.symbol]["funding_fees"] += tx.amount
        elif tx.type == TransactionType.COMMISSION_FEE:
            commission_fees += tx.amount
            if tx.symbol:
                pair_analysis[tx.symbol]["commission_fees"] += tx.amount
        elif tx.type == TransactionType.REALIZED_PNL:
            realized_pnl += tx.amount
            pnl_transactions.append(tx)
            if tx.symbol:
                pair_analysis[tx.symbol]["realized_pnl"] += tx.amount

    return {
        "funding_fees": funding_fees,
        "commission_fees": commission_fees,
        "realized_pnl": realized_pnl,
        "total_transactions": len(all_transactions),
        "transactions": pnl_transactions,
        "pair_analysis": pair_analysis,
    }


@mcp.tool
async def analyze_crypto(
    pair: str, timeframe: str = "1h", lookback_periods: int = 200
) -> Dict[str, Any]:
    """
    Comprehensive technical analysis of a cryptocurrency pair.

    Provides extensive market analysis including:
    - Multiple technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
    - Support and resistance levels detection
    - Trend analysis and momentum
    - Volume analysis
    - Chart pattern recognition
    - Market structure analysis

    Args:
        pair: The cryptocurrency pair (e.g., 'BTC/USDT')
        timeframe: Time interval for candles ('1m', '5m', '15m', '1h', '4h', '1d')
        lookback_periods: Number of historical candles to analyze (default: 200)

    Returns:
        Comprehensive analysis dictionary with indicators, levels, and market assessment
    """
    try:
        await ensure_markets_loaded()

        # Fetch OHLCV data
        ohlcv = await client.get_last_ohlcv(pair, timeframe, limit=lookback_periods)

        if len(ohlcv) < 50:
            return {"error": "Insufficient data for analysis"}

        # Current price and basic info
        current_price = float(ohlcv["close"].iloc[-1])
        price_change_24h = (
            ((ohlcv["close"].iloc[-1] / ohlcv["close"].iloc[-24]) - 1) * 100
            if len(ohlcv) >= 24
            else 0
        )

        # Calculate RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        rsi = calculate_rsi(ohlcv["close"])
        current_rsi = float(rsi.iloc[-1])

        # Calculate MACD
        exp1 = ohlcv["close"].ewm(span=12, adjust=False).mean()
        exp2 = ohlcv["close"].ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - signal_line

        # Bollinger Bands
        sma_20 = ohlcv["close"].rolling(window=20).mean()
        std_20 = ohlcv["close"].rolling(window=20).std()
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        bb_width = ((bb_upper - bb_lower) / sma_20 * 100).iloc[-1]

        # ATR (Average True Range)
        high_low = ohlcv["high"] - ohlcv["low"]
        high_close = np.abs(ohlcv["high"] - ohlcv["close"].shift())
        low_close = np.abs(ohlcv["low"] - ohlcv["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean()
        current_atr = float(atr.iloc[-1])
        atr_percentage = (current_atr / current_price) * 100

        # Moving Averages
        ma_periods = [7, 20, 50, 100, 200]
        moving_averages = {}
        ma_signals = {}

        for period in ma_periods:
            if len(ohlcv) >= period:
                ma = ohlcv["close"].rolling(window=period).mean().iloc[-1]
                moving_averages[f"MA{period}"] = round(float(ma), 2)
                ma_signals[f"MA{period}"] = "above" if current_price > ma else "below"

        # EMA (Exponential Moving Averages)
        ema_20 = ohlcv["close"].ewm(span=20, adjust=False).mean().iloc[-1]
        ema_50 = (
            ohlcv["close"].ewm(span=50, adjust=False).mean().iloc[-1]
            if len(ohlcv) >= 50
            else None
        )

        # Support and Resistance Detection with clustering
        def find_support_resistance(
            prices, window=20, num_levels=5, cluster_threshold=0.005
        ):
            """
            Find support and resistance levels using clustering to group nearby levels
            cluster_threshold: 0.5% price difference to consider levels as the same zone
            """
            # Find local maxima and minima
            highs = prices["high"].rolling(window=window, center=True).max()
            lows = prices["low"].rolling(window=window, center=True).min()

            # Identify turning points
            potential_resistance = prices[prices["high"] == highs]["high"].values
            potential_support = prices[prices["low"] == lows]["low"].values

            # Cluster nearby levels for resistance
            def cluster_levels(levels, threshold_pct):
                if len(levels) == 0:
                    return []

                clustered = []
                levels = sorted(levels)
                current_cluster = [levels[0]]

                for level in levels[1:]:
                    # Check if level is within threshold of cluster average
                    cluster_avg = np.mean(current_cluster)
                    if abs(level - cluster_avg) / cluster_avg <= threshold_pct:
                        current_cluster.append(level)
                    else:
                        # Save current cluster if it has at least 2 points
                        if len(current_cluster) >= 2:
                            clustered.append(
                                {
                                    "level": np.mean(current_cluster),
                                    "strength": len(current_cluster),
                                    "touches": current_cluster,
                                }
                            )
                        elif len(current_cluster) == 1:
                            # Keep single strong levels too, but mark them as weaker
                            clustered.append(
                                {
                                    "level": current_cluster[0],
                                    "strength": 1,
                                    "touches": current_cluster,
                                }
                            )
                        current_cluster = [level]

                # Don't forget the last cluster
                if len(current_cluster) >= 2:
                    clustered.append(
                        {
                            "level": np.mean(current_cluster),
                            "strength": len(current_cluster),
                            "touches": current_cluster,
                        }
                    )
                elif len(current_cluster) == 1:
                    clustered.append(
                        {
                            "level": current_cluster[0],
                            "strength": 1,
                            "touches": current_cluster,
                        }
                    )

                return clustered

            # Cluster the levels
            resistance_clusters = cluster_levels(
                potential_resistance[potential_resistance > current_price],
                cluster_threshold,
            )
            support_clusters = cluster_levels(
                potential_support[potential_support < current_price], cluster_threshold
            )

            # Sort by strength (number of touches) and proximity to current price
            resistance_clusters.sort(key=lambda x: (-x["strength"], x["level"]))
            support_clusters.sort(key=lambda x: (-x["strength"], -x["level"]))

            # Extract the strongest levels
            resistance_levels = [c["level"] for c in resistance_clusters[:num_levels]]
            support_levels = [c["level"] for c in support_clusters[:num_levels]]

            # Sort by proximity to current price
            resistance_levels = sorted(resistance_levels)[:num_levels]
            support_levels = sorted(support_levels, reverse=True)[:num_levels]

            return support_levels, resistance_levels

        support_levels, resistance_levels = find_support_resistance(ohlcv)

        # Also get detailed support/resistance info for strength analysis
        def get_sr_details(prices, window=20, cluster_threshold=0.005):
            highs = prices["high"].rolling(window=window, center=True).max()
            lows = prices["low"].rolling(window=window, center=True).min()
            potential_resistance = prices[prices["high"] == highs]["high"].values
            potential_support = prices[prices["low"] == lows]["low"].values

            def cluster_levels_detailed(levels, threshold_pct):
                if len(levels) == 0:
                    return []
                clustered = []
                levels = sorted(levels)
                current_cluster = [levels[0]]

                for level in levels[1:]:
                    cluster_avg = np.mean(current_cluster)
                    if abs(level - cluster_avg) / cluster_avg <= threshold_pct:
                        current_cluster.append(level)
                    else:
                        if len(current_cluster) >= 2:
                            clustered.append(
                                {
                                    "level": round(np.mean(current_cluster), 2),
                                    "strength": len(current_cluster),
                                    "touches": len(current_cluster),
                                }
                            )
                        elif len(current_cluster) == 1:
                            clustered.append(
                                {
                                    "level": round(current_cluster[0], 2),
                                    "strength": 1,
                                    "touches": 1,
                                }
                            )
                        current_cluster = [level]

                if len(current_cluster) >= 2:
                    clustered.append(
                        {
                            "level": round(np.mean(current_cluster), 2),
                            "strength": len(current_cluster),
                            "touches": len(current_cluster),
                        }
                    )
                elif len(current_cluster) == 1:
                    clustered.append(
                        {
                            "level": round(current_cluster[0], 2),
                            "strength": 1,
                            "touches": 1,
                        }
                    )

                return clustered

            resistance_details = cluster_levels_detailed(
                potential_resistance[potential_resistance > current_price],
                cluster_threshold,
            )
            support_details = cluster_levels_detailed(
                potential_support[potential_support < current_price], cluster_threshold
            )

            resistance_details.sort(key=lambda x: x["level"])
            support_details.sort(key=lambda x: -x["level"])

            return support_details[:5], resistance_details[:5]

        support_details, resistance_details = get_sr_details(ohlcv)

        # Volume Analysis
        avg_volume = ohlcv["volume"].rolling(window=20).mean().iloc[-1]
        current_volume = ohlcv["volume"].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        # OBV (On-Balance Volume)
        obv = (np.sign(ohlcv["close"].diff()) * ohlcv["volume"]).fillna(0).cumsum()
        obv_trend = "bullish" if obv.iloc[-1] > obv.iloc[-20] else "bearish"

        # Stochastic Oscillator
        low_14 = ohlcv["low"].rolling(window=14).min()
        high_14 = ohlcv["high"].rolling(window=14).max()
        stoch_k = 100 * ((ohlcv["close"] - low_14) / (high_14 - low_14))
        stoch_d = stoch_k.rolling(window=3).mean()

        # Trend Detection
        def detect_trend(prices, period=20):
            sma = prices.rolling(window=period).mean()
            if len(prices) < period:
                return "undefined"

            recent_sma = sma.iloc[-10:]
            if recent_sma.is_monotonic_increasing:
                return "strong_uptrend"
            elif recent_sma.is_monotonic_decreasing:
                return "strong_downtrend"
            elif prices.iloc[-1] > sma.iloc[-1]:
                return "uptrend"
            else:
                return "downtrend"

        short_trend = detect_trend(ohlcv["close"], 20)
        medium_trend = (
            detect_trend(ohlcv["close"], 50) if len(ohlcv) >= 50 else "undefined"
        )
        long_trend = (
            detect_trend(ohlcv["close"], 200) if len(ohlcv) >= 200 else "undefined"
        )

        # Volatility metrics
        returns = ohlcv["close"].pct_change()
        volatility_daily = returns.std() * np.sqrt(24) * 100  # Daily volatility
        volatility_weekly = returns.std() * np.sqrt(24 * 7) * 100  # Weekly volatility

        # Price action patterns
        last_candle = {
            "open": float(ohlcv["open"].iloc[-1]),
            "high": float(ohlcv["high"].iloc[-1]),
            "low": float(ohlcv["low"].iloc[-1]),
            "close": float(ohlcv["close"].iloc[-1]),
            "volume": float(ohlcv["volume"].iloc[-1]),
        }

        # Candle pattern detection
        body_size = abs(last_candle["close"] - last_candle["open"])
        upper_wick = last_candle["high"] - max(
            last_candle["close"], last_candle["open"]
        )
        lower_wick = min(last_candle["close"], last_candle["open"]) - last_candle["low"]

        candle_type = "neutral"
        if body_size < (last_candle["high"] - last_candle["low"]) * 0.1:
            candle_type = "doji"
        elif lower_wick > body_size * 2 and upper_wick < body_size * 0.5:
            candle_type = (
                "hammer"
                if last_candle["close"] > last_candle["open"]
                else "hanging_man"
            )
        elif upper_wick > body_size * 2 and lower_wick < body_size * 0.5:
            candle_type = (
                "shooting_star"
                if last_candle["close"] < last_candle["open"]
                else "inverted_hammer"
            )
        elif last_candle["close"] > last_candle["open"]:
            candle_type = "bullish"
        else:
            candle_type = "bearish"

        # Market structure (keep 20 for recent structure)
        recent_high = float(ohlcv["high"].iloc[-20:].max())
        recent_low = float(ohlcv["low"].iloc[-20:].min())
        price_position = (
            ((current_price - recent_low) / (recent_high - recent_low)) * 100
            if recent_high != recent_low
            else 50
        )

        # Momentum indicators
        momentum = ((ohlcv["close"].iloc[-1] / ohlcv["close"].iloc[-10]) - 1) * 100
        roc = (
            (ohlcv["close"].iloc[-1] / ohlcv["close"].iloc[-12]) - 1
        ) * 100  # Rate of Change

        # Fibonacci retracement levels using ALL available data (up to 200 periods)
        fib_levels = {}
        # Use all available data for Fibonacci (max 200 periods as per lookback_periods)
        fib_high = float(ohlcv["high"].max())  # Highest high in entire dataset
        fib_low = float(ohlcv["low"].min())  # Lowest low in entire dataset

        if fib_high != fib_low:
            fib_ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
            for ratio in fib_ratios:
                level = fib_low + (fib_high - fib_low) * (1 - ratio)
                fib_levels[f"fib_{int(ratio*100)}"] = round(level, 2)

            # Add Fibonacci extension levels for potential targets
            extension_ratios = [1.272, 1.414, 1.618, 2.0, 2.618]
            for ratio in extension_ratios:
                level = fib_low + (fib_high - fib_low) * (1 - ratio)
                fib_levels[f"fib_ext_{int(ratio*100)}"] = round(level, 2)

        # Signal generation
        signals = []
        signal_strength = 0

        # RSI signals
        if current_rsi < 30:
            signals.append("RSI oversold")
            signal_strength += 1
        elif current_rsi > 70:
            signals.append("RSI overbought")
            signal_strength -= 1

        # MACD signals
        if (
            macd_line.iloc[-1] > signal_line.iloc[-1]
            and macd_line.iloc[-2] <= signal_line.iloc[-2]
        ):
            signals.append("MACD bullish crossover")
            signal_strength += 2
        elif (
            macd_line.iloc[-1] < signal_line.iloc[-1]
            and macd_line.iloc[-2] >= signal_line.iloc[-2]
        ):
            signals.append("MACD bearish crossover")
            signal_strength -= 2

        # Bollinger Band signals
        if current_price < bb_lower.iloc[-1]:
            signals.append("Price below lower Bollinger Band")
            signal_strength += 1
        elif current_price > bb_upper.iloc[-1]:
            signals.append("Price above upper Bollinger Band")
            signal_strength -= 1

        # Moving average signals
        if "MA50" in moving_averages and "MA200" in moving_averages:
            if moving_averages["MA50"] > moving_averages["MA200"]:
                signals.append("Golden cross (MA50 > MA200)")
                signal_strength += 2
            elif moving_averages["MA50"] < moving_averages["MA200"]:
                signals.append("Death cross (MA50 < MA200)")
                signal_strength -= 2

        # Volume signals
        if volume_ratio > 2:
            signals.append("Unusual high volume")
            signal_strength += 1 if last_candle["close"] > last_candle["open"] else -1

        # Overall market assessment
        if signal_strength >= 3:
            market_bias = "strong_bullish"
        elif signal_strength >= 1:
            market_bias = "bullish"
        elif signal_strength <= -3:
            market_bias = "strong_bearish"
        elif signal_strength <= -1:
            market_bias = "bearish"
        else:
            market_bias = "neutral"

        return {
            "pair": pair,
            "timeframe": timeframe,
            "timestamp": pd.Timestamp.now().isoformat(),
            "current_price": round(current_price, 4),
            "price_change_24h": round(price_change_24h, 2),
            "technical_indicators": {
                "rsi": {
                    "value": round(current_rsi, 2),
                    "status": (
                        "oversold"
                        if current_rsi < 30
                        else "overbought" if current_rsi > 70 else "neutral"
                    ),
                },
                "macd": {
                    "macd_line": round(float(macd_line.iloc[-1]), 4),
                    "signal_line": round(float(signal_line.iloc[-1]), 4),
                    "histogram": round(float(macd_histogram.iloc[-1]), 4),
                    "trend": (
                        "bullish"
                        if macd_line.iloc[-1] > signal_line.iloc[-1]
                        else "bearish"
                    ),
                },
                "bollinger_bands": {
                    "upper": round(float(bb_upper.iloc[-1]), 2),
                    "middle": round(float(sma_20.iloc[-1]), 2),
                    "lower": round(float(bb_lower.iloc[-1]), 2),
                    "width_pct": round(float(bb_width), 2),
                    "position": (
                        "above_upper"
                        if current_price > bb_upper.iloc[-1]
                        else (
                            "below_lower"
                            if current_price < bb_lower.iloc[-1]
                            else "within_bands"
                        )
                    ),
                },
                "atr": {
                    "value": round(current_atr, 4),
                    "percentage": round(atr_percentage, 2),
                    "volatility": (
                        "high"
                        if atr_percentage > 5
                        else "medium" if atr_percentage > 2 else "low"
                    ),
                },
                "stochastic": {
                    "k": round(float(stoch_k.iloc[-1]), 2),
                    "d": round(float(stoch_d.iloc[-1]), 2),
                    "status": (
                        "oversold"
                        if stoch_k.iloc[-1] < 20
                        else "overbought" if stoch_k.iloc[-1] > 80 else "neutral"
                    ),
                },
            },
            "moving_averages": moving_averages,
            "ma_signals": ma_signals,
            "ema": {
                "EMA20": round(float(ema_20), 2),
                "EMA50": round(float(ema_50), 2) if ema_50 else None,
            },
            "support_resistance": {
                "immediate_support": (
                    round(float(support_levels[0]), 2)
                    if len(support_levels) > 0
                    else None
                ),
                "support_levels": [round(float(s), 2) for s in support_levels],
                "support_details": support_details,  # Includes strength (number of touches)
                "immediate_resistance": (
                    round(float(resistance_levels[0]), 2)
                    if len(resistance_levels) > 0
                    else None
                ),
                "resistance_levels": [round(float(r), 2) for r in resistance_levels],
                "resistance_details": resistance_details,  # Includes strength (number of touches)
            },
            "fibonacci_levels": {
                "levels": fib_levels,
                "high": fib_high if "fib_high" in locals() else None,
                "low": fib_low if "fib_low" in locals() else None,
                "data_periods": len(ohlcv),
                "note": f"Based on {len(ohlcv)} periods of {timeframe} data",
            },
            "volume_analysis": {
                "current_volume": round(current_volume, 2),
                "average_volume_20": round(avg_volume, 2),
                "volume_ratio": round(volume_ratio, 2),
                "volume_trend": (
                    "increasing"
                    if volume_ratio > 1.5
                    else "decreasing" if volume_ratio < 0.5 else "normal"
                ),
                "obv_trend": obv_trend,
            },
            "trend_analysis": {
                "short_term": short_trend,
                "medium_term": medium_trend,
                "long_term": long_trend,
                "momentum": round(momentum, 2),
                "rate_of_change": round(roc, 2),
            },
            "volatility": {
                "daily": round(volatility_daily, 2),
                "weekly": round(volatility_weekly, 2),
            },
            "market_structure": {
                "recent_high": round(recent_high, 2),
                "recent_low": round(recent_low, 2),
                "price_position_pct": round(price_position, 2),
                "last_candle_pattern": candle_type,
            },
            "signals": signals,
            "market_bias": market_bias,
            "signal_strength": signal_strength,
            "trading_recommendation": {
                "bias": market_bias,
                "confidence": (
                    "high"
                    if abs(signal_strength) >= 3
                    else "medium" if abs(signal_strength) >= 1 else "low"
                ),
                "key_levels": {
                    "stop_loss_long": (
                        round(support_levels[0] * 0.99, 2)
                        if len(support_levels) > 0
                        else None
                    ),
                    "take_profit_long": (
                        round(resistance_levels[0] * 1.005, 2)
                        if len(resistance_levels) > 0
                        else None
                    ),
                    "stop_loss_short": (
                        round(resistance_levels[0] * 1.01, 2)
                        if len(resistance_levels) > 0
                        else None
                    ),
                    "take_profit_short": (
                        round(support_levels[0] * 0.995, 2)
                        if len(support_levels) > 0
                        else None
                    ),
                },
            },
        }

    except Exception as e:
        return {"error": f"Error analyzing crypto: {str(e)}"}


@mcp.tool
async def detect_up_triangle(
    pair: str,
    timeframe: str = "1h",
) -> Dict[str, Any]:
    """
    Detects upward triangles in the given timeframe for the given pair.
    """
    df = await client.get_last_ohlcv(pair, timeframe, limit=200)

    @dataclass
    class Pivot:
        i: int  # index position (entier)
        t: pd.Timestamp  # timestamp
        price: float
        kind: str  # 'H' ou 'L'

    def fractal_pivots(df: pd.DataFrame, wing: int = 2) -> List[Pivot]:
        """
        Pivots fractals (5-bar par défaut : wing=2).
        Sommet: close[i] = max(close[i-wing : i+wing])
        Creux : close[i]  = min(close[i-wing  : i+wing])
        """
        # H, L = df['high'].to_numpy(), df['low'].to_numpy()
        C = df["close"].to_numpy()
        idx = df.index
        n = len(df)
        pivots: List[Pivot] = []
        for i in range(wing, n - wing):
            window_H = C[i - wing : i + wing + 1]
            window_L = C[i - wing : i + wing + 1]
            if C[i] == window_H.max() and (window_H.argmax() == wing):
                pivots.append(Pivot(i, idx[i], C[i], "H"))
            if C[i] == window_L.min() and (window_L.argmin() == wing):
                pivots.append(Pivot(i, idx[i], C[i], "L"))
        pivots.sort(key=lambda p: p.i)
        return pivots

    def find_bottoms_between_tops(pivots, tops):
        bottoms = []

        # Trier les tops par index pour s'assurer qu'ils sont dans
        tops_sorted = sorted(tops, key=lambda x: x.i)

        # Parcourir chaque paire de tops consécutifs
        for i in range(len(tops_sorted)):
            # Définir les bornes de recherche
            if i == 0:
                # Avant le premier top
                start_idx = 0
                end_idx = tops_sorted[i].i
            else:
                # Entre deux tops
                start_idx = tops_sorted[i - 1].i
                end_idx = tops_sorted[i].i

            # Filtrer les pivots low dans cette plage
            lows_in_range = [
                p for p in pivots if start_idx < p.i < end_idx and p.kind == "L"
            ]

            # Si des lows existent, prendre le minimum
            if lows_in_range:
                min_low = min(lows_in_range, key=lambda x: x.price)
                bottoms.append(min_low)

        # Après le dernier top
        if tops_sorted:
            last_top_idx = tops_sorted[-1].i
            lows_after_last = [
                p for p in pivots if p.i > last_top_idx and p.kind == "L"
            ]

            if lows_after_last:
                min_low = min(lows_after_last, key=lambda x: x.price)
                bottoms.append(min_low)

        return bottoms

    def get_line_price(b1, slope, idx):
        return b1.price + slope * (idx - b1.i)

    def get_low_triangle(bottoms, df, atr):
        low_triangles = []
        best_low_triangle = None
        for b1 in bottoms:
            # print("----")
            # print(b1)
            lines = []
            low_triangle = {"points": [b1], "slope": 0}
            for b2 in [b for b in bottoms if b.i > b1.i]:
                slope = (b2.price - b1.price) / (b2.i - b1.i)

                for idx in range(b1.i, len(df)):
                    if get_line_price(b1, slope, idx) > df["close"].iloc[idx] + atr:
                        break
                lines.append([b2, slope])
                low_triangle["points"].append(b2)
            if len(lines) >= 1:
                mean_slope = sum([l[1] for l in lines]) / len(lines)
                mean_slope_pct = (mean_slope / df["close"].iloc[b1.i]) * 100
                low_triangle["slope"] = mean_slope_pct
                low_triangles.append(low_triangle)
                # print(low_triangle)

        max_points = 0
        for lt in low_triangles:
            if len(lt["points"]) >= max_points:
                max_points = len(lt["points"])
                best_low_triangle = lt
        return best_low_triangle

    def find_upward_triangle(df):
        atr = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=14, fillna=False
        ).average_true_range()
        mean_atr = atr.mean()
        max_close = df["close"].max()
        pivs = fractal_pivots(df, wing=5)
        highs = [p for p in pivs if p.kind == "H"]
        tops = [h for h in highs if h.price > max_close - mean_atr]
        if len(tops) > 2:
            bottoms = find_bottoms_between_tops(pivs, tops)
            best_low_triangle = get_low_triangle(bottoms, df, mean_atr)
            if best_low_triangle:
                triangle = {
                    "top_points": tops,
                    "bottom_points": best_low_triangle["points"],
                    "slope": best_low_triangle["slope"],
                }
                return triangle
            else:
                return None
        else:
            return None

    triangle = find_upward_triangle(df)
    return triangle


@mcp.tool
async def calculate_position_risk(historical_days: int = 45) -> Dict[str, Any]:
    """
    Calculate Value at Risk (VaR) at 99% confidence for multiple time horizons.

    VaR estimates the potential loss in value of positions over different periods
    (1h, 4h, 1d, 1w) using direct calculation on historical price data.

    Args:
        historical_days: Number of days of historical data to use (default: 45)

    Returns:
        Dictionary containing:
        - portfolio_var: Portfolio-level VaR 99% for each time horizon
        - position_risks: Individual position risk metrics with VaR for each horizon
        - risk_summary: Overall risk assessment
    """
    try:
        await ensure_markets_loaded()

        # Define time horizons (in hours)
        time_horizons = {"1h": 1, "4h": 4, "1d": 24, "1w": 168}

        # Get current positions
        positions = await client.get_open_positions()

        if not positions:
            return {
                "message": "No open positions to analyze",
                "portfolio_var": {},
                "position_risks": [],
                "risk_summary": {},
            }

        # Get current balance
        balance = await client.get_balance()

        position_risks = []
        portfolio_returns_by_horizon = {horizon: None for horizon in time_horizons}
        total_position_value = 0

        # Calculate risk for each position
        for position in positions:
            try:
                # Get historical hourly data (need enough for 1 week returns + buffer)
                data_points_needed = max(
                    historical_days * 24, 168 * 2
                )  # At least 2 weeks for weekly returns
                ohlcv = await client.get_last_ohlcv(
                    position.pair, "1h", limit=data_points_needed
                )

                # Calculate VaR for each time horizon
                position_var = {}
                var_1h_pct = None  # Store 1h VaR for sqrt scaling

                for horizon_name, hours in time_horizons.items():
                    if horizon_name == "1w":
                        # Use sqrt scaling for 1 week (more reliable with limited data)
                        if var_1h_pct is not None:
                            var_pct = var_1h_pct * np.sqrt(168)  # sqrt(168 hours)
                            var_dollar = var_pct / 100 * position.usd_size
                            position_var[horizon_name] = {
                                "pct": round(var_pct, 2),
                                "usd": round(var_dollar, 2),
                            }
                    else:
                        # Direct calculation for shorter horizons
                        returns = (
                            ohlcv["close"] / ohlcv["close"].shift(hours) - 1
                        ).dropna()

                        # Calculate VaR 99%
                        var_pct = abs(np.percentile(returns * 100, 1))
                        var_dollar = var_pct / 100 * position.usd_size
                        position_var[horizon_name] = {
                            "pct": round(var_pct, 2),
                            "usd": round(var_dollar, 2),
                        }

                        # Store 1h VaR for later scaling
                        if horizon_name == "1h":
                            var_1h_pct = var_pct

                        # Aggregate returns for portfolio VaR (except 1w which uses scaling)
                        weighted_returns = returns * (position.usd_size / balance.total)
                        if portfolio_returns_by_horizon[horizon_name] is None:
                            portfolio_returns_by_horizon[horizon_name] = (
                                weighted_returns
                            )
                        else:
                            # Align indices before adding
                            portfolio_returns_by_horizon[horizon_name] = (
                                portfolio_returns_by_horizon[horizon_name].add(
                                    weighted_returns, fill_value=0
                                )
                            )

                # Calculate additional risk metrics (using daily returns)
                daily_returns = (ohlcv["close"] / ohlcv["close"].shift(24) - 1).dropna()
                volatility = daily_returns.std()
                max_drawdown = (ohlcv["close"] / ohlcv["close"].cummax() - 1).min()

                # Risk relative to account
                position_risk_pct = (position.usd_size / balance.total) * 100

                # Distance to liquidation
                if position.liquidation_price > 0:
                    if position.side == "long":
                        liq_distance = (
                            (position.current_price - position.liquidation_price)
                            / position.current_price
                        ) * 100
                    else:
                        liq_distance = (
                            (position.liquidation_price - position.current_price)
                            / position.current_price
                        ) * 100
                else:
                    liq_distance = 100  # No liquidation price set

                position_risk = {
                    "pair": position.pair,
                    "side": position.side,
                    "size_usd": round(position.usd_size, 2),
                    "leverage": position.leverage,
                    "unrealized_pnl": round(position.unrealized_pnl, 2),
                    "var_99": position_var,  # VaR 99% for each time horizon
                    "daily_volatility": round(volatility * 100, 2),
                    "max_drawdown": round(max_drawdown * 100, 2),
                    "position_weight": round(position_risk_pct, 2),
                    "liquidation_distance": round(liq_distance, 2),
                    "risk_score": (
                        "HIGH"
                        if liq_distance < 10 or position_risk_pct > 30
                        else (
                            "MEDIUM"
                            if liq_distance < 20 or position_risk_pct > 15
                            else "LOW"
                        )
                    ),
                }

                position_risks.append(position_risk)
                total_position_value += position.usd_size

            except Exception as e:
                position_risks.append(
                    {
                        "pair": position.pair,
                        "error": f"Could not calculate risk: {str(e)}",
                    }
                )

        # Calculate portfolio-level VaR for each horizon
        portfolio_var = {}
        portfolio_var_1h_pct = None

        for horizon_name in time_horizons.keys():
            if horizon_name == "1w":
                # Use sqrt scaling for 1 week based on 1h VaR
                if portfolio_var_1h_pct is not None:
                    var_pct = portfolio_var_1h_pct * np.sqrt(168)
                    var_dollar = var_pct / 100 * total_position_value
                    portfolio_var[horizon_name] = {
                        "pct": round(var_pct, 2),
                        "usd": round(var_dollar, 2),
                        "pct_of_balance": round((var_dollar / balance.total) * 100, 2),
                    }
            else:
                returns = portfolio_returns_by_horizon[horizon_name]
                if returns is not None and len(returns) > 0:
                    var_pct = abs(np.percentile(returns * 100, 1))  # 99% confidence
                    var_dollar = var_pct / 100 * total_position_value
                    portfolio_var[horizon_name] = {
                        "pct": round(var_pct, 2),
                        "usd": round(var_dollar, 2),
                        "pct_of_balance": round((var_dollar / balance.total) * 100, 2),
                    }
                    # Store 1h for scaling
                    if horizon_name == "1h":
                        portfolio_var_1h_pct = var_pct

        # Calculate correlation matrix for positions if multiple
        correlation_matrix = {}
        if len(positions) > 1:
            returns_dict = {}
            for position in positions:
                try:
                    ohlcv = await client.get_last_ohlcv(
                        position.pair, horizon_name, limit=100
                    )
                    returns_dict[position.pair] = ohlcv["close"].pct_change().dropna()
                except:
                    continue

            if len(returns_dict) > 1:
                returns_df = pd.DataFrame(returns_dict)
                corr = returns_df.corr()
                correlation_matrix = corr.to_dict()

        # Risk summary
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        risk_summary = {
            "total_positions": len(positions),
            "total_exposure_usd": round(total_position_value, 2),
            "exposure_pct_of_balance": round(
                (total_position_value / balance.total) * 100, 2
            ),
            "total_unrealized_pnl": round(total_unrealized_pnl, 2),
            "avg_leverage": round(
                sum(p.leverage for p in positions) / len(positions), 2
            ),
            "high_risk_positions": sum(
                1
                for p in position_risks
                if isinstance(p, dict) and p.get("risk_score") == "HIGH"
            ),
            "diversification": (
                "LOW"
                if len(positions) == 1
                else "MEDIUM" if len(positions) <= 3 else "HIGH"
            ),
            "recommendation": get_risk_recommendation(
                position_risks, portfolio_var, balance.total
            ),
        }

        return {
            "portfolio_var": portfolio_var,
            "position_risks": position_risks,
            "risk_summary": risk_summary,
            "correlation_matrix": (
                correlation_matrix
                if correlation_matrix
                else "N/A - Single position or data unavailable"
            ),
        }

    except Exception as e:
        return {"error": f"Error calculating position risk: {str(e)}"}


def get_risk_recommendation(
    position_risks: List[Dict], portfolio_var: Dict, total_balance: float
) -> str:
    """Generate risk recommendation based on analysis"""
    recommendations = []

    # Check daily VaR levels (1d horizon)
    if portfolio_var and "1d" in portfolio_var:
        var_1d_amount = portfolio_var["1d"].get("usd", 0)
        if var_1d_amount > total_balance * 0.2:
            recommendations.append(
                "Critical: Potential 1-day loss exceeds 20% of balance"
            )
        elif var_1d_amount > total_balance * 0.1:
            recommendations.append(
                "Warning: Potential 1-day loss exceeds 10% of balance"
            )

    # Check weekly VaR
    if portfolio_var and "1w" in portfolio_var:
        var_1w_pct = portfolio_var["1w"].get("pct_of_balance", 0)
        if var_1w_pct > 50:
            recommendations.append("Critical: Weekly VaR exceeds 50% of balance")

    # Check individual positions
    high_risk_count = sum(
        1
        for p in position_risks
        if isinstance(p, dict) and p.get("risk_score") == "HIGH"
    )
    if high_risk_count > 0:
        recommendations.append(
            f"{high_risk_count} position(s) at high risk - consider reducing size or adding stops"
        )

    # Check concentration
    max_weight = max(
        (p.get("position_weight", 0) for p in position_risks if isinstance(p, dict)),
        default=0,
    )
    if max_weight > 30:
        recommendations.append(
            "Position concentration too high - consider diversifying"
        )

    if not recommendations:
        recommendations.append("Risk levels within acceptable range")

    return " | ".join(recommendations)


if __name__ == "__main__":
    mcp.run()
