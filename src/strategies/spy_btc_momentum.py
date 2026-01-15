from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from datetime import datetime
from datetime import timezone

from loguru import logger

from snapper.core.types import TradeSide
from snapper.messaging.schemas.messages import BarEnvelope
from snapper.strategies.base import BaseStrategy
from snapper.strategies.base import Signal
from snapper.strategies.base import StrategyConfig
from snapper.strategies.decorators import create_strategy_process
from snapper.strategies.decorators import register_strategy


@dataclass
class HourlyBar:
    bar_date: date
    hour: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    tick_count: int = 0

    @property
    def bar_key(self) -> tuple[date, int]:
        return (self.bar_date, self.hour)


@dataclass
class OHLCVAggregator:
    current_bar: HourlyBar | None = None
    last_completed_bar: HourlyBar | None = None
    bar_history: deque[HourlyBar] = field(default_factory=lambda: deque(maxlen=24))

    def update(self, price: float, volume: float, timestamp: datetime) -> HourlyBar | None:
        bar_date = timestamp.date()
        hour = timestamp.hour
        bar_key = (bar_date, hour)
        if self.current_bar is None or self.current_bar.bar_key != bar_key:
            if self.current_bar is not None:
                self.last_completed_bar = self.current_bar
                self.bar_history.append(self.current_bar)
            self.current_bar = HourlyBar(
                bar_date=bar_date,
                hour=hour,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                tick_count=1,
            )
            return self.last_completed_bar
        self.current_bar.high = max(self.current_bar.high, price)
        self.current_bar.low = min(self.current_bar.low, price)
        self.current_bar.close = price
        self.current_bar.volume += volume
        self.current_bar.tick_count += 1
        return None

    def reset(self) -> None:
        self.current_bar = None
        self.last_completed_bar = None
        self.bar_history.clear()


@register_strategy("SPYBTCMomentum")
@create_strategy_process(
    process_name="strategy_spy_btc_momentum",
    default_config={
        "name": "spy_btc_momentum",
        "inputs": [
            "market.kraken.SPY.candles.1m",
            "market.kraken.BTC-USD.candles.1m",
        ],
        "outputs": ["BTC-USD"],
        "exchange": "paper",
        "params": {
            "spy_threshold": 0.002,
            "us_session_start_utc": 14,
            "us_session_end_utc": 21,
            "spy_instrument": "SPY",
            "btc_instrument": "BTC-USD",
        },
    },
)
class SPYBTCMomentum(BaseStrategy):
    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.spy_threshold = self.params.get("spy_threshold", 0.002)
        self.us_session_start = self.params.get("us_session_start_utc", 14)
        self.us_session_end = self.params.get("us_session_end_utc", 21)
        self.spy_instrument = self.params.get("spy_instrument", "SPY")
        self.btc_instrument = self.params.get("btc_instrument", "BTC-USD")
        self._spy_aggregator = OHLCVAggregator()
        self._btc_aggregator = OHLCVAggregator()
        self._current_hour: int | None = None
        self._last_signal_bar: tuple[date, int] | None = None

    def _match_instrument(self, instrument: str, target: str) -> bool:
        if instrument == target:
            return True
        parts = instrument.split(".")
        return target in parts

    async def on_bar(self, instrument: str, bar: BarEnvelope) -> Signal | None:
        dt = bar.timestamp
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        self._current_hour = dt.hour
        if self._match_instrument(instrument, self.spy_instrument):
            completed_bar = self._spy_aggregator.update(bar.close, bar.volume, dt)
            if completed_bar:
                logger.debug(f"SPY hourly bar completed: {completed_bar.close:.2f}")
            return None
        if self._match_instrument(instrument, self.btc_instrument):
            self._btc_aggregator.update(bar.close, bar.volume, dt)
            logger.debug(f"BTC update: {bar.close:.2f}")
            return self._generate_signal(dt.date())
        return None

    def _generate_signal(self, current_date: date) -> Signal | None:
        spy_last = self._spy_aggregator.last_completed_bar
        spy_current = self._spy_aggregator.current_bar
        btc_current = self._btc_aggregator.current_bar
        if spy_last is None or spy_current is None:
            return None
        if btc_current is None:
            return None
        if self._current_hour is None:
            return None
        if not (self.us_session_start <= self._current_hour < self.us_session_end):
            logger.debug(f"Outside US session (hour={self._current_hour}), no signal")
            return None
        current_bar_key = (current_date, self._current_hour)
        if self._last_signal_bar == current_bar_key:
            return None
        spy_return = (spy_current.close - spy_last.close) / spy_last.close
        if abs(spy_return) < self.spy_threshold:
            return None
        self._last_signal_bar = current_bar_key
        side: TradeSide = "buy" if spy_return > 0 else "sell"
        strength = min(abs(spy_return) / self.spy_threshold, 1.0)
        return Signal(
            instrument=self.btc_instrument,
            side=side,
            strength=strength,
            price=btc_current.close,
            reason=f"SPY momentum {spy_return * 100:+.2f}% -> {side.upper()} BTC",
            metadata={
                "spy_return": spy_return,
                "spy_threshold": self.spy_threshold,
                "spy_close": spy_current.close,
                "spy_last_close": spy_last.close,
                "hour_utc": self._current_hour,
            },
        )

    async def reset(self) -> None:
        self._spy_aggregator.reset()
        self._btc_aggregator.reset()
        self._current_hour = None
        self._last_signal_bar = None
        logger.info(f"Strategy {self.name} reset")
