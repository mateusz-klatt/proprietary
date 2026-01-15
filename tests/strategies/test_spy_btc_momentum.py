from datetime import date
from datetime import datetime
from datetime import timezone

import pytest
from strategies.spy_btc_momentum import HourlyBar
from strategies.spy_btc_momentum import OHLCVAggregator
from strategies.spy_btc_momentum import SPYBTCMomentum

from snapper.messaging.schemas.messages import BarEnvelope
from snapper.strategies.base import Signal
from snapper.strategies.base import StrategyConfig


def make_bar_envelope(
    instrument: str,
    close: float,
    ts: datetime,
    volume: float = 0.0,
    exchange: str = "paper",
) -> BarEnvelope:
    return BarEnvelope(
        instrument=instrument,
        timeframe="1m",
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=volume,
        exchange=exchange,
        timestamp=ts,
    )


async def feed_bar_to_strategy(
    strategy: SPYBTCMomentum,
    instrument: str,
    close: float,
    ts: datetime,
    volume: float = 0.0,
) -> Signal | None:
    bar = make_bar_envelope(instrument, close, ts, volume)
    result: Signal | None = await strategy.on_bar(instrument, bar)
    return result


@pytest.fixture
def config() -> StrategyConfig:
    return StrategyConfig(
        name="spy_btc_momentum_test",
        strategy_class="SPYBTCMomentum",
        inputs=["market.paper.SPY.candles.1m", "market.paper.BTC-USD.candles.1m"],
        outputs=["BTC-USD"],
        exchange="paper",
        params={
            "spy_threshold": 0.002,
            "us_session_start_utc": 14,
            "us_session_end_utc": 21,
            "spy_instrument": "SPY",
            "btc_instrument": "BTC-USD",
        },
    )


@pytest.fixture
def strategy(config: StrategyConfig) -> SPYBTCMomentum:
    return SPYBTCMomentum(config)


class TestHourlyBar:
    def test_creation(self) -> None:
        bar = HourlyBar(
            bar_date=date(2025, 1, 15), hour=14, open=100.0, high=105.0, low=99.0, close=102.0
        )
        assert bar.bar_date == date(2025, 1, 15)
        assert bar.hour == 14
        assert bar.open == 100.0
        assert bar.high == 105.0
        assert bar.low == 99.0
        assert bar.close == 102.0
        assert bar.volume == 0.0
        assert bar.tick_count == 0

    def test_with_volume(self) -> None:
        bar = HourlyBar(
            bar_date=date(2025, 1, 15),
            hour=15,
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1000.0,
        )
        assert bar.volume == 1000.0

    def test_bar_key(self) -> None:
        bar = HourlyBar(
            bar_date=date(2025, 1, 15), hour=14, open=100.0, high=100.0, low=100.0, close=100.0
        )
        assert bar.bar_key == (date(2025, 1, 15), 14)


class TestOHLCVAggregator:
    def test_initial_state(self) -> None:
        agg = OHLCVAggregator()
        assert agg.current_bar is None
        assert agg.last_completed_bar is None
        assert len(agg.bar_history) == 0

    def test_first_update_creates_bar(self) -> None:
        agg = OHLCVAggregator()
        dt = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        result = agg.update(100.0, 50.0, dt)
        assert result is None
        assert agg.current_bar is not None
        assert agg.current_bar.bar_date == date(2025, 1, 15)
        assert agg.current_bar.hour == 14
        assert agg.current_bar.open == 100.0
        assert agg.current_bar.close == 100.0
        assert agg.current_bar.volume == 50.0
        assert agg.current_bar.tick_count == 1

    def test_same_hour_updates_bar(self) -> None:
        agg = OHLCVAggregator()
        dt1 = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        agg.update(100.0, 50.0, dt1)
        result = agg.update(105.0, 60.0, dt2)
        assert result is None
        assert agg.current_bar is not None
        assert agg.current_bar.open == 100.0
        assert agg.current_bar.high == 105.0
        assert agg.current_bar.close == 105.0
        assert agg.current_bar.volume == 110.0
        assert agg.current_bar.tick_count == 2

    def test_low_updated_correctly(self) -> None:
        agg = OHLCVAggregator()
        dt1 = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        agg.update(100.0, 50.0, dt1)
        agg.update(95.0, 60.0, dt2)
        assert agg.current_bar is not None
        assert agg.current_bar.low == 95.0

    def test_hour_change_completes_bar(self) -> None:
        agg = OHLCVAggregator()
        dt1 = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        dt2 = datetime(2025, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
        agg.update(100.0, 50.0, dt1)
        result = agg.update(105.0, 60.0, dt2)
        assert result is not None
        assert result.hour == 14
        assert result.close == 100.0
        assert agg.last_completed_bar == result
        assert agg.current_bar is not None
        assert agg.current_bar.hour == 15

    def test_midnight_crossing_creates_new_bar(self) -> None:
        agg = OHLCVAggregator()
        dt1 = datetime(2025, 1, 15, 23, 30, 0, tzinfo=timezone.utc)
        dt2 = datetime(2025, 1, 16, 0, 0, 0, tzinfo=timezone.utc)
        agg.update(100.0, 50.0, dt1)
        result = agg.update(105.0, 60.0, dt2)
        assert result is not None
        assert result.bar_date == date(2025, 1, 15)
        assert result.hour == 23
        assert agg.current_bar is not None
        assert agg.current_bar.bar_date == date(2025, 1, 16)
        assert agg.current_bar.hour == 0

    def test_same_hour_different_day_creates_new_bar(self) -> None:
        agg = OHLCVAggregator()
        dt1 = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        dt2 = datetime(2025, 1, 16, 14, 0, 0, tzinfo=timezone.utc)
        agg.update(100.0, 50.0, dt1)
        result = agg.update(105.0, 60.0, dt2)
        assert result is not None
        assert result.bar_date == date(2025, 1, 15)
        assert agg.current_bar is not None
        assert agg.current_bar.bar_date == date(2025, 1, 16)

    def test_bar_history_populated(self) -> None:
        agg = OHLCVAggregator()
        for hour in range(14, 17):
            dt = datetime(2025, 1, 15, hour, 30, 0, tzinfo=timezone.utc)
            agg.update(100.0 + hour, 50.0, dt)
        assert len(agg.bar_history) == 2
        assert agg.bar_history[0].hour == 14
        assert agg.bar_history[1].hour == 15

    def test_reset_clears_state(self) -> None:
        agg = OHLCVAggregator()
        dt = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        agg.update(100.0, 50.0, dt)
        agg.reset()
        assert agg.current_bar is None
        assert agg.last_completed_bar is None
        assert len(agg.bar_history) == 0


class TestSPYBTCMomentumInit:
    def test_default_params(self, strategy: SPYBTCMomentum) -> None:
        assert strategy.spy_threshold == 0.002
        assert strategy.us_session_start == 14
        assert strategy.us_session_end == 21
        assert strategy.spy_instrument == "SPY"
        assert strategy.btc_instrument == "BTC-USD"

    def test_custom_params(self, config: StrategyConfig) -> None:
        config.params = {
            "spy_threshold": 0.005,
            "us_session_start_utc": 13,
            "us_session_end_utc": 20,
            "spy_instrument": "SPYx",
            "btc_instrument": "BTC/USD",
        }
        s = SPYBTCMomentum(config)
        assert s.spy_threshold == 0.005
        assert s.us_session_start == 13
        assert s.us_session_end == 20
        assert s.spy_instrument == "SPYx"
        assert s.btc_instrument == "BTC/USD"

    def test_initial_state(self, strategy: SPYBTCMomentum) -> None:
        assert strategy._spy_aggregator.current_bar is None
        assert strategy._btc_aggregator.current_bar is None
        assert strategy._current_hour is None
        assert strategy._last_signal_bar is None


class TestSPYBTCMomentumInstrumentMatching:
    def test_match_exact_instrument(self, strategy: SPYBTCMomentum) -> None:
        assert strategy._match_instrument("SPY", "SPY") is True
        assert strategy._match_instrument("BTC-USD", "BTC-USD") is True

    def test_match_with_prefix(self, strategy: SPYBTCMomentum) -> None:
        assert strategy._match_instrument("market.paper.SPY.candles.1m", "SPY") is True
        assert strategy._match_instrument("market.kraken.BTC-USD.candles.1m", "BTC-USD") is True

    def test_no_match_substring(self, strategy: SPYBTCMomentum) -> None:
        assert strategy._match_instrument("SPYV", "SPY") is False
        assert strategy._match_instrument("WBTC-USD", "BTC-USD") is False

    def test_no_match_different_instrument(self, strategy: SPYBTCMomentum) -> None:
        assert strategy._match_instrument("QQQ", "SPY") is False
        assert strategy._match_instrument("ETH-USD", "BTC-USD") is False


class TestSPYBTCMomentumOnBar:

    @pytest.mark.asyncio
    async def test_spy_update_aggregates_bar(self, strategy: SPYBTCMomentum) -> None:
        ts = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        await feed_bar_to_strategy(strategy, "SPY", 100.0, ts)
        assert strategy._spy_aggregator.current_bar is not None
        assert strategy._spy_aggregator.current_bar.close == 100.0

    @pytest.mark.asyncio
    async def test_btc_update_aggregates_bar(self, strategy: SPYBTCMomentum) -> None:
        ts = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        await feed_bar_to_strategy(strategy, "BTC-USD", 50000.0, ts)
        assert strategy._btc_aggregator.current_bar is not None
        assert strategy._btc_aggregator.current_bar.close == 50000.0

    @pytest.mark.asyncio
    async def test_timestamp_sets_hour(self, strategy: SPYBTCMomentum) -> None:
        ts = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        await feed_bar_to_strategy(strategy, "BTC-USD", 50000.0, ts)
        assert strategy._current_hour == 16

    @pytest.mark.asyncio
    async def test_naive_timestamp_gets_utc_tzinfo(self, strategy: SPYBTCMomentum) -> None:
        ts_naive = datetime(2025, 1, 15, 16, 0, 0)
        bar = BarEnvelope(
            instrument="BTC-USD",
            timeframe="1m",
            open=49999,
            high=50001,
            low=49998,
            close=50000.0,
            volume=0.0,
            exchange="paper",
            timestamp=ts_naive,
        )
        await strategy.on_bar("BTC-USD", bar)
        assert strategy._current_hour == 16

    @pytest.mark.asyncio
    async def test_volume_extracted(self, strategy: SPYBTCMomentum) -> None:
        ts = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        await feed_bar_to_strategy(strategy, "SPY", 100.0, ts, volume=1000.0)
        assert strategy._spy_aggregator.current_bar is not None
        assert strategy._spy_aggregator.current_bar.volume == 1000.0

    @pytest.mark.asyncio
    async def test_volume_defaults_to_zero(self, strategy: SPYBTCMomentum) -> None:
        ts = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        await feed_bar_to_strategy(strategy, "SPY", 100.0, ts)
        assert strategy._spy_aggregator.current_bar is not None
        assert strategy._spy_aggregator.current_bar.volume == 0.0

    @pytest.mark.asyncio
    async def test_unknown_instrument_returns_none(self, strategy: SPYBTCMomentum) -> None:
        ts = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        result = await feed_bar_to_strategy(strategy, "ETH-USD", 3000.0, ts)
        assert result is None

    @pytest.mark.asyncio
    async def test_spy_hourly_bar_completion_logged(self, strategy: SPYBTCMomentum) -> None:
        ts1 = datetime(2025, 1, 15, 15, 30, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        await feed_bar_to_strategy(strategy, "SPY", 100.0, ts1)
        assert strategy._spy_aggregator.current_bar is not None
        assert strategy._spy_aggregator.current_bar.hour == 15
        await feed_bar_to_strategy(strategy, "SPY", 101.0, ts2)
        assert strategy._spy_aggregator.last_completed_bar is not None
        assert strategy._spy_aggregator.last_completed_bar.hour == 15
        assert strategy._spy_aggregator.last_completed_bar.close == 100.0
        assert strategy._spy_aggregator.current_bar.hour == 16


class TestSPYBTCMomentumSignalGeneration:
    async def _setup_spy_bars(
        self, strategy: SPYBTCMomentum, spy_prev: float, spy_curr: float, hour: int
    ) -> None:
        ts_prev = datetime(2025, 1, 15, hour - 1, 30, 0, tzinfo=timezone.utc)
        ts_curr = datetime(2025, 1, 15, hour, 0, 0, tzinfo=timezone.utc)
        await feed_bar_to_strategy(strategy, "SPY", spy_prev, ts_prev)
        await feed_bar_to_strategy(strategy, "SPY", spy_curr, ts_curr)

    @pytest.mark.asyncio
    async def test_no_signal_without_completed_spy_bar(self, strategy: SPYBTCMomentum) -> None:
        ts = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        await feed_bar_to_strategy(strategy, "SPY", 100.0, ts)
        result = await feed_bar_to_strategy(strategy, "BTC-USD", 50000.0, ts)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_signal_without_btc_data(self, strategy: SPYBTCMomentum) -> None:
        ts1 = datetime(2025, 1, 15, 15, 30, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        await feed_bar_to_strategy(strategy, "SPY", 100.0, ts1)
        result = await feed_bar_to_strategy(strategy, "SPY", 100.5, ts2)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_signal_spy_completed_but_no_btc_bar(self, strategy: SPYBTCMomentum) -> None:
        ts1 = datetime(2025, 1, 15, 15, 30, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        await feed_bar_to_strategy(strategy, "SPY", 100.0, ts1)
        await feed_bar_to_strategy(strategy, "SPY", 100.5, ts2)
        result = await feed_bar_to_strategy(strategy, "ETH-USD", 3000.0, ts2)
        assert result is None

    def test_generate_signal_with_hour_none(self, strategy: SPYBTCMomentum) -> None:
        strategy._spy_aggregator.last_completed_bar = HourlyBar(
            bar_date=date(2025, 1, 15), hour=15, open=100.0, high=101.0, low=99.0, close=100.0
        )
        strategy._spy_aggregator.current_bar = HourlyBar(
            bar_date=date(2025, 1, 15), hour=16, open=100.5, high=101.0, low=100.0, close=100.5
        )
        strategy._btc_aggregator.current_bar = HourlyBar(
            bar_date=date(2025, 1, 15),
            hour=16,
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50000.0,
        )
        strategy._current_hour = None
        result = strategy._generate_signal(date(2025, 1, 15))
        assert result is None

    def test_generate_signal_with_btc_none(self, strategy: SPYBTCMomentum) -> None:
        strategy._spy_aggregator.last_completed_bar = HourlyBar(
            bar_date=date(2025, 1, 15), hour=15, open=100.0, high=101.0, low=99.0, close=100.0
        )
        strategy._spy_aggregator.current_bar = HourlyBar(
            bar_date=date(2025, 1, 15), hour=16, open=100.5, high=101.0, low=100.0, close=100.5
        )
        strategy._btc_aggregator.current_bar = None
        strategy._current_hour = 16
        result = strategy._generate_signal(date(2025, 1, 15))
        assert result is None

    @pytest.mark.asyncio
    async def test_no_signal_below_threshold(self, strategy: SPYBTCMomentum) -> None:
        await self._setup_spy_bars(strategy, 100.0, 100.1, 16)
        ts = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        result = await feed_bar_to_strategy(strategy, "BTC-USD", 50000.0, ts)
        assert result is None

    @pytest.mark.asyncio
    async def test_buy_signal_on_positive_momentum(self, strategy: SPYBTCMomentum) -> None:
        await self._setup_spy_bars(strategy, 100.0, 100.5, 16)
        ts = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        result = await feed_bar_to_strategy(strategy, "BTC-USD", 50000.0, ts)
        assert result is not None
        assert result.side == "buy"
        assert result.instrument == "BTC-USD"
        assert result.price == 50000.0
        assert "BUY" in result.reason

    @pytest.mark.asyncio
    async def test_sell_signal_on_negative_momentum(self, strategy: SPYBTCMomentum) -> None:
        await self._setup_spy_bars(strategy, 100.0, 99.5, 16)
        ts = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        result = await feed_bar_to_strategy(strategy, "BTC-USD", 50000.0, ts)
        assert result is not None
        assert result.side == "sell"
        assert "SELL" in result.reason

    @pytest.mark.asyncio
    async def test_no_signal_outside_us_session(self, strategy: SPYBTCMomentum) -> None:
        await self._setup_spy_bars(strategy, 100.0, 100.5, 10)
        ts = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = await feed_bar_to_strategy(strategy, "BTC-USD", 50000.0, ts)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_signal_at_session_end_boundary(self, strategy: SPYBTCMomentum) -> None:
        await self._setup_spy_bars(strategy, 100.0, 100.5, 21)
        ts = datetime(2025, 1, 15, 21, 0, 0, tzinfo=timezone.utc)
        result = await feed_bar_to_strategy(strategy, "BTC-USD", 50000.0, ts)
        assert result is None

    @pytest.mark.asyncio
    async def test_signal_at_session_start_boundary(self, strategy: SPYBTCMomentum) -> None:
        await self._setup_spy_bars(strategy, 100.0, 100.5, 14)
        ts = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        result = await feed_bar_to_strategy(strategy, "BTC-USD", 50000.0, ts)
        assert result is not None
        assert result.side == "buy"

    @pytest.mark.asyncio
    async def test_signal_metadata_content(self, strategy: SPYBTCMomentum) -> None:
        await self._setup_spy_bars(strategy, 100.0, 100.5, 16)
        ts = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        result = await feed_bar_to_strategy(strategy, "BTC-USD", 50000.0, ts)
        assert result is not None
        assert result.metadata is not None
        assert "spy_return" in result.metadata
        assert result.metadata["spy_return"] == pytest.approx(0.005, rel=1e-3)
        assert result.metadata["spy_threshold"] == 0.002
        assert result.metadata["spy_close"] == 100.5
        assert result.metadata["spy_last_close"] == 100.0
        assert result.metadata["hour_utc"] == 16

    @pytest.mark.asyncio
    async def test_signal_strength_capped_at_one(self, strategy: SPYBTCMomentum) -> None:
        await self._setup_spy_bars(strategy, 100.0, 110.0, 16)
        ts = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        result = await feed_bar_to_strategy(strategy, "BTC-USD", 50000.0, ts)
        assert result is not None
        assert result.strength == 1.0


class TestSPYBTCMomentumSignalDedupe:
    async def _setup_spy_bars(
        self, strategy: SPYBTCMomentum, spy_prev: float, spy_curr: float, hour: int
    ) -> None:
        ts_prev = datetime(2025, 1, 15, hour - 1, 30, 0, tzinfo=timezone.utc)
        ts_curr = datetime(2025, 1, 15, hour, 0, 0, tzinfo=timezone.utc)
        await feed_bar_to_strategy(strategy, "SPY", spy_prev, ts_prev)
        await feed_bar_to_strategy(strategy, "SPY", spy_curr, ts_curr)

    @pytest.mark.asyncio
    async def test_no_duplicate_signal_same_bar(self, strategy: SPYBTCMomentum) -> None:
        await self._setup_spy_bars(strategy, 100.0, 100.5, 16)
        ts1 = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 16, 1, 0, tzinfo=timezone.utc)
        result1 = await feed_bar_to_strategy(strategy, "BTC-USD", 50000.0, ts1)
        assert result1 is not None
        result2 = await feed_bar_to_strategy(strategy, "BTC-USD", 50100.0, ts2)
        assert result2 is None

    @pytest.mark.asyncio
    async def test_signal_on_new_hour(self, strategy: SPYBTCMomentum) -> None:
        await self._setup_spy_bars(strategy, 100.0, 100.5, 16)
        ts1 = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        result1 = await feed_bar_to_strategy(strategy, "BTC-USD", 50000.0, ts1)
        assert result1 is not None
        ts_spy = datetime(2025, 1, 15, 17, 0, 0, tzinfo=timezone.utc)
        await feed_bar_to_strategy(strategy, "SPY", 100.8, ts_spy)
        ts2 = datetime(2025, 1, 15, 17, 0, 0, tzinfo=timezone.utc)
        result2 = await feed_bar_to_strategy(strategy, "BTC-USD", 50100.0, ts2)
        assert result2 is not None

    @pytest.mark.asyncio
    async def test_signal_on_new_day_same_hour(self, strategy: SPYBTCMomentum) -> None:
        await self._setup_spy_bars(strategy, 100.0, 100.5, 16)
        ts1 = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        result1 = await feed_bar_to_strategy(strategy, "BTC-USD", 50000.0, ts1)
        assert result1 is not None
        ts_spy1 = datetime(2025, 1, 16, 15, 30, 0, tzinfo=timezone.utc)
        ts_spy2 = datetime(2025, 1, 16, 16, 0, 0, tzinfo=timezone.utc)
        await feed_bar_to_strategy(strategy, "SPY", 101.0, ts_spy1)
        await feed_bar_to_strategy(strategy, "SPY", 101.5, ts_spy2)
        ts2 = datetime(2025, 1, 16, 16, 0, 0, tzinfo=timezone.utc)
        result2 = await feed_bar_to_strategy(strategy, "BTC-USD", 50200.0, ts2)
        assert result2 is not None


class TestSPYBTCMomentumReset:
    @pytest.mark.asyncio
    async def test_reset_clears_state(self, strategy: SPYBTCMomentum) -> None:
        ts1 = datetime(2025, 1, 15, 15, 30, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        await feed_bar_to_strategy(strategy, "SPY", 100.0, ts1)
        await feed_bar_to_strategy(strategy, "SPY", 100.5, ts2)
        await feed_bar_to_strategy(strategy, "BTC-USD", 50000.0, ts2)
        await strategy.reset()
        assert strategy._spy_aggregator.current_bar is None
        assert strategy._spy_aggregator.last_completed_bar is None
        assert strategy._btc_aggregator.current_bar is None
        assert strategy._current_hour is None
        assert strategy._last_signal_bar is None
