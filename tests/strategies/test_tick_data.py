from backlight.strategies import tick_data as module
import pytest
import itertools
import numpy as np
import pandas as pd


def get_columns(depth):
    """Get marketdepth columns."""
    return list(
        itertools.chain(
            *[
                (
                    "l{}-askprice".format(i),
                    "l{}-bidprice".format(i),
                    "l{}-askcount".format(i),
                    "l{}-bidcount".format(i),
                    "l{}-askamount".format(i),
                    "l{}-bidamount".format(i),
                )
                for i in range(1, depth + 1)
            ]
        )
    )


def _cumsum_prices(N, tick_size, depth, p_skip_lvl=0.05, volatility=0.2):
    """Price levels stacked with at least tick_size diff between levels"""
    sa = np.abs(np.cumsum(np.random.randn(N) * (tick_size * volatility)))
    ask = [sa]
    for r in range(depth - 1):
        ex = np.where(
            np.random.rand(N) > p_skip_lvl,
            np.full(N, tick_size),
            np.full(N, tick_size * 2),
        )
        ask.append(ask[-1] + ex)
    ask = np.array(ask).T
    bid = [sa - tick_size]
    for r in range(depth - 1):
        ex = np.where(
            np.random.rand(N) > p_skip_lvl,
            np.full(N, tick_size),
            np.full(N, tick_size * 2),
        )
        bid.append(bid[-1] - ex)
    bid = np.array(bid).T
    pad = np.abs(min(0, np.min(bid))) + 0.1
    ask, bid = ask + pad, bid + pad
    return _round_to_tick(ask, tick_size), _round_to_tick(bid, tick_size)


def _round_to_tick(a, tick_size):
    # Without numpy: decimal.Decimal(str(tick_size)).as_tuple().exponent
    expo = np.floor(np.log10(np.abs(tick_size))).astype(int)
    return np.around(np.around(a / tick_size) * tick_size, decimals=-1 * expo)


def mock_query_tick(start, end, depth=10, seed=42, tick_size=0.005):
    """Create a DataFrame with tick-like data."""
    np.random.seed(seed=seed)
    index = pd.date_range(start, end, freq="10ms", closed="left")
    columns = get_columns(depth=depth)
    df = pd.DataFrame(
        index=index, data=np.random.rand(len(index), len(columns)), columns=columns
    )
    df = df.sample(frac=0.1).sort_index()
    # Rescale amount to typical ranges
    am = df.filter(regex="amount").columns
    df[am] = (df[am] * 1.2e7 + 1).astype(int)

    # Rescale count to typical ranges
    co = df.filter(regex="count").columns
    df[co] = (df[co] * 9.0 + 1).astype(int)

    # Better prices
    ask, bid = _cumsum_prices(len(df), tick_size, depth)
    df[df.filter(regex="askprice").columns] = ask
    df[df.filter(regex="bidprice").columns] = bid

    return df.astype(np.float32)


@pytest.fixture
def sample_df():
    return mock_query_tick("2018-01-01 00:00:00", "2018-01-01 00:00:10")


def test_ticks_around(sample_df):
    sim = module.TickDataSimulator(sample_df)
    with pytest.raises(module.InvalidTrade):
        sim._ticks_around(pd.Timestamp("2042-01-01"))

    res = sim._ticks_around(sample_df.index[12])
    assert res.equals(sample_df[: sample_df.index[12]])


def test_get_pair_no_slippage(sample_df):
    sim = module.TickDataSimulator(sample_df)
    entry, exit = sim._get_pair_no_slippage(sample_df.index[24], sample_df.index[42])

    assert entry.index == pd.Timestamp("2018-01-01 00:00:02.180000")
    assert np.equal(entry.value, np.float32(0.1675))
    assert exit.index == pd.Timestamp("2018-01-01 00:00:04.490000")
    assert np.equal(exit.value, np.float32(0.1575))


def test_get_pair_long(sample_df):
    sim = module.TickDataSimulator(sample_df)
    entry, exit = sim._get_pair_long(sample_df.index[24], sample_df.index[42])

    assert entry.index == pd.Timestamp("2018-01-01 00:00:02.180000")
    assert np.equal(entry.value, np.float32(0.17))
    assert exit.index == pd.Timestamp("2018-01-01 00:00:04.490000")
    assert np.equal(exit.value, np.float32(0.155))


def test_get_pair_short(sample_df):
    sim = module.TickDataSimulator(sample_df)
    entry, exit = sim._get_pair_short(sample_df.index[24], sample_df.index[42])

    assert entry.index == pd.Timestamp("2018-01-01 00:00:02.180000")
    assert np.equal(entry.value, np.float32(0.165))
    assert exit.index == pd.Timestamp("2018-01-01 00:00:04.490000")
    assert np.equal(exit.value, np.float32(0.16))


def test_get_entry_exit_pair(sample_df):
    sim = module.TickDataSimulator(sample_df)
    entry, exit = sim.get_entry_exit_pair(sample_df.index[24], sample_df.index[42], 1)

    assert entry.index == pd.Timestamp("2018-01-01 00:00:02.180000")
    assert np.equal(entry.value, np.float32(0.17))
    assert exit.index == pd.Timestamp("2018-01-01 00:00:04.490000")
    assert np.equal(exit.value, np.float32(0.155))

    entry, exit = sim.get_entry_exit_pair(sample_df.index[24], sample_df.index[42], -1)

    assert entry.index == pd.Timestamp("2018-01-01 00:00:02.180000")
    assert np.equal(entry.value, np.float32(0.165))
    assert exit.index == pd.Timestamp("2018-01-01 00:00:04.490000")
    assert np.equal(exit.value, np.float32(0.16))


def test_get_pl(sample_df):
    sim = module.TickDataSimulator(sample_df)

    pl = sim.get_pl(sample_df.index[24], sample_df.index[42], 1)
    np.testing.assert_allclose(pl, -0.015, atol=1e-05, rtol=0)

    pl = sim.get_pl(sample_df.index[24], sample_df.index[42], -1)
    np.testing.assert_allclose(pl, 0.005, atol=1e-05, rtol=0)


def test_iter_on_pairs(sample_df):
    sim = module.TickDataSimulator(sample_df)
    pairs = [
        [[sample_df.index[i], j], [sample_df.index[i + 42]]]
        for i, j in zip([10, 30], [1, -1])
    ]

    it = sim.iter_on_pairs(pairs)

    res = next(it)
    np.testing.assert_allclose(res["pl"], -0.005, atol=1e-05, rtol=0)
    np.testing.assert_allclose(res["entry_value"], 0.165, atol=1e-05, rtol=0)
    np.testing.assert_allclose(res["exit_value"], 0.16, atol=1e-05, rtol=0)
    assert res["entry"] == pd.Timestamp("2018-01-01 00:00:01.190000")
    assert res["entry_preceding_tick"] == pd.Timestamp("2018-01-01 00:00:01.190000")
    assert res["exit"] == pd.Timestamp("2018-01-01 00:00:05.260000")
    assert res["exit_preceding_tick"] == pd.Timestamp("2018-01-01 00:00:05.260000")

    res = next(it)
    np.testing.assert_allclose(res["pl"], 0.005, atol=1e-05, rtol=0)
