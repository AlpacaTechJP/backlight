#!/usr/bin/env Python3
# coding=UTF-8
import datetime
import os
import sys

import pome
from pome import datasource, metrics, signal, simulator
from pome.labelizer import (
    DiffReturnDiffLabelizer,
    FutureDirectionLabelizer,
    ReturnDifferenceLabelizer,
)

root_path = os.getcwd()
# need export FILESTORE_ROOT=/shared/work/chenwei/pome/data
# and create a default folder to put ModelPackage.tar.gz
signal_url = "file://{}/data/sample_signal.csv".format(root_path)
data_url = "file://{}/data/sample_datasource_topix.csv".format(root_path)
sig = signal.load_signal(
    "BBG-USDJPY_Curncy",
    signal_url,
    datetime.datetime(2015, 1, 5),
    datetime.datetime(2017, 1, 5),
)
print("sig", sig.head())
print("sig", sig.pred.head())
mkt = datasource.load_marketdata(sig.symbol, sig._start_dt, sig._end_dt, data_url)
print("mkt", mkt.head())
lbl = pome.generate_labels(
    mkt, FutureDirectionLabelizer(lookahead=3, target_column_name="lag0_dt_close")
)
print("FutureDirectionLabelizer", lbl.head())
lbl = pome.generate_labels(
    mkt,
    ReturnDifferenceLabelizer(
        lookahead=3,
        compared_column_name="lag0_dt_close",
        standard_column_name="lag0_dt_max",
    ),
)
lbl.head()
print("ReturnDifferenceLabelizer", lbl.head())
lbl = pome.generate_labels(
    mkt,
    DiffReturnDiffLabelizer(
        lookahead=3,
        p1_compared_column_name="lag0_dt_max",
        n1_compared_column_name="lag0_dt_min",
        standard_column_name="lag0_dt_close",
    ),
)
print("DiffReturnDiffLabelizer", lbl.head())
trades = simulator.simple_buy_sell(mkt, sig, target_column_name="lag0_dt_close")
print("simple_buy_sell", trades.tail())
trades = simulator.only_take_long(mkt, sig, target_column_name="lag0_dt_close")
print("only_take_long", trades.tail())
mts = metrics.calc_metrics(sig, lbl)
print("binary_metrics.get", mts.get())
print("binary_metrics.to_frame", mts.to_frame())
mts = metrics.calc_trade_performance(sig, lbl, trades)
print("binary_metrics.get", mts.get())
print("binary_metrics.to_frame", mts.to_frame())
