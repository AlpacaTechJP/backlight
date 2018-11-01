from backlight.query import common as module

from backlight.query.csv_adapter import CSVAdapter, S3CSVAdapter
from backlight.query.kdb_adapter import KDBAdapter
from backlight.query.mktsdb_adapter import MarketstoreAdapter
from backlight.query.merge_adapter import MergeAdapter
from backlight.query.rds_adapter import RDSAdapter


def test_adapter_factory():
    url = ["file://hoge.csv", "file://huga.csv"]
    adapter = module.adapter_factory(url)
    assert isinstance(adapter, MergeAdapter)

    url = "file://hoge.csv"
    adapter = module.adapter_factory(url)
    assert isinstance(adapter, CSVAdapter)

    url = "s3://hoge.csv"
    adapter = module.adapter_factory(url)
    assert isinstance(adapter, S3CSVAdapter)

    url = "kdb://hoge.csv"
    adapter = module.adapter_factory(url)
    assert isinstance(adapter, KDBAdapter)

    url = "mktsdb://hoge.csv"
    adapter = module.adapter_factory(url)
    assert isinstance(adapter, MarketstoreAdapter)

    url = "rds://hoge.csv"
    adapter = module.adapter_factory(url)
    assert isinstance(adapter, RDSAdapter)
