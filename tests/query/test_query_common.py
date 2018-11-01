from pome.query import common as module

from pome.query.csv_adapter import CSVAdapter, S3CSVAdapter
from pome.query.kdb_adapter import KDBAdapter
from pome.query.mktsdb_adapter import MarketstoreAdapter
from pome.query.merge_adapter import MergeAdapter
from pome.query.rds_adapter import RDSAdapter


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
