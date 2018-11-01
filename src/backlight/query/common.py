from urllib.parse import urlparse


def adapter_factory(url, **kwargs):
    from .csv_adapter import CSVAdapter, S3CSVAdapter
    from .csv_glob_adapter import CSVGlobAdapter, S3CSVGlobAdapter
    from .kdb_adapter import KDBAdapter
    from .mktsdb_adapter import MarketstoreAdapter
    from .merge_adapter import MergeAdapter
    from .rds_adapter import RDSAdapter

    if hasattr(url, "__iter__") and not isinstance(url, str):  # check if iterable
        return MergeAdapter(url, **kwargs)

    o = urlparse(url)
    if o.scheme in ("file",) and "*" in o.path:
        cls = CSVGlobAdapter
    elif o.scheme in ("file",) and "*" not in o.path:
        cls = CSVAdapter
    elif o.scheme in ("s3",) and "*" in o.path:
        cls = S3CSVGlobAdapter
    elif o.scheme in ("s3",) and "*" not in o.path:
        cls = S3CSVAdapter
    elif o.scheme in ("kdb",):
        cls = KDBAdapter
    elif o.scheme in ("mktsdb",):
        cls = MarketstoreAdapter
    elif o.scheme in ("rds",):
        cls = RDSAdapter
    else:
        raise NotImplementedError("Unsupported url: {}".format(url))
    return cls(url, **kwargs)
