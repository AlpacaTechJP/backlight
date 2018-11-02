from urllib.parse import urlparse


def adapter_factory(url, **kwargs):

    if hasattr(url, "__iter__") and not isinstance(url, str):  # check if iterable
        from backlight.query.adapters.merge import MergeAdapter

        return MergeAdapter(url, **kwargs)

    o = urlparse(url)
    if o.scheme in ("file",) and "*" in o.path:
        from backlight.query.adapters.csv_glob import CSVGlobAdapter

        cls = CSVGlobAdapter
    elif o.scheme in ("file",) and "*" not in o.path:
        from backlight.query.adapters.csv import CSVAdapter

        cls = CSVAdapter
    elif o.scheme in ("s3",) and "*" in o.path:
        from backlight.query.adapters.csv_glob import S3CSVGlobAdapter

        cls = S3CSVGlobAdapter
    elif o.scheme in ("s3",) and "*" not in o.path:
        from backlight.query.adapters.csv import S3CSVAdapter

        cls = S3CSVAdapter
    elif o.scheme in ("kdb",):
        from backlight.query.adapters.kdb import KDBAdapter

        cls = KDBAdapter
    elif o.scheme in ("mktsdb",):
        from backlight.query.adapters.mktsdb import MarketstoreAdapter

        cls = MarketstoreAdapter
    elif o.scheme in ("rds",):
        from backlight.query.adapters.rds import RDSAdapter

        cls = RDSAdapter
    else:
        raise NotImplementedError("Unsupported url: {}".format(url))
    return cls(url, **kwargs)
