from urllib.parse import urlparse
from typing import Type

from backlight.query.adapter import DataSourceAdapter


def adapter_factory(url: str, **kwargs: str) -> DataSourceAdapter:

    if hasattr(url, "__iter__") and not isinstance(url, str):  # check if iterable
        from backlight.query.adapters.merge import MergeAdapter

        return MergeAdapter(url, **kwargs)

    o = urlparse(url)
    if o.scheme in ("file",) and "*" in o.path:
        from backlight.query.adapters.csv_glob import CSVGlobAdapter

        return CSVGlobAdapter(url)
    elif o.scheme in ("file",) and "*" not in o.path:
        if o.path.endswith(".h5"):
            from backlight.query.adapters.h5 import H5Adapter

            return H5Adapter(url)
        else:
            from backlight.query.adapters.csv import CSVAdapter

            return CSVAdapter(url)
    elif o.scheme in ("s3",) and "*" in o.path:
        from backlight.query.adapters.csv_glob import S3CSVGlobAdapter

        return S3CSVGlobAdapter(url)
    elif o.scheme in ("s3",) and "*" not in o.path:
        from backlight.query.adapters.csv import S3CSVAdapter

        return S3CSVAdapter(url)
    elif o.scheme in ("kdb",):
        from backlight.query.adapters.kdb import KDBAdapter

        return KDBAdapter(url)
    elif o.scheme in ("mktsdb",):
        from backlight.query.adapters.mktsdb import MarketstoreAdapter

        return MarketstoreAdapter(url, **kwargs)
    elif o.scheme in ("rds",):
        from backlight.query.adapters.rds import RDSAdapter

        return RDSAdapter(url)
    else:
        raise NotImplementedError("Unsupported url: {}".format(url))
