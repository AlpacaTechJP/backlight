from abc import ABC, abstractmethod
import pandas as pd


class DataSourceAdapter(ABC):
    @abstractmethod
    def query(
        self, symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp
    ) -> pd.DataFrame:
        """Query pandas dataframe.

        Args:
            symbol    : Symbol name.
            start_dt  : Start date of dataframe.
            end_dt    : End date of dataframe.

        Returns:
            A pandas DataFrame indexed by date from `start_dt` to `end_dt`.
        """
        pass
