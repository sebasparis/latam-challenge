import pandas as pd

from datetime import datetime
from typing import List, Tuple, Union

import xgboost as xgb


class DelayModel:

    FEATURE_COLUMNS = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]

    def __init__(
        self
    ):
        self._model = None  # Model should be saved in this attribute.

    @staticmethod
    def _get_period_day(date: str) -> str:
        """Return the period of the day based on the flight scheduled time."""
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("04:59", '%H:%M').time()

        if morning_min <= date_time <= morning_max:
            return 'mañana'
        if afternoon_min <= date_time <= afternoon_max:
            return 'tarde'
        if (evening_min <= date_time <= evening_max) or (night_min <= date_time <= night_max):
            return 'noche'
        return 'noche'

    @staticmethod
    def _is_high_season(fecha: str) -> int:
        """Return 1 if the flight date falls within high-season windows."""
        fecha_año = int(fecha.split('-')[0])
        fecha_dt = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')

        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

        if (
            (range1_min <= fecha_dt <= range1_max)
            or (range2_min <= fecha_dt <= range2_max)
            or (range3_min <= fecha_dt <= range3_max)
            or (range4_min <= fecha_dt <= range4_max)
        ):
            return 1
        return 0

    @staticmethod
    def _get_min_diff(row: pd.Series) -> float:
        fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        return ((fecha_o - fecha_i).total_seconds()) / 60

    @staticmethod
    def _build_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
        if 'Fecha-I' in data.columns:
            data = data.copy()
            data['period_day'] = data['Fecha-I'].apply(DelayModel._get_period_day)
            data['high_season'] = data['Fecha-I'].apply(DelayModel._is_high_season)
            if {'Fecha-O', 'Fecha-I'}.issubset(data.columns):
                data['min_diff'] = data.apply(DelayModel._get_min_diff, axis=1)

        feature_frame = pd.concat(
            [
                pd.get_dummies(data['OPERA'], prefix='OPERA'),
                pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
                pd.get_dummies(data['MES'], prefix='MES'),
            ],
            axis=1,
        )

        for column_name in DelayModel.FEATURE_COLUMNS:
            if column_name not in feature_frame.columns:
                feature_frame[column_name] = 0

        return feature_frame[DelayModel.FEATURE_COLUMNS]

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data = data.copy()

        if 'Fecha-I' in data.columns:
            data['period_day'] = data['Fecha-I'].apply(self._get_period_day)
            data['high_season'] = data['Fecha-I'].apply(self._is_high_season)
            if {'Fecha-O', 'Fecha-I'}.issubset(data.columns):
                data['min_diff'] = data.apply(self._get_min_diff, axis=1)
                data['delay'] = (data['min_diff'] > 15).astype(int)

        features = self._build_feature_matrix(data)

        if target_column is not None:
            target = pd.DataFrame({target_column: data[target_column].astype(int)})
            return features, target

        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        target_series = target.iloc[:, 0].astype(int)
        n_y0 = len(target_series[target_series == 0])
        n_y1 = len(target_series[target_series == 1])
        scale = n_y0 / n_y1 if n_y1 else 1.0

        self._model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=scale,
        )
        self._model.fit(features, target_series)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            print('Warning: model was not fitted before predicting. Predicting all labels as 0.')
            return [0 for _ in range(len(features))]

        predictions = self._model.predict(features)
        return [int(prediction) for prediction in predictions]