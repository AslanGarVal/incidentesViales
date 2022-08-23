import os
import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
# metrics to evaluate
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_absolute_error


path_to_processed = 'C:\\Users\\Aslan García\\PycharmProjects\\incidentesViales\\data\\processed'
processed_file_path = os.path.join(path_to_processed, 'incidentes_viales_dataset.csv')
target_encoding_path = os.path.join(path_to_processed, 'delegacion_target_encoding.csv')


def read_split_dataset(path_to_dataset: str) -> (pd.DataFrame, pd.DataFrame):
    # read the csv dataset
    incidentes_df = pd.read_csv(path_to_dataset)

    # drop nulls (caused by the lagging function) and sort by creation timestamp before splitting
    return train_test_split(incidentes_df.dropna().sort_values('creacion_timestamp'), shuffle=False)


def label_encode(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cat_col: str,
        target: str
) -> (pd.DataFrame, pd.DataFrame):
    # use the training set to target encode categorical variable
    label_encoding = train_df.groupby(cat_col).agg({'incidente': 'mean'}). \
        rename({target: 'tgt_enc_deleg'}, axis=1)

    # join to existing training and test datasets
    train_df = train_df.merge(label_encoding, left_on=cat_col, right_index=True). \
        drop([cat_col, 'creacion_timestamp', 'timestamp'], axis=1)
    test_df = test_df.merge(label_encoding, left_on=cat_col, right_index=True). \
        drop([cat_col, 'creacion_timestamp', 'timestamp'], axis=1)

    # save target encoding for future reference
    label_encoding.to_csv(target_encoding_path)

    return train_df, test_df


def score(X: pd.DataFrame, y: pd.DataFrame, model: XGBRegressor) -> None:
    print(f'Mean Poisson deviance: {mean_poisson_deviance(y, model.predict(X))}')
    print(f'Mean absolute error: {mean_absolute_error(y, model.predict(X))}')


def train_model(model: XGBRegressor, label: str, train_df: pd.DataFrame) -> None:
    X_train = train_df.drop([label], axis='columns')
    y_train = train_df[label]
    model.fit(X_train, y_train)

    print('training set metrics -')
    score(X_train, y_train, model)


def evaluate_model(trained_model: XGBRegressor, label: str, test_df: pd.DataFrame) -> None:
    X_test = test_df.drop([label], axis='columns')
    y_test = test_df[label]

    print('test score metrics -')
    score(X_test, y_test, trained_model)


train_data, test_data = read_split_dataset(processed_file_path)
train_data, test_data = label_encode(train_data, test_data, 'delegacion', 'incidente')

# model to train, based on preliminary EDA and ML experiments
gb_model = XGBRegressor(objective='reg:tweedie', max_depth=5, colsample_bytree=0.75, n_estimators=65, reg_alpha=5.5)

train_model(gb_model, 'incidente', train_data)
evaluate_model(gb_model, 'incidente', test_data)

print(train_data.columns)
# serialise trained model
gb_model.save_model('car_incidents_model.json')