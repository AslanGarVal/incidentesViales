import pandas as pd
import glob
import os

path_to_raw_data = 'C:\\Users\\Aslan García\\PycharmProjects\\incidentesViales\\data\\raw'
path_to_processed = 'C:\\Users\\Aslan García\\PycharmProjects\\incidentesViales\\data\\processed'


def get_all_raw_data(path: str) -> pd.DataFrame:
    raw_files = glob.glob(os.path.join(path, '*.csv'))
    # initialise empty dataframe
    df = pd.DataFrame()
    for file in raw_files:
        # for every raw file append to existing iteration of accumulated df
        file_df = pd.read_csv(file)
        df = pd.concat([df, file_df])

    return df


def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    # create timestamp columns instead of split two columns
    df['creacion_timestamp'] = pd.to_datetime(df['fecha_creacion'] + 'T' + df['hora_creacion'])
    df['cierre_timestamp'] = pd.to_datetime(df['fecha_cierre'] + 'T' + df['hora_cierre'])

    # filter only those incidents registered on a single borough and exclude false alarms
    matching_delegacion = '''delegacion_inicio == delegacion_cierre and clas_con_f_alarma != 'FALSA ALARMA' '''
    # select only relevant columns
    desired_fields = ['folio', 'creacion_timestamp', 'cierre_timestamp', 'delegacion_inicio']
    filtered_data = df.query(matching_delegacion)[desired_fields]

    return filtered_data.rename(columns={'folio': 'incidente'})


def add_shifts(df: pd.DataFrame, target_col: str, shifts: int) -> None:
    """auxiliary function to implement lag for the target variable."""
    if shifts == 0:
        pass
    else:
        df[f'{target_col}_{shifts}'] = df[f'{target_col}'].shift(shifts)
        add_shifts(df, target_col, shifts - 1)


def group_by_borough_and_hour(filtered_df: pd.DataFrame, shifts_to_add: int) -> pd.DataFrame:
    acc_df = pd.DataFrame()
    # group by borough
    for (deleg, dataset) in filtered_df.groupby('delegacion_inicio'):
        # then group by hour and count number of registered incidents
        group_hour = dataset.resample('H', on='creacion_timestamp')['incidente'].count()
        group_hour_df = pd.DataFrame(group_hour)

        group_hour_df['delegacion'] = deleg
        group_hour_df['timestamp'] = group_hour.index

        # add lag variables
        add_shifts(group_hour_df, 'incidente', shifts_to_add)
        group_hour_df.reset_index()

        # append to accumulated result
        acc_df = pd.concat([acc_df, group_hour_df])

    acc_df['day'] = acc_df['timestamp'].dt.day
    acc_df['month'] = acc_df['timestamp'].dt.month
    acc_df['day_of_week'] = acc_df['timestamp'].dt.day_of_week
    acc_df['hour'] = acc_df['timestamp'].dt.hour

    return acc_df


raw_df = get_all_raw_data(path_to_raw_data)
filter_data = preprocess_raw_data(raw_df)
final_data = group_by_borough_and_hour(filter_data, 5)

processed_file_path = os.path.join(path_to_processed, 'incidentes_viales_dataset.csv')
final_data.to_csv(processed_file_path)