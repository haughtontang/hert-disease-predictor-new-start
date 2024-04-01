import pandas as pd
import numpy as np
from predictor_src.data_loading import scale_data, return_train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def predict_missing_values(raw_data_df: pd.DataFrame, labels_df: pd.DataFrame, feature: str, features_to_scale: list) -> np.ndarray:
    tog_df = raw_data_df.copy(deep=True)
    tog_df["output"] = labels_df["num"]
    tog_df_scaled = scale_data(data_df=tog_df, features_to_scale=features_to_scale)
    tog_df_scaled_no_nan = tog_df_scaled.dropna(how="any")
    target = tog_df_scaled_no_nan[feature]
    tog_df_scaled_no_target = tog_df_scaled_no_nan[[i for i in tog_df_scaled_no_nan.columns.to_list() if i != feature]]
    x_train, x_test, y_train, y_test = return_train_test_split(data_df=tog_df_scaled_no_target, label_data_df=target)

    # Create the model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    accuracy = accuracy_score(y_train, y_pred)
    print(feature, "train accuracy", accuracy * 100)
    test_prediction = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_prediction)
    print(feature, "test accuracy", test_accuracy * 100)

    new_feature = np.zeros_like(raw_data_df[feature])
    for idx in range(len(new_feature)):
        col_value_for_row = tog_df_scaled.loc[idx, feature]
        if np.isnan(col_value_for_row):
            row_array = np.array(tog_df_scaled.iloc[idx])
            row_array = row_array[~np.isnan(row_array)]
            col_value_for_row = model.predict(row_array.reshape(1, -1))
        new_feature[idx] = int(col_value_for_row)
    return new_feature
