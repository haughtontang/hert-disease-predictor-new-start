from predictor_src.data_loading import load_raw_data, scale_data, return_train_test_split
from predictor_src.predict_missing_values import predict_missing_values
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    features_to_scale = ["age", "trestbps", "chol", "thalach"]
    features_to_fill_in = ["ca", "thal"]
    # Load and transform the data
    raw_df, labels_df = load_raw_data()
    # I tried to replace nan values with the median and it made the performance worse overall
    # TODO - make another model that predicts ca and thal based on the other features as a method to replace them
    # Remove rows where there are any nan values - I may want to do this more inteligently but I will remove it for now just to get it done
    for feature in features_to_fill_in:
        filled_values = predict_missing_values(raw_data_df=raw_df, labels_df=labels_df,
                                               features_to_scale=features_to_scale, feature=feature)
        raw_df[feature] = filled_values
    raw_df_no_nan = raw_df.dropna(how="any")
    labels_df.drop(raw_df.index.difference(raw_df_no_nan.index), inplace=True)
    scaled_df = scale_data(data_df=raw_df_no_nan, features_to_scale=features_to_scale)
    x_train, x_test, y_train, y_test = return_train_test_split(data_df=scaled_df, label_data_df=labels_df)

    # Create the model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    accuracy = accuracy_score(y_train, y_pred)
    print("train accuracy", accuracy * 100)
    test_prediction = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_prediction)
    print("test accuracy", test_accuracy * 100)


if __name__ == '__main__':
    main()
