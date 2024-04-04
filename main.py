from predictor_src.data_loading import load_raw_data, scale_data, return_train_test_split
from predictor_src.predict_missing_values import predict_missing_values
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


def detect_outliers(df_out, drop=False):
    indexes_to_drop = list()
    for each_feature in df_out.columns:
        feature_data = df_out[each_feature]
        Q1 = np.percentile(feature_data, 25.)  # 25th percentile of the data of the given feature
        Q3 = np.percentile(feature_data, 75.)  # 75th percentile of the data of the given feature
        IQR = Q3 - Q1  # Interquartile Range
        outlier_step = IQR * 1.5  # That's we were talking about above
        outliers = feature_data[~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()
        if not drop:
            print('For the feature {}, No of Outliers is {}'.format(each_feature, len(outliers)))
        if drop:
            indexes_to_drop.extend(outliers)
            # df_out.drop(outliers, inplace=True, errors='ignore')
            # print('Outliers from {} feature removed'.format(each_feature))
    return list(set(indexes_to_drop))


def main():
    features_to_scale = ["age", "trestbps", "chol", "thalach"]
    features_to_fill_in = ["ca", "thal"]
    continous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    # Load and transform the data
    raw_df, labels_df = load_raw_data()
    # I tried to replace nan values with the median and it made the performance worse overall
    # TODO - make another model that predicts ca and thal based on the other features as a method to replace them
    # Remove rows where there are any nan values - I may want to do this more inteligently but I will remove it for now just to get it done
    # for feature in features_to_fill_in:
    #     filled_values = predict_missing_values(raw_data_df=raw_df, labels_df=labels_df,
    #                                            features_to_scale=features_to_scale, feature=feature)
    #     raw_df[feature] = filled_values
    outliers_index = detect_outliers(raw_df[continous_features], drop=True)
    raw_df.drop(outliers_index, inplace=True, errors='ignore')
    labels_df.drop(outliers_index, inplace=True, errors='ignore')
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
