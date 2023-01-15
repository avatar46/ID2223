import hopsworks
import joblib




def model(daily_sentiment, stock_df):
    daily_sentiment = daily_sentiment.rename(columns={'publish_date': 'date', 'ticker': 'name'})
    daily_sentiment['name'] = daily_sentiment['name'].str.upper()
    stock_df['date'] = stock_df['date'].apply(lambda x : x.date())

    X = daily_sentiment.merge(stock_df)
    X = X.drop(['date', 'name'], axis=1)

    project = hopsworks.login()

    mr = project.get_model_registry()
    model = mr.get_model("random_forest_classifier", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/model.pkl")
    arr = model.predict(X)
    return arr[0]