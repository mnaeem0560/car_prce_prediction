from xgboost import XGBRegressor

def predict(input_df, loaded_model):
    model = loaded_model["model"]
    pred = model.predict(input_df)
    pred = float(pred[0])
    return pred
