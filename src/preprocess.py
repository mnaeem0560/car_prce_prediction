from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

def preprocess_data(input_df, load_model):
    model = load_model["model"]
    oh_tranformer = load_model["onehot_encoder"]
    scaler = load_model["scaler"]

    cat_feature = input_df.select_dtypes(include='object').columns

    classes = oh_tranformer.get_feature_names_out(cat_feature)
    df_onehot = pd.DataFrame(oh_tranformer.transform(input_df[cat_feature]).toarray(), columns=classes)
    input_df = pd.concat([input_df.drop(columns=cat_feature), df_onehot], axis=1)

    num_feature = ['Year', 'Engine HP', 'Engine Cylinders', 'Number of Doors', 'highway MPG', 'Fuel_Efficiency']
    input_df[num_feature] = scaler.transform(input_df[num_feature])

    return input_df