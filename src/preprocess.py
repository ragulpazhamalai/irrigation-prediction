import pandas as pd

def preprocess_data(df):
    
    # Binary encoding
    df['Mulching_Used'] = df['Mulching_Used'].map({'No': 0, 'Yes': 1})
    
    # Ordinal encoding
    stage_map = {
        'Seedling': 0,
        'Vegetative': 1,
        'Flowering': 2,
        'Maturity': 3
    }
    df['Crop_Growth_Stage'] = df['Crop_Growth_Stage'].map(stage_map)

    # One-hot encoding
    df = pd.get_dummies(df, columns=[
        'Soil_Type','Crop_Type','Season',
        'Irrigation_Type','Water_Source','Region'
    ])

    return df
