import pandas as pd
from typing import List, Union
from pydantic import BaseModel


class PropertyInput(BaseModel):
    habitableSurface: float
    bedroomCount: int
    buildingCondition: int
    hasGarden: int
    gardenSurface: float
    hasTerrace: int
    epcScore: float
    hasParking: int
    postCode: int
    type: str
    province: str
    subtype: str
    region: str
    # Convert input to DataFrame for easier handling

# Function converts the input into a dataframe that can be read by the pkl file
    # Union allows the code to read one single property or a list of properties

def preprocess_input(data: Union[PropertyInput, List[PropertyInput]]) -> pd.DataFrame:
    numeric_features = ['habitableSurface', 'bedroomCount', 'buildingCondition',
                        'hasGarden', 'gardenSurface', 'hasTerrace', 'epcScore', 'hasParking']
    categorical_features = ['postCode', 'type', 'province', 'subtype', 'region']
    expected_columns = numeric_features + categorical_features
    
    # If it is a list of properties then it goes into a loop reading all the properties data

    if isinstance(data, list):
        records = [item.dict() for item in data]
    else:       # Else means there is info only for 1 property
        records = [data.dict()]

    input_df = pd.DataFrame(records)

    # Code raises error if anything is missing (i.e. every field is mandatory). I did not set rules around this for simplification uroses
    missing_cols = set(expected_columns) - set(input_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in input data: {missing_cols}")
    
    # reorders and selects the columns of the DataFrame input_df to match the exact order and set of columns defined in expected_columns.
    # this line ensures that: Only the expected columns are kept AND the order of columns is exactly as expected by the model
    input_df = input_df[expected_columns]

    for col in categorical_features:
        input_df[col] = input_df[col].astype('category')

    return input_df
  
 

