import pandas as pd


class Preprocessor:
    def __init__(self):
        """Initialize the preprocessor with required fields and outlier thresholds."""
        self.required_fields = ['size', 'bedrooms', 'location']
        self.iqr_multiplier = 1.5  # Threshold for outlier detection

    def preprocess(self, df):
        """
        Preprocess the input imoweb-dataset.csv for price prediction, including outlier removal.
        
        Args:
            house_data (dict): Dictionary containing house features (e.g., size, bedrooms, location, etc.)
        
        Returns:
            dict: Preprocessed data ready for the model, or an error message if data is invalid.
        """
        # Convert input to DataFrame for easier handling
        df = pd.DataFrame(['immoweb-dataset.csv'])
        
        # Check for missing required fields
        missing_fields = [field for field in self.required_fields if field not in 'immoweb-dataset.csv' or pd.isna('immoweb-dataset.csv'[field])]
        if missing_fields:
            return f"Error: Missing required fields: {', '.join(missing_fields)}"
        
        # Fill NA values with defaults
        df['size'] = df['size'].fillna(df['size'].median())
        df['bedrooms'] = df['bedrooms'].fillna(0)  # Assuming 0 bedrooms is a valid default
        df['location'] = df['location'].fillna('unknown')

 # Normalize size
        df['size'] = (df['size'] - df['size'].min()) / (df['size'].max() - df['size'].min())
        
        # Encode location
        location_mapping = {'unknown': 0, 'urban': 1, 'suburban': 2, 'rural': 3}
        df['location'] = df['location'].map(location_mapping).fillna(0)
        
        # Convert back to dictionary
        preprocessed_data = df.to_dict(orient='records')[0]

# Remove outliers using IQR method for 'size' (numerical feature)
        Q1 = df['size'].quantile(0.25)
        Q3 = df['size'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (self.iqr_multiplier * IQR)
        upper_bound = Q3 + (self.iqr_multiplier * IQR)
        df = df[(df['size'] >= lower_bound) & (df['size'] <= upper_bound)]
        
        # If no data remains after outlier removal, return an error
        if df.empty:
            return "Error: All data points are outliers and removed."
        
        # convert features epcScore and buildingConstruction to numerical
        epc_score = {"A++": 9,
                    "A+": 8,
                    "A": 7,
                    "B": 6,
                    "C": 5,
                    "D": 4,
                    "E": 3,
                    "F": 2,
                    "G": 1}
        df["epc_score"] = df["epcScore"].map(epc_score)

        building_condition = {"AS_NEW": 6,
                            "GOOD": 5,
                            "JUST_RENOVATED": 4,
                            "TO_BE_DONE_UP": 3,
                            "TO_RENOVATE": 2,
                            "TO_RESTORE": 1}
        df["building_condition"] = df["buildingCondition"].map(building_condition)

        df = df.drop["epcScore", "buildingCondition"] # remove old features

        return self.df
        
