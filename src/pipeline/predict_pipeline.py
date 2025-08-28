import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    @staticmethod
    def _ensure_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features used during training if they are missing.
        This mirrors the feature engineering done in notebooks before saving cleaned data.
        """
        d = df.copy()
        # Binary flags
        if 'has_failures' not in d.columns and 'failures' in d.columns:
            d['has_failures'] = (d['failures'] > 0).astype(int)
        if 'high_studytime' not in d.columns and 'studytime' in d.columns:
            d['high_studytime'] = (d['studytime'] >= 3).astype(int)
        if 'avg_parent_edu' not in d.columns and {'Medu','Fedu'}.issubset(d.columns):
            d['avg_parent_edu'] = (d['Medu'] + d['Fedu']) / 2.0
        if 'early_failure' not in d.columns and 'G1' in d.columns:
            d['early_failure'] = (d['G1'] < 5).astype(int)
        if 'parents_together' not in d.columns and 'Pstatus' in d.columns:
            d['parents_together'] = (d['Pstatus'] == 'T').astype(int)
        if 'big_family' not in d.columns and 'famsize' in d.columns:
            d['big_family'] = (d['famsize'] == 'GT3').astype(int)
        if 'urban_student' not in d.columns and 'address' in d.columns:
            d['urban_student'] = (d['address'] == 'U').astype(int)
        if 'long_travel' not in d.columns and 'traveltime' in d.columns:
            d['long_travel'] = (d['traveltime'] >= 3).astype(int)
        if 'has_internet' not in d.columns and 'internet' in d.columns:
            d['has_internet'] = (d['internet'] == 'yes').astype(int)
        if 'romantic_rel' not in d.columns and 'romantic' in d.columns:
            d['romantic_rel'] = (d['romantic'] == 'yes').astype(int)
        if 'high_alcohol_use' not in d.columns and {'Dalc','Walc'}.issubset(d.columns):
            d['high_alcohol_use'] = ((d['Dalc'] + d['Walc']) >= 6).astype(int)
        if 'high_absenteeism' not in d.columns and 'absences' in d.columns:
            d['high_absenteeism'] = (d['absences'] >= 10).astype(int)
        return d

    def predict(self, features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            # Add engineered features expected by the preprocessor/model
            features_fe = self._ensure_engineered_features(features)
            data_scaled = preprocessor.transform(features_fe)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e,sys)
        

        
class CustomData:
    def __init__(self,
                 school: str,
                 sex: str,
                 age: int,
                 address: str,
                 famsize: str,
                 Pstatus: str,
                 Medu: int,
                 Fedu: int,
                 Mjob: str,
                 Fjob: str,
                 reason: str,
                 guardian: str,
                 traveltime: int,
                 studytime: int,
                 failures: int,
                 schoolsup: str,
                 famsup: str,
                 paid: str,
                 activities: str,
                 nursery: str,
                 higher: str,
                 internet: str,
                 romantic: str,
                 famrel: int,
                 freetime: int,
                 goout: int,
                 Dalc: int,
                 Walc: int,
                 health: int,
                 absences: int,
                 G1: int,
                 G2: int
                 ):
        self.school = school
        self.sex = sex
        self.age = age
        self.address = address
        self.famsize = famsize
        self.Pstatus = Pstatus
        self.Medu = Medu
        self.Fedu = Fedu
        self.Mjob = Mjob
        self.Fjob = Fjob
        self.reason = reason
        self.guardian = guardian
        self.traveltime = traveltime
        self.studytime = studytime
        self.failures = failures
        self.schoolsup = schoolsup
        self.famsup = famsup
        self.paid = paid
        self.activities = activities
        self.nursery = nursery
        self.higher = higher
        self.internet = internet
        self.romantic = romantic
        self.famrel = famrel
        self.freetime = freetime
        self.goout = goout
        self.Dalc = Dalc
        self.Walc = Walc
        self.health = health
        self.absences = absences
        self.G1 = G1
        self.G2 = G2
    def to_dataframe(self):
        try:
            data_dict = {
                "school": [self.school],
                "sex": [self.sex],
                "age": [self.age],
                "address": [self.address],
                "famsize": [self.famsize],
                "Pstatus": [self.Pstatus],
                "Medu": [self.Medu],
                "Fedu": [self.Fedu],
                "Mjob": [self.Mjob],
                "Fjob": [self.Fjob],
                "reason": [self.reason],
                "guardian": [self.guardian],
                "traveltime": [self.traveltime],
                "studytime": [self.studytime],
                "failures": [self.failures],
                "schoolsup": [self.schoolsup],
                "famsup": [self.famsup],
                "paid": [self.paid],
                "activities": [self.activities],
                "nursery": [self.nursery],
                "higher": [self.higher],
                "internet": [self.internet],
                "romantic": [self.romantic],
                "famrel": [self.famrel],
                "freetime": [self.freetime],
                "goout": [self.goout],
                "Dalc": [self.Dalc],
                "Walc": [self.Walc],
                "health": [self.health],
                "absences": [self.absences],
                "G1": [self.G1],
                "G2": [self.G2]
                }
            return pd.DataFrame(data_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
