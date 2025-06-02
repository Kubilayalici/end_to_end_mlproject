import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
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