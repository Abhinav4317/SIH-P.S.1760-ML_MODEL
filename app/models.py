# app/models.py
from pydantic import BaseModel

class FarmerDetails(BaseModel):
    income: int
    land_size: float
    rainfall: float
    insurance_history: str 

class PromotionPlan(BaseModel):
    scheme_name: str
    plan: str


class PredictionRequest(BaseModel):
    post_office_name: str
    top_n_schemes: int = 3
    include_neighbor_vote: bool = False


class PlanRequest(BaseModel):
    post_office_name: str
    top_n_schemes: int = 3
    include_neighbor_vote: bool = True


class TrendsRequest(BaseModel):
    district_name: str

class IndividualRequest(BaseModel):
    Name: str
    Age_Group: str
    Gender: str
    Income_Level: int
    Occupation: str
    is_insurance: bool = False