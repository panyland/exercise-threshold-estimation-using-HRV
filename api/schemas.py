from typing import Optional, Literal
from pydantic import BaseModel

model_type = Literal["rf", "xgboost"]

class ThresholdEstimate(BaseModel):
    vt1_power_W: float | None
    vt1_hr_bpm: float | None
    vt2_power_W: float | None
    vt2_hr_bpm: float | None