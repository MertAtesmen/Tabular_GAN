from pydantic import BaseModel
from pydantic import Field, field_validator
from typing_extensions import Annotated
from src.schemas.synthesizer_hyper_parameters import SynthesizerHyperParameters

class TrialInfo(BaseModel):
    model_saved_file: Annotated[str, Field()]
    parameters: Annotated[SynthesizerHyperParameters, Field()]
    metrics: Annotated[dict, Field()]