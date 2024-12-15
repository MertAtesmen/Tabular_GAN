from pydantic import BaseModel
from pydantic import Field, field_validator
from typing_extensions import Annotated

class SynthesizerHyperParameters(BaseModel):
    generator_dim:          Annotated[list[int], Field(default=[256, 256])]
    discriminator_dim:      Annotated[list[int], Field(default=[256, 256])]
    generator_lr:           Annotated[float, Field(default=0.0002, gt=0)]
    generator_decay:        Annotated[float, Field(default=0.000001, ge=0)]
    discriminator_lr:       Annotated[float, Field(default=0.0002, gt=0)]
    discriminator_decay:    Annotated[float, Field(default=0.000001, ge=0)]
    batch_size:             Annotated[int, Field(default=500, gt=0)]
    discriminator_steps:    Annotated[int, Field(default=1, ge=1)]
    log_frequency:          Annotated[bool, Field(default=True)]
    epochs:                 Annotated[int, Field(default=300, ge=1)]
    pac:                    Annotated[int, Field(default=10, ge=1)]

    @field_validator('generator_dim', 'discriminator_dim')
    def validate_tuples(cls, value):
        for elem in value:
            if elem < 1:
                raise ValueError('List elements must be bigger than zero')
        return value