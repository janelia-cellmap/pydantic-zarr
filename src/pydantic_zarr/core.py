from pydantic import BaseModel, ConfigDict


class StrictBase(BaseModel):
    model_config = ConfigDict(frozen=True)
