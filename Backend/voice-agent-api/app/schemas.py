from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class VoiceOut(BaseModel):
    id: str
    name: str
    owner: str | None
    transcript: str | None
    sample_rate: int
    duration_seconds: float
    lora_path: str | None
    has_lora: bool
    created_at: datetime

    class Config:
        from_attributes = True


class LoraAttachResponse(BaseModel):
    voice_id: str
    lora_path: str
    message: str


class GenerationRequest(BaseModel):
    voice_id: str
    text: str = Field(min_length=1)
    cfg_value: float = Field(default=2.0, gt=0, le=10)
    inference_timesteps: int = Field(default=10, ge=1, le=100)
    denoise: bool = True
    normalize: bool = True

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be empty")
        return stripped


class GenerationOut(BaseModel):
    id: str
    voice_id: str
    text: str
    cfg_value: float
    inference_timesteps: int
    duration_seconds: float
    used_lora: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ErrorResponse(BaseModel):
    error: str
    code: str
