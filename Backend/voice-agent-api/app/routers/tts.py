from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.exceptions import GenerationNotFoundError, TextTooLongError, VoiceNotFoundError
from app.models import Generation, Voice
from app.schemas import GenerationOut, GenerationRequest
from app.voice_engine import VoiceEngine

router = APIRouter(prefix="/tts", tags=["tts"])


@router.post("", response_model=GenerationOut, status_code=201)
async def generate_speech(
    payload: GenerationRequest,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Generation:
    if len(payload.text) > settings.max_text_length:
        raise TextTooLongError(f"text exceeds {settings.max_text_length} characters")

    voice: Voice | None = db.query(Voice).filter(Voice.id == payload.voice_id).first()
    if voice is None:
        raise VoiceNotFoundError(f"Voice '{payload.voice_id}' not found")

    engine: VoiceEngine = request.app.state.voice_engine

    gen = Generation(
        voice_id=voice.id,
        text=payload.text,
        file_path="",
        cfg_value=payload.cfg_value,
        inference_timesteps=payload.inference_timesteps,
        duration_seconds=0.0,
        used_lora=voice.lora_path is not None,
    )
    db.add(gen)
    db.flush()

    output_path = settings.outputs_dir / f"{gen.id}.wav"

    duration = await engine.synthesize(
        text=payload.text,
        output_path=output_path,
        lora_path=voice.lora_path,
        prompt_wav_path=Path(voice.file_path) if voice.file_path else None,
        prompt_text=voice.transcript,
        cfg_value=payload.cfg_value,
        inference_timesteps=payload.inference_timesteps,
        normalize=payload.normalize,
        denoise=payload.denoise,
    )

    gen.file_path = str(output_path)
    gen.duration_seconds = duration
    db.commit()
    db.refresh(gen)
    return gen


@router.get("", response_model=list[GenerationOut])
async def list_generations(
    voice_id: str | None = None,
    db: Session = Depends(get_db),
) -> list[Generation]:
    q = db.query(Generation)
    if voice_id:
        q = q.filter(Generation.voice_id == voice_id)
    return q.order_by(Generation.created_at.desc()).all()


@router.get("/{generation_id}", response_model=GenerationOut)
async def get_generation(generation_id: str, db: Session = Depends(get_db)) -> Generation:
    gen = db.query(Generation).filter(Generation.id == generation_id).first()
    if gen is None:
        raise GenerationNotFoundError(f"Generation '{generation_id}' not found")
    return gen


@router.get("/{generation_id}/audio")
async def download_audio(generation_id: str, db: Session = Depends(get_db)) -> FileResponse:
    gen = db.query(Generation).filter(Generation.id == generation_id).first()
    if gen is None:
        raise GenerationNotFoundError(f"Generation '{generation_id}' not found")
    return FileResponse(gen.file_path, media_type="audio/wav", filename=f"{generation_id}.wav")
