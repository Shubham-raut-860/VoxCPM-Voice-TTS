import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.audio_utils import convert_to_pcm16_wav
from app.config import Settings, get_settings
from app.database import get_db
from app.exceptions import InvalidAudioError, LoraError, VoiceNotFoundError
from app.models import Voice
from app.schemas import LoraAttachResponse, VoiceOut

router = APIRouter(prefix="/voices", tags=["voices"])

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
LORA_SENTINEL_FILES = {"lora_config.json", "lora_weights.safetensors", "adapter_config.json"}


def _find_lora_root(extract_dir: Path) -> Path:
    """Walk extracted zip tree and return the directory that contains LoRA weights."""
    for candidate in sorted(extract_dir.rglob("*")):
        if candidate.is_dir() and any((candidate / sentinel).exists() for sentinel in LORA_SENTINEL_FILES):
            return candidate
    # Fallback: return extract_dir itself if sentinels are at root
    if any((extract_dir / s).exists() for s in LORA_SENTINEL_FILES):
        return extract_dir
    raise LoraError(
        f"No LoRA checkpoint found in zip. Expected one of {sorted(LORA_SENTINEL_FILES)} inside."
    )


@router.post("", response_model=VoiceOut, status_code=201)
async def create_voice(
    name: str = Form(...),
    owner: str | None = Form(None),
    transcript: str | None = Form(None),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Voice:
    ext = Path(audio_file.filename or "").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {sorted(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(audio_file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        voice = Voice(name=name, owner=owner, transcript=transcript, file_path="", sample_rate=0, duration_seconds=0)
        db.add(voice)
        db.flush()

        target_path = settings.voices_dir / f"{voice.id}.wav"
        sample_rate, duration = convert_to_pcm16_wav(tmp_path, target_path, settings.sample_rate)

        voice.file_path = str(target_path)
        voice.sample_rate = sample_rate
        voice.duration_seconds = duration
        db.commit()
        db.refresh(voice)
        return voice
    except Exception:
        db.rollback()
        raise
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/{voice_id}/lora", response_model=LoraAttachResponse)
async def upload_lora(
    voice_id: str,
    lora_zip: UploadFile = File(..., description="Zip of the LoRA checkpoint directory"),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> LoraAttachResponse:
    """Upload a zipped LoRA checkpoint and attach it to the voice profile.

    The zip can have any top-level structure; we walk it to find the directory
    containing lora_config.json / lora_weights.safetensors.
    """
    voice = db.query(Voice).filter(Voice.id == voice_id).first()
    if voice is None:
        raise VoiceNotFoundError(f"Voice '{voice_id}' not found")

    if not (lora_zip.filename or "").endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip archive")

    lora_base_dir = settings.loras_dir / voice_id
    if lora_base_dir.exists():
        shutil.rmtree(lora_base_dir)
    lora_base_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        shutil.copyfileobj(lora_zip.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(lora_base_dir)

        lora_root = _find_lora_root(lora_base_dir)
        voice.lora_path = str(lora_root)
        db.commit()
        return LoraAttachResponse(
            voice_id=voice_id,
            lora_path=str(lora_root),
            message="LoRA adapter attached successfully",
        )
    except LoraError:
        shutil.rmtree(lora_base_dir, ignore_errors=True)
        raise
    except zipfile.BadZipFile as exc:
        shutil.rmtree(lora_base_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Invalid zip file: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@router.delete("/{voice_id}/lora", status_code=204)
async def detach_lora(voice_id: str, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> None:
    """Detach and delete the LoRA adapter for a voice (reverts to zero-shot cloning)."""
    voice = db.query(Voice).filter(Voice.id == voice_id).first()
    if voice is None:
        raise VoiceNotFoundError(f"Voice '{voice_id}' not found")

    lora_base_dir = settings.loras_dir / voice_id
    shutil.rmtree(lora_base_dir, ignore_errors=True)
    voice.lora_path = None
    db.commit()


@router.get("", response_model=list[VoiceOut])
async def list_voices(db: Session = Depends(get_db)) -> list[Voice]:
    return db.query(Voice).order_by(Voice.created_at.desc()).all()


@router.get("/{voice_id}", response_model=VoiceOut)
async def get_voice(voice_id: str, db: Session = Depends(get_db)) -> Voice:
    voice = db.query(Voice).filter(Voice.id == voice_id).first()
    if voice is None:
        raise VoiceNotFoundError(f"Voice '{voice_id}' not found")
    return voice


@router.delete("/{voice_id}", status_code=204)
async def delete_voice(voice_id: str, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> None:
    voice = db.query(Voice).filter(Voice.id == voice_id).first()
    if voice is None:
        raise VoiceNotFoundError(f"Voice '{voice_id}' not found")

    file_path = Path(voice.file_path)
    lora_base_dir = settings.loras_dir / voice_id
    db.delete(voice)
    db.commit()
    file_path.unlink(missing_ok=True)
    shutil.rmtree(lora_base_dir, ignore_errors=True)
