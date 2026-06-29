from pathlib import Path

import soundfile as sf
from pydub import AudioSegment

from app.exceptions import InvalidAudioError


def convert_to_pcm16_wav(source_path: Path, target_path: Path, target_sample_rate: int) -> tuple[int, float]:
    """Convert any ffmpeg-readable audio to 16kHz mono PCM16 wav.

    Returns (sample_rate, duration_seconds).
    """
    try:
        audio = AudioSegment.from_file(source_path)
    except Exception as exc:
        raise InvalidAudioError(f"Could not decode audio: {exc}") from exc

    audio = audio.set_channels(1).set_frame_rate(target_sample_rate).set_sample_width(2)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(target_path, format="wav")

    with sf.SoundFile(target_path) as snd:
        sr = snd.samplerate
        duration = len(snd) / sr

    return sr, duration
