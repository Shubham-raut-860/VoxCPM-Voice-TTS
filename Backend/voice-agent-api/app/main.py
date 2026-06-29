import logging

from fastapi import Depends, FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.database import init_db
from app.exceptions import UnauthorizedError, VoiceAgentError
from app.logging_config import setup_logging
from app.routers import tts, voices
from app.voice_engine import VoiceEngine

setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(title=settings.app_name, version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(VoiceAgentError)
async def voice_agent_error_handler(request: Request, exc: VoiceAgentError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message, "code": exc.code})


async def _verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise UnauthorizedError("Invalid or missing X-API-Key header")


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Starting %s", settings.app_name)
    init_db()
    engine = VoiceEngine(settings)
    engine.load()
    app.state.voice_engine = engine
    logger.info("Ready")


@app.get("/health")
async def health(request: Request) -> dict:
    engine: VoiceEngine | None = getattr(request.app.state, "voice_engine", None)
    return {"status": "ok", "model_loaded": bool(engine and engine.is_loaded)}


app.include_router(voices.router, dependencies=[Depends(_verify_api_key)])
app.include_router(tts.router, dependencies=[Depends(_verify_api_key)])
