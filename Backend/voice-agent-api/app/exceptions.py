class VoiceAgentError(Exception):
    code = "internal_error"
    status_code = 500

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class VoiceNotFoundError(VoiceAgentError):
    code = "voice_not_found"
    status_code = 404


class GenerationNotFoundError(VoiceAgentError):
    code = "generation_not_found"
    status_code = 404


class InvalidAudioError(VoiceAgentError):
    code = "invalid_audio"
    status_code = 400


class ModelInferenceError(VoiceAgentError):
    code = "model_inference_failed"
    status_code = 502


class TextTooLongError(VoiceAgentError):
    code = "text_too_long"
    status_code = 400


class UnauthorizedError(VoiceAgentError):
    code = "unauthorized"
    status_code = 401


class LoraError(VoiceAgentError):
    code = "lora_error"
    status_code = 400
