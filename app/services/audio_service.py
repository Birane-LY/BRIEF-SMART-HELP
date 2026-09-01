import subprocess
import tempfile
from pathlib import Path

from app.models.asr_model import get_asr_model


class ASRService:
    def __init__(self):
        self.model = get_asr_model()

    async def transcribe(self, audio_bytes: bytes) -> str | None:
        """Convertit l’audio reçu en WAV puis le transmet à faster-whisper."""
        if not audio_bytes:
            return None

        try:
            with tempfile.TemporaryDirectory(prefix="smart-help-audio-") as temp_dir:
                temp_path = Path(temp_dir)
                input_path = temp_path / "input.webm"
                wav_path = temp_path / "audio.wav"
                input_path.write_bytes(audio_bytes)

                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(input_path),
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        str(wav_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=60,
                )

                segments, _ = self.model.transcribe(
                    str(wav_path),
                    beam_size=5,
                    language=None,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
                return text or None

        except FileNotFoundError:
            print("[ERREUR AUDIO] FFmpeg n'est pas installé sur le serveur.")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            print(f"[ERREUR AUDIO] Conversion WebM vers WAV impossible : {error}")
        except Exception as error:
            print(f"[ERREUR AUDIO] Transcription impossible : {error}")

        return None


asr_service = ASRService()
