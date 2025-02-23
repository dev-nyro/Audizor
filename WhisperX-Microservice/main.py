"""
# main.py
WhisperX Transcription Microservice
----------------------------------

A microservice that handles audio transcription using WhisperX, integrated with:
- Google Cloud Pub/Sub for message queuing
- Google Cloud Storage for file storage
- Supabase for state management and results storage

Environment Variables Required:
- SUPABASE_URL: URL of your Supabase instance
- SUPABASE_KEY: API key for Supabase
- GCP_PROJECT: Google Cloud project ID
- HF_TOKEN: HuggingFace token for diarization
- GOOGLE_APPLICATION_CREDENTIALS: Path to service account key file

Tables Required in Supabase:
1. transcription_jobs:
   - id (UUID, primary key)
   - id_usuario (UUID)
   - ruta_audio (TEXT)
   - estado (TEXT)
   - intentos (INT4)
   - fecha_creacion (TIMESTAMPTZ)
   - fecha_actualizacion (TIMESTAMPTZ)
   - mensaje_error (TEXT)
   - nivel (TEXT)

2. transcriptions:
   - id (UUID, primary key)
   - archivo_id (UUID, foreign key to transcription_jobs)
   - texto_transcripcion (TEXT)
   - fecha_generacion (TIMESTAMPTZ)
   - estado (TEXT)
   - idioma (TEXT)
"""
# Standard library imports
import os
import json
import gc
import base64
import logging
import asyncio
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional
from enum import Enum
from uuid import uuid4

# Third-party imports
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, validator
    from google.cloud import storage
    from google.api_core import retry
    from supabase import create_client, Client
    import torch
    import threading
    import tempfile
    import whisperx
    import uvicorn
    from starlette.middleware.timeout import TimeoutMiddleware
    from dotenv import load_dotenv
    import psycopg
    from psycopg_pool import AsyncConnectionPool
    from google.cloud import pubsub_v1
    from concurrent.futures import TimeoutError
except ImportError as e:
    raise ImportError(f"Required package not installed: {str(e)}")

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TierType(str, Enum):
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"

@lru_cache()
def get_env_config():
    """Get and validate all required environment variables"""
    required_vars = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_KEY": os.getenv("SUPABASE_KEY"),
        "GCP_PROJECT": os.getenv("GCP_PROJECT"),
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "GOOGLE_APPLICATION_CREDENTIALS": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "DATABASE_URL": os.getenv("DATABASE_URL")
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    return required_vars

# Initialize FastAPI
app = FastAPI(
    title="WhisperX Transcription Service",
    description="Audio transcription service using WhisperX",
    version="1.0.0",
    on_startup=[get_env_config]
)
app.add_middleware(TimeoutMiddleware, timeout=600)

# Initialize Supabase
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Initialize PostgreSQL connection pool
db_pool = AsyncConnectionPool(os.getenv("DATABASE_URL"), min_size=2, max_size=10)

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TranscriptionJob(BaseModel):
    """Model for transcription jobs matching Supabase schema"""
    id: str
    id_usuario: str
    ruta_audio: str
    estado: JobStatus
    intentos: int = 0
    nivel: Optional[TierType] = None
    fecha_creacion: Optional[datetime] = None
    fecha_actualizacion: Optional[datetime] = None
    mensaje_error: Optional[str] = None

    @validator('intentos')
    def validate_intentos(cls, v):
        if v < 0:
            raise ValueError("Intentos no puede ser negativo")
        if v > 3:
            raise ValueError("Intentos no puede ser mayor a 3")
        return v

    @validator('ruta_audio')
    def validate_ruta_audio(cls, v):
        if not v.startswith('gs://'):
            raise ValueError("ruta_audio debe comenzar con gs://")
        return v

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PubSubMessage(BaseModel):
    message: dict
    subscription: str

# GCS helpers
def parse_gcs_url(gcs_url: str) -> tuple[str, str]:
    if not gcs_url.startswith("gs://"):
        raise ValueError("La URL debe iniciar con gs://")
    parts = gcs_url[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""

async def download_from_gcs(gcs_url: str, local_path: str) -> None:
    """Download file from GCS asynchronously"""
    bucket_name, blob_name = parse_gcs_url(gcs_url)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Implement retry logic
    @retry.Retry(predicate=retry.if_exception_type(Exception))
    def download_with_retry():
        blob.download_to_filename(local_path)
        
    await asyncio.to_thread(download_with_retry)
    logger.info(f"Downloaded {gcs_url} to {local_path}")

async def transcribe_audio(audio_path: str, tier: str) -> dict:
    """Transcribe audio using WhisperX based on tier"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Add timeout to prevent long-running transcriptions
        async with asyncio.timeout(300):  # 5 minutes
            if tier == "tier1":
                model = whisperx.load_model("small", device, compute_type="int8")
                audio = whisperx.load_audio(audio_path)
                result = model.transcribe(audio, batch_size=16)
                return result
                
            elif tier == "tier2":
                model = whisperx.load_model("large-v2", device, compute_type="float16")
                audio = whisperx.load_audio(audio_path)
                result = model.transcribe(audio, batch_size=8)
                align_model, metadata = whisperx.load_align_model(
                    language_code=result["language"], 
                    device=device
                )
                result = whisperx.align(
                    result["segments"], 
                    align_model, 
                    metadata, 
                    audio, 
                    device
                )
                return result
                
            elif tier == "tier3":
                hf_token = os.getenv("HF_TOKEN")
                if not hf_token:
                    raise ValueError("HF_TOKEN required for tier3")
                    
                model = whisperx.load_model("large-v3", device, compute_type="float16")
                audio = whisperx.load_audio(audio_path)
                result = model.transcribe(audio, batch_size=4)
                
                align_model, metadata = whisperx.load_align_model(
                    language_code=result["language"],
                    device=device
                )
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio,
                    device
                )
                
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=device
                )
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                return result
                
            else:
                raise ValueError(f"Invalid tier: {tier}")
                
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise

async def upload_result_to_gcs(result: dict, user_id: str, job_id: str) -> str:
    try:
        # Subir resultado completo a GCS
        client = storage.Client()
        bucket = client.bucket("whisperx-results")
        blob_name = f"WX-results-transcript-{user_id}-{job_id}.json"
        blob = bucket.blob(blob_name)
        result_json = json.dumps(result, ensure_ascii=False)
        blob.upload_from_string(result_json, content_type="application/json")
        result_url = f"gs://whisperx-results/{blob_name}"
        logger.info(f"Uploaded full result to {result_url}")

        # Publicar notificación a Pub/Sub (usar transcription-jobs como en los requisitos)
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(os.getenv("GCP_PROJECT"), "transcription-jobs")
        message_dict = {
            "user_id": user_id,
            "job_id": job_id,
            "result_url": result_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "language": result.get("language", "unknown"),
            "tier": result.get("tier", "unknown")
        }
        message_data = json.dumps(message_dict).encode("utf-8")
        future = publisher.publish(topic_path, data=message_data)
        message_id = future.result()
        logger.info(f"Published notification to {topic_path} with ID: {message_id}")

        return result_url
    except Exception as e:
        logger.error(f"Error uploading result to GCS: {str(e)}")
        raise
        
    except Exception as e:
        logger.error(f"Error uploading result to GCS: {str(e)}")
        raise

# --- Supabase helpers ---

# Database connection helper
async def get_db_connection():
    """Get a database connection from the pool"""
    async with db_pool.connection() as conn:
        return conn
    

async def update_transcription_records(
    job_id: str,
    status: JobStatus,
    result: dict,
    user_id: str
) -> None:
    """Actualiza las tablas transcription_jobs y transcriptions usando psycopg en una transacción asíncrona."""
    try:
        now = datetime.now(timezone.utc)
        async with db_pool.connection() as conn:
            async with conn.transaction():
                if status == JobStatus.FAILED:
                    # Actualización en caso de error
                    update_query = """
                        UPDATE transcription_jobs
                        SET estado = %s,
                            fecha_actualizacion = %s,
                            mensaje_error = %s,
                            intentos = intentos + 1
                        WHERE id = %s;
                    """
                    update_params = (
                        status.value,
                        now,
                        result.get("error", "Unknown error"),
                        job_id
                    )
                    await conn.execute(update_query, update_params)
                elif status == JobStatus.COMPLETED and result:
                    transcription_data = result.get("transcription", {})
                    segments = transcription_data.get("segments", [])
                    transcription_text = " ".join(segment.get("text", "") for segment in segments if segment.get("text"))
                    if not transcription_text:
                        raise ValueError("No transcription text provided")
                    
                    # Actualiza el estado del job
                    update_query = """
                        UPDATE transcription_jobs
                        SET estado = %s,
                            fecha_actualizacion = %s
                        WHERE id = %s;
                    """
                    update_params = (status.value, now, job_id)
                    await conn.execute(update_query, update_params)

                    # Inserta el registro en la tabla de transcriptions
                    transcription_id = str(uuid4())
                    insert_query = """
                        INSERT INTO transcriptions (id, archivo_id, texto_transcripcion, fecha_generacion, estado, idioma)
                        VALUES (%s, %s, %s, %s, %s, %s);
                    """
                    insert_params = (
                        transcription_id,
                        job_id,
                        transcription_text,
                        now,
                        status.value,
                        transcription_data.get("language", "unknown")
                    )
                    await conn.execute(insert_query, insert_params)
                else:
                    # En otros casos, se actualiza sólo el estado
                    update_query = """
                        UPDATE transcription_jobs
                        SET estado = %s,
                            fecha_actualizacion = %s
                        WHERE id = %s;
                    """
                    update_params = (status.value, now, job_id)
                    await conn.execute(update_query, update_params)
        logger.info(f"Updated job {job_id} with status {status.value}")
    except Exception as e:
        logger.error(f"Error updating transcription records: {str(e)}")
        raise


async def process_pubsub_message(message: PubSubMessage):
    """Process incoming Pub/Sub message"""
    job = None
    try:
        message_data = base64.b64decode(message.message.get("data", "")).decode("utf-8")
        job_data = json.loads(message_data)
        
        # Convert estado to JobStatus enum if needed
        if "estado" in job_data:
            job_data["estado"] = JobStatus(job_data["estado"])
        
        # Convert nivel to TierType enum if needed
        if job_data.get("nivel"):
            job_data["nivel"] = TierType(job_data["nivel"])
            
        # Convert date strings to datetime objects
        for date_field in ["fecha_creacion", "fecha_actualizacion"]:
            if job_data.get(date_field):
                job_data[date_field] = datetime.fromisoformat(job_data[date_field].replace('Z', '+00:00'))
        
        job = TranscriptionJob(**job_data)
        
        if job.intentos >= 3:
            await update_transcription_records(
                job.id,
                JobStatus.FAILED,
                {"error": "Maximum retry attempts exceeded", "attempts": job.intentos},
                job.id_usuario
            )
            return {"status": "failed", "reason": "max_retries"}

        result = await process_transcription_job(job)
        await update_transcription_records(
            job.id,
            JobStatus.COMPLETED,
            result,
            job.id_usuario
        )
        
        return {"status": "success", "job_id": job.id}
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        if job:
            current_attempts = job.intentos + 1
            error_data = {
                "error": str(e),
                "attempts": job.intentos + 1
            }

            if current_attempts >= 3:
                error_data["error"] = "Maximum retry attempts exceeded"

            await update_transcription_records(
                job.id,
                JobStatus.FAILED,
                error_data,
                job.id_usuario
            )
        raise

async def process_transcription_job(job: TranscriptionJob) -> dict:
    local_audio_path = os.path.join(tempfile.gettempdir(), f"{job.id}.mp3")
    tier = job.nivel.value if job.nivel else "tier1"
    
    try:
        await download_from_gcs(job.ruta_audio, local_audio_path)
        result = await transcribe_audio(local_audio_path, tier)
        result["tier"] = tier  # Añadir tier al resultado
        result_path = await upload_result_to_gcs(result, job.id_usuario, job.id)
        return {
            "transcription": result,
            "result_path": result_path
        }
    finally:
        if os.path.exists(local_audio_path):
            os.remove(local_audio_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Supabase
        await supabase.table("transcription_jobs").select("id").limit(1).execute()
        # Database
        async with get_db_connection() as conn:
            await conn.execute("SELECT 1")
        # GCS
        storage.Client().list_buckets(max_results=1)
        # Pub/Sub (simple chequeo de suscripción)
        subscriber = pubsub_v1.SubscriberClient()
        subscriber.get_subscription(subscription="projects/nyro-450117/subscriptions/whisperx-microservice")
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database": "connected",
            "supabase": "connected",
            "gcs": "connected",
            "pubsub": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    
# Cleanup resources
@app.on_event("shutdown")
async def cleanup_resources():
    """Cleanup resources on shutdown"""
    await db_pool.close()

async def process_and_ack(pubsub_message):
    try:
        decoded_data = base64.b64decode(pubsub_message.data).decode("utf-8")
        message_wrapper = PubSubMessage(
            message={"data": base64.b64encode(pubsub_message.data).decode("utf-8")},
            subscription=pubsub_message.subscription
        )
        await process_pubsub_message(message_wrapper)
        pubsub_message.ack()
    except Exception as e:
        logger.error(f"Error procesando mensaje Pub/Sub desde {pubsub_message.subscription}: {e}")
        pubsub_message.nack()

def pubsub_callback(message):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(process_and_ack(message))

def start_pubsub_listener():
    """Inicia el listener de Pub/Sub en un hilo separado."""
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = "projects/nyro-450117/subscriptions/whisperx-microservice"
    future = subscriber.subscribe(subscription_path, pubsub_callback)
    logger.info(f"Escuchando mensajes en {subscription_path}...")
    try:
        future.result()
    except Exception as e:
        logger.error(f"Excepción en la escucha de {subscription_path}: {e}")

@app.on_event("startup")
async def startup_pubsub_listener():
    """Inicia el listener de Pub/Sub en un hilo aparte al arrancar la aplicación."""
    thread = threading.Thread(target=start_pubsub_listener, daemon=True)
    thread.start()

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        workers=4,
        timeout_keep_alive=75,
        limit_concurrency=4,
        limit_max_requests=500
    )