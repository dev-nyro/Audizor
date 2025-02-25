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
    import re
    import threading
    import tempfile
    import whisperx
    import uvicorn
    from dotenv import load_dotenv
    import psycopg
    from psycopg_pool import AsyncConnectionPool
    from google.cloud import pubsub_v1
    from concurrent.futures import TimeoutError
except ImportError as e:
    raise ImportError(f"Required package not installed: {str(e)}")

# Load environment variables from .env file
load_dotenv()

pending_tasks = set()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TierType(str, Enum):
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"

@lru_cache()
def get_env_config():
    """Get and validate all required environment variables, including service account credentials"""
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
    
    # Verificar archivo de credenciales de Google Cloud
    credentials_path = required_vars["GOOGLE_APPLICATION_CREDENTIALS"]
    if not os.path.isfile(credentials_path):
        raise ValueError(f"Service account credentials file does not exist: {credentials_path}")
    
    try:
        with open(credentials_path, 'r') as f:
            credentials_data = json.load(f)
            service_account_id = credentials_data.get("client_email", "Unknown")
            logger.info(f"Using service account: {service_account_id} from {credentials_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in service account credentials file: {credentials_path}")
    except Exception as e:
        raise ValueError(f"Error reading service account credentials file {credentials_path}: {str(e)}")
    
    return required_vars

# Initialize FastAPI
app = FastAPI(
    title="WhisperX Transcription Service",
    description="Audio transcription service using WhisperX",
    version="1.0.0",
    on_startup=[get_env_config]
)

# Initialize Supabase
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

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
        if not (v.startswith('gs://') or v.startswith('https://storage.googleapis.com/')):
            raise ValueError("ruta_audio debe comenzar con gs:// o https://storage.googleapis.com/")
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
    """Parse GCS URL (gs:// or https://storage.googleapis.com/) into bucket and blob name"""
    if gcs_url.startswith("gs://"):
        if len(gcs_url) <= 5:
            raise ValueError("La URL gs:// está vacía o incompleta")
        parts = gcs_url[5:].split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""
    elif gcs_url.startswith("https://storage.googleapis.com/"):
        if len(gcs_url) <= 31:
            raise ValueError("La URL HTTPS está vacía o incompleta")
        parts = gcs_url[31:].split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""
    else:
        raise ValueError("La URL debe comenzar con gs:// o https://storage.googleapis.com/")

async def download_from_gcs(gcs_url: str, local_path: str) -> None:
    """Download file from GCS asynchronously with improved handling"""
    bucket_name, blob_name = parse_gcs_url(gcs_url)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Verificar si el archivo ya existe localmente y es válido
    if os.path.exists(local_path):
        logger.warning(f"El archivo {local_path} ya existe, se sobrescribirá")

    # Implementar reintentos y descargar
    @retry.Retry(predicate=retry.if_exception_type(Exception), initial=1.0, maximum=10.0, multiplier=2.0)
    def download_with_retry():
        # Opcional: Verificar existencia del blob antes de descargar
        if not blob.exists():
            raise FileNotFoundError(f"El objeto {blob_name} no existe en el bucket {bucket_name}")
        blob.download_to_filename(local_path)

    try:
        await asyncio.to_thread(download_with_retry)
        logger.info(f"Descargado {gcs_url} a {local_path}")
    except Exception as e:
        logger.error(f"Error descargando {gcs_url}: {str(e)}")
        raise

async def transcribe_audio(audio_path: str, tier: str) -> dict:
    """Transcribe audio using WhisperX based on tier"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Reemplazar asyncio.timeout con asyncio.wait_for para compatibilidad con Python 3.10
        async def transcribe_with_timeout():
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

        # Usar wait_for con un timeout de 300 segundos (5 minutos)
        return await asyncio.wait_for(transcribe_with_timeout(), timeout=300)
                
    except asyncio.TimeoutError:
        logger.error("Transcription exceeded 5 minute timeout")
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise

async def upload_result_to_gcs(result: dict, user_id: str, job_id: str) -> str:
    try:
        client = storage.Client()
        bucket = client.bucket("whisperx-results")
        
        user_folder = f"{user_id}/"
        job_folder = f"{user_folder}{job_id}/"
        
        logger.info(f"Preparando subida de resultados para usuario {user_id}, job {job_id} en {job_folder}")
        
        files_to_upload = {
            "transcript.json": json.dumps(result.get("transcription", {}), ensure_ascii=False),
            "metadata.json": json.dumps({
                "language": result.get("language", "unknown"),
                "tier": result.get("tier", "unknown"),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, ensure_ascii=False),
        }
        
        uploaded_urls = []
        for filename, content in files_to_upload.items():
            blob_name = f"{job_folder}{filename}"
            blob = bucket.blob(blob_name)
            logger.info(f"Subiendo {blob_name} a gs://whisperx-results/{blob_name}")
            blob.upload_from_string(content, content_type="application/json")
            
            # No verificamos blob.exists() inmediatamente, confiamos en la subida
            file_url = f"gs://whisperx-results/{blob_name}"
            uploaded_urls.append(file_url)
            logger.info(f"Archivo subido exitosamente: {file_url}")
        
        result_url = f"gs://whisperx-results/{job_folder}"
        logger.info(f"Todos los archivos subidos a {result_url}")
        
        return result_url
        
    except Exception as e:
        logger.error(f"Error subiendo resultados a GCS: {str(e)}. Bucket: whisperx-results, Job: {job_id}")
        raise

# --- Supabase helpers ---
async def get_archivo_id_from_ruta_audio(ruta_audio: str) -> Optional[str]:
    """Obtiene el ID de archivos_subidos basado en la ruta_audio."""
    try:
        async with db_pool.connection() as conn:
            query = """
                SELECT id FROM archivos_subidos WHERE ruta_archivo = %s;
            """
            result = await conn.execute(query, (ruta_audio,))
            row = await result.fetchone()
            if row:
                return row[0]
            else:
                logger.error(f"No se encontró un registro en archivos_subidos para ruta_audio: {ruta_audio}")
                return None
    except Exception as e:
        logger.error(f"Error consultando archivos_subidos para ruta_audio {ruta_audio}: {str(e)}")
        raise

# Database connection helper
async def get_db_connection():
    """Get a database connection from the pool"""
    async with db_pool.connection() as conn:
        yield conn
    

async def update_transcription_records(
    job_id: str,
    status: JobStatus,
    result: dict,
    user_id: str,
    ruta_audio: str  # Añadimos ruta_audio como parámetro
) -> None:
    try:
        now = datetime.now(timezone.utc)
        async with db_pool.connection() as conn:
            async with conn.transaction():
                if status == JobStatus.FAILED:
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
                    logger.info(f"Job {job_id} marcado como FAILED en transcription_jobs")
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
                    logger.info(f"Job {job_id} actualizado a COMPLETED en transcription_jobs")

                    # Obtener el archivo_id desde archivos_subidos
                    archivo_id = await get_archivo_id_from_ruta_audio(ruta_audio)
                    if not archivo_id:
                        raise ValueError(f"No se encontró archivo_id para ruta_audio: {ruta_audio}")

                    # Inserta el registro en transcriptions usando archivo_id
                    transcription_id = str(uuid4())
                    insert_query = """
                        INSERT INTO transcriptions (id, archivo_id, texto_transcripcion, fecha_generacion, estado, idioma)
                        VALUES (%s, %s, %s, %s, %s, %s);
                    """
                    insert_params = (
                        transcription_id,
                        archivo_id,  # Usamos archivo_id en lugar de job_id
                        transcription_text,
                        now,
                        status.value,
                        transcription_data.get("language", "unknown")
                    )
                    await conn.execute(insert_query, insert_params)
                    logger.info(f"Transcripción {transcription_id} registrada en transcriptions con archivo_id {archivo_id}")
                else:
                    update_query = """
                        UPDATE transcription_jobs
                        SET estado = %s,
                            fecha_actualizacion = %s
                        WHERE id = %s;
                    """
                    update_params = (status.value, now, job_id)
                    await conn.execute(update_query, update_params)
                    logger.info(f"Job {job_id} actualizado a {status.value} en transcription_jobs")
    except Exception as e:
        logger.error(f"Error actualizando registros: {str(e)}")
        raise


async def process_pubsub_message(message: PubSubMessage):
    """Process incoming Pub/Sub message"""
    job = None
    try:
        job_data = message.message.get("data", "")
        if not isinstance(job_data, dict):
            raise ValueError(f"Los datos del mensaje deben ser un diccionario, recibido: {job_data}")

        # Normalizar claves a minúsculas
        job_data = {k.lower(): v for k, v in job_data.items()}

        # Verificar si el mensaje tiene los campos obligatorios mínimos
        required_fields = {"id", "id_usuario", "ruta_audio", "estado"}
        if not all(field in job_data for field in required_fields):
            logger.info(f"Descartando mensaje incompatible: {job_data}")
            return {"status": "skipped", "reason": "incompatible_format"}

        # Convertir valores con tolerancia a None
        if job_data.get("estado"):
            estado = job_data["estado"].lower()
            if estado == "pendiente":
                estado = "pending"
            job_data["estado"] = JobStatus(estado)
        if job_data.get("nivel"):
            job_data["nivel"] = TierType(job_data["nivel"])
        for date_field in ["fecha_creacion", "fecha_actualizacion"]:
            if job_data.get(date_field):
                try:
                    job_data[date_field] = datetime.fromisoformat(job_data[date_field].replace('Z', '+00:00'))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error parsing {date_field}: {e}, usando valor por defecto")
                    job_data[date_field] = datetime.now(timezone.utc)

        # Crear el job con manejo de valores opcionales
        job = TranscriptionJob(**{k: v for k, v in job_data.items() if v is not None})

        if job.intentos >= 3:
            await update_transcription_records(
                job.id,
                JobStatus.FAILED,
                {"error": "Maximum retry attempts exceeded", "attempts": job.intentos},
                job.id_usuario,
                job.ruta_audio
            )
            return {"status": "failed", "reason": "max_retries"}

        result = await process_transcription_job(job)
        await update_transcription_records(
            job.id,
            JobStatus.COMPLETED,
            result,
            job.id_usuario,
            job.ruta_audio  # Pasamos ruta_audio aquí
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
                job.id_usuario,
                job.ruta_audio  # Pasamos ruta_audio aquí también
            )
        raise

async def process_transcription_job(job: TranscriptionJob) -> dict:
    local_audio_path = os.path.join(tempfile.gettempdir(), f"{job.id}.mp3")
    tier = job.nivel.value if job.nivel else "tier1"
    
    try:
        await download_from_gcs(job.ruta_audio, local_audio_path)
        result = await transcribe_audio(local_audio_path, tier)
        result["tier"] = tier
        result_path = await upload_result_to_gcs(result, job.id_usuario, job.id)
        return {
            "transcription": result,
            "result_path": result_path  # Esto ahora es la URL de la carpeta del job
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
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "database": "unknown",
            "supabase": "unknown",
            "gcs": "unknown",
            "pubsub": "unknown"
        },
        "errors": {},
        "service_account": "unknown"  # Nuevo campo para mostrar la cuenta de servicio
    }

    # Obtener información de las credenciales
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.isfile(credentials_path):
        try:
            with open(credentials_path, 'r') as f:
                credentials_data = json.load(f)
                health_status["service_account"] = credentials_data.get("client_email", "Unknown")
        except Exception as e:
            health_status["errors"]["credentials"] = f"Error reading credentials: {str(e)}"

    try:
        # Supabase check
        data = supabase.table("transcription_jobs").select("id").limit(1).execute()
        health_status["services"]["supabase"] = "connected"
    except Exception as e:
        health_status["services"]["supabase"] = "disconnected"
        health_status["errors"]["supabase"] = str(e)

    try:
        # Database check
        async with db_pool.connection() as conn:
            await conn.execute("SELECT 1")
        health_status["services"]["database"] = "connected"
    except Exception as e:
        health_status["services"]["database"] = "disconnected"
        health_status["errors"]["database"] = str(e)

    try:
        # GCS check
        storage.Client().list_buckets(max_results=1)
        health_status["services"]["gcs"] = "connected"
    except Exception as e:
        health_status["services"]["gcs"] = "disconnected"
        health_status["errors"]["gcs"] = str(e)

    try:
        # Pub/Sub check
        subscriber = pubsub_v1.SubscriberClient()
        subscription = "projects/nyro-450117/subscriptions/whisperx-microservice"
        subscriber.get_subscription(subscription=subscription)
        health_status["services"]["pubsub"] = "connected"
    except Exception as e:
        health_status["services"]["pubsub"] = "disconnected"
        health_status["errors"]["pubsub"] = str(e)

    # Check if any service is disconnected
    if any(status == "disconnected" for status in health_status["services"].values()):
        health_status["status"] = "unhealthy"
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Service partially available or unavailable",
                "status": health_status
            }
        )

    return health_status
    
# Cleanup resources
@app.on_event("shutdown")
async def cleanup_resources():
    """Cleanup resources on shutdown"""
    global pending_tasks
    if pending_tasks:
        logger.info(f"Esperando {len(pending_tasks)} tareas pendientes...")
        await asyncio.gather(*pending_tasks, return_exceptions=True)
    await db_pool.close()
    logger.info("Recursos limpiados exitosamente")

def fix_malformed_json(raw_data: str) -> dict:
    """Convierte un mensaje malformado como {key:value,...} en un JSON válido."""
    try:
        # Eliminar llaves externas y limpiar espacios
        cleaned_data = raw_data.strip().strip('{}').strip()
        
        # Usar regex para dividir pares clave-valor, preservando URLs y valores complejos
        pattern = r'(\w+):([^,]+?(?=(?:,\w+:)|$))'
        pairs = re.findall(pattern, cleaned_data)
        
        # Construir diccionario
        result = {}
        for key, value in pairs:
            value = value.strip()
            # Manejar valores especiales
            if value == 'null':
                result[key] = None
            elif value.startswith('http') or value.startswith('gs://'):
                result[key] = value  # Mantener URLs sin comillas adicionales
            elif re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', value):
                result[key] = value  # Mantener fechas como cadenas
            else:
                result[key] = value.strip('"')  # Quitar comillas si las hay
        
        # Convertir a JSON y devolver
        json_str = json.dumps(result)
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Error reparando mensaje JSON: {e}")
        raise ValueError(f"No se pudo reparar el mensaje: {e}")

async def process_and_ack(pubsub_message):
    task = asyncio.current_task()
    pending_tasks.add(task)
    try:
        raw_data = pubsub_message.data.decode("utf-8")
        logger.info(f"Mensaje recibido (crudo): {raw_data}")
        
        try:
            message_data = json.loads(raw_data)
        except json.JSONDecodeError:
            logger.warning(f"Parseando mensaje malformado: {raw_data}")
            message_data = fix_malformed_json(raw_data)
            logger.info(f"Mensaje reparado: {message_data}")

        if not isinstance(message_data, dict):
            logger.error(f"El mensaje reparado no es un diccionario: {message_data}")
            pubsub_message.nack()
            return

        message_wrapper = PubSubMessage(
            message={"data": message_data},
            subscription="projects/nyro-450117/subscriptions/whisperx-microservice"
        )
        
        await process_pubsub_message(message_wrapper)
        pubsub_message.ack()
        logger.info(f"Mensaje procesado con éxito: {message_data.get('id', 'unknown')}")
    except Exception as e:
        logger.error(f"Error procesando mensaje Pub/Sub: {e}")
        pubsub_message.nack()
    finally:
        pending_tasks.remove(task)

def pubsub_callback(message):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(process_and_ack(message))
    finally:
        loop.close()

def start_pubsub_listener():
    """Inicia el listener de Pub/Sub en un hilo separado."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.isfile(credentials_path):
        try:
            with open(credentials_path, 'r') as f:
                credentials_data = json.load(f)
                service_account_id = credentials_data.get("client_email", "Unknown")
                logger.info(f"Starting Pub/Sub listener with service account: {service_account_id}")
        except Exception as e:
            logger.error(f"Error reading credentials for Pub/Sub listener: {str(e)}")
    
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = "projects/nyro-450117/subscriptions/whisperx-microservice"
    future = subscriber.subscribe(subscription_path, pubsub_callback)
    logger.info(f"Escuchando mensajes en {subscription_path}...")
    try:
        future.result()
    except Exception as e:
        logger.error(f"Excepción en la escucha de {subscription_path}: {e}")
        future.cancel()

@app.on_event("startup")
async def startup_event():
    global db_pool
    db_pool = AsyncConnectionPool(os.getenv("DATABASE_URL"), min_size=2, max_size=10)
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