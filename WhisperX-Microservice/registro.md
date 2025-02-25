# Registro del Microservicio WhisperX

## Para Crear un env en Windows
```bash
python3.11 -m venv test_env
.\test_env\Scripts\activate
pip install -r requirements.txt
```

Para test, correr:
```bash
docker run --gpus all -p 8080:8080 --env-file .env ms-whisperx-local
```
Para hacer build:
```bash
docker build --progress=plain --no-cache -t ms-whisperx-local -f Dockerfile .
```

Publicar mensaje de prueba:
```bash
gcloud pubsub topics publish projects/nyro-450117/topics/transcription-jobs --message '{"id":"44a94574-807a-4da1-a37f-24a987bcf6d6","id_usuario":"3c04aecd-092d-4b48-a5fd-3fd097aa5536","ruta_audio":"https://storage.googleapis.com/nyro-files/usuarios/3c04aecd-092d-4b48-a5fd-3fd097aa5536/audios/2025/02/21/audio_1740155940551_videoprueba.mp3","estado":"pendiente","intentos":0,"fecha_creacion":"2025-02-21T16:39:17.154+00:00","fecha_actualizacion":"2025-02-21T16:39:17.154+00:00","mensaje_error":null,"nivel":null}'
```

## Subir imagen a Google Cloud
Solución 2: Subir una imagen Docker al repositorio audizor-docker
Si tu intención es usar el Dockerfile para construir una imagen Docker y subirla al repositorio audizor-docker (formato DOCKER), sigue estos pasos:

Construye la imagen Docker localmente:
 
docker build -t southamerica-east1-docker.pkg.dev/nyro-450117/audizor-docker/ms-whisperx:1.0.0 .
Esto usa el Dockerfile en el directorio actual (asegúrate de estar en E:\Nyro\Audizor\WhisperX-Microservice).
El nombre southamerica-east1-docker.pkg.dev/nyro-450117/audizor-docker/ms-whisperx:1.0.0 sigue la convención de Artifact Registry: <location>-docker.pkg.dev/<project>/<repository>/<image>:<tag>.
Autentica Docker con Google Cloud:

gcloud auth configure-docker southamerica-east1-docker.pkg.dev
Responde "Y" si te pide confirmar la adición al archivo de configuración de Docker.
Sube la imagen:

docker push southamerica-east1-docker.pkg.dev/nyro-450117/audizor-docker/ms-whisperx:1.0.0
Verifica el resultado: Lista las imágenes en el repositorio para confirmar:

gcloud artifacts docker images list southamerica-east1-docker.pkg.dev/nyro-450117/audizor-docker
