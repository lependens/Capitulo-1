from fastapi import FastAPI, Request, Form, BackgroundTasks, HTTPException
from fastapi.templating import Jinja2Templates
from typing import Optional
import os

import siar_worker

app = FastAPI()
templates = Jinja2Templates(directory="templates")

API_KEY = os.getenv("SIAR_API_KEY")  # Compatibilidad con versiones antiguas; v2.2 usa NIF/password.
TG_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT = os.getenv("TG_CHAT_ID")
GH_TOKEN = os.getenv("GITHUB_TOKEN")
TARGET_DATE_DEFAULT = os.getenv("SIAR_TARGET_DATE", "2025-12-31")

# Diccionario en memoria para Live Status.
task_status = {}


def _normalizar_estacion(estacion: str) -> str:
    return (estacion or "").strip().upper()


def _normalizar_objetivo(fecha: str) -> str:
    try:
        return siar_worker.fecha_iso(fecha)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/")
def home(request: Request, objetivo: str = TARGET_DATE_DEFAULT):
    try:
        objetivo_iso = _normalizar_objetivo(objetivo)
    except HTTPException:
        objetivo_iso = TARGET_DATE_DEFAULT
    estaciones = siar_worker.scan_repo(objetivo_iso)
    leyenda = siar_worker.get_leyenda()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "estaciones": estaciones,
            "leyenda": leyenda,
            "objetivo": objetivo_iso,
            "objetivo_es": siar_worker.formatear_fecha_es(objetivo_iso),
        },
    )


@app.get("/status")
def get_status():
    return task_status


@app.get("/api_usage")
def api_usage():
    data = siar_worker.get_api_usage()
    if "error" in data or "error_token" in data:
        raise HTTPException(status_code=502, detail=data.get("error") or data.get("error_token"))
    return data


@app.get("/analyze")
def analyze(estacion: str, objetivo: str = TARGET_DATE_DEFAULT):
    estacion = _normalizar_estacion(estacion)
    if not estacion:
        raise HTTPException(status_code=400, detail="Falta la estación")
    objetivo_iso = _normalizar_objetivo(objetivo)
    return siar_worker.analyze_station(estacion, objetivo_iso)


@app.post("/start")
def start_download(
    background_tasks: BackgroundTasks,
    estacion: str = Form(...),
    objetivo: Optional[str] = Form(None),
    # Compatibilidad con el formulario antiguo.
    f_inicio: Optional[str] = Form(None),
    f_fin: Optional[str] = Form(None),
):
    estacion = _normalizar_estacion(estacion)
    if not estacion:
        raise HTTPException(status_code=400, detail="Falta la estación")

    raw_target = objetivo or f_fin or TARGET_DATE_DEFAULT
    objetivo_iso = _normalizar_objetivo(raw_target)

    current = task_status.get(estacion, {})
    if current.get("running"):
        return {"status": "busy", "estacion": estacion, "message": "Ya hay una tarea activa para esta estación."}

    task_status[estacion] = {
        "msg": f"Encolando sincronización hasta {siar_worker.formatear_fecha_es(objetivo_iso)}…",
        "error": False,
        "ts": "",
        "running": True,
        "target": objetivo_iso,
    }
    background_tasks.add_task(
        _run_sync_task,
        estacion,
        objetivo_iso,
    )
    return {"status": "ok", "estacion": estacion, "objetivo": objetivo_iso}


def _run_sync_task(estacion: str, objetivo_iso: str):
    try:
        siar_worker.sync_station(
            estacion,
            objetivo_iso,
            TG_TOKEN,
            TG_CHAT,
            GH_TOKEN,
            task_status,
        )
    except Exception as exc:
        import traceback
        ts = __import__("datetime").datetime.now().strftime("%H:%M:%S")
        task_status[estacion] = {
            "msg": f"❌ Excepción no controlada: {exc}",
            "error": True,
            "ts": ts,
            "running": False,
            "target": objetivo_iso,
            "trace": traceback.format_exc()[-3000:],
        }
        if TG_TOKEN and TG_CHAT:
            siar_worker.send_telegram(TG_TOKEN, TG_CHAT, f"[{ts}] {estacion}: Excepción no controlada: {exc}")
    finally:
        if estacion in task_status:
            task_status[estacion]["running"] = False
