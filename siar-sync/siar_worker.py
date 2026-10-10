import os
import subprocess
import tempfile
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

API_BASE = "https://servicio.mapa.gob.es/siarapi/API/V1"
TIPO_DATOS = "Diarios"
AMBITO = "ESTACION"

REPO_PATH = "/app/Capitulo-1"
DATOS_PATH = os.path.join(REPO_PATH, "datos_siar_baleares")

SIAR_NIF = os.environ.get("SIAR_NIF")
SIAR_PASSWORD = os.environ.get("SIAR_PASSWORD")
SIAR_CA_BUNDLE = (os.environ.get("SIAR_CA_BUNDLE") or "").strip() or None

# Parámetros conservadores y configurables desde .env.
CHUNK_DAYS = max(1, int(os.environ.get("SIAR_CHUNK_DAYS", "7")))
API_PAUSE_SECONDS = max(0, int(os.environ.get("SIAR_PAUSE_SECONDS", "3")))
API_RETRIES = max(1, int(os.environ.get("SIAR_API_RETRIES", "3")))
PUSH_EVERY_CHUNKS = max(1, int(os.environ.get("SIAR_PUSH_EVERY_CHUNKS", "4")))
TARGET_DATE_DEFAULT = os.environ.get("SIAR_TARGET_DATE", "2025-12-31")
SIAR_CIVIL_TIMEZONE = "Europe/Madrid"
# Reconsulta los últimos N días del objetivo para recoger correcciones publicadas posteriormente.
REFRESH_RECENT_DAYS = max(0, int(os.environ.get("SIAR_REFRESH_RECENT_DAYS", "31")))

_GIT_LOCK = threading.Lock()

CANONICAL_COLUMNS = [
    "Fecha",
    "TempMedia",
    "TempMax",
    "HorMinTempMax",
    "TempMin",
    "HorMinTempMin",
    "HumedadMedia",
    "HumedadMax",
    "HorMinHumMax",
    "HumedadMin",
    "HorMinHumMin",
    "VelViento",
    "DirViento",
    "VelVientoMax",
    "HorMinVelMax",
    "DirVientoVelMax",
    "Radiacion",
    "Precipitacion",
    "TempSuelo1",
    "TempSuelo2",
    "EtPMon",
    "PePMon",
    "Estacion",
]

FIELD_ALIASES = {"humedadMin": "HumedadMin"}


class SiarAuthError(Exception):
    pass


class SiarDataError(Exception):
    pass


class SiarDailyLimitError(Exception):
    pass


class SiarTlsConfigError(requests.RequestException):
    """Configuración TLS SIAR inválida; nunca desactiva la verificación."""


def _siar_verify_value():
    """Devuelve True o la ruta del CA bundle opcional usado solo por SIAR.

    Sin SIAR_CA_BUNDLE se conserva el comportamiento estándar de requests
    (verify=True). Si la variable está configurada, la ruta debe existir y ser
    legible. Nunca se devuelve False.
    """
    if not SIAR_CA_BUNDLE:
        return True

    bundle = os.path.abspath(os.path.expanduser(SIAR_CA_BUNDLE))
    if not os.path.isfile(bundle):
        raise SiarTlsConfigError(
            f"SIAR_CA_BUNDLE no existe o no es un archivo: {bundle}"
        )
    if not os.access(bundle, os.R_OK):
        raise SiarTlsConfigError(
            f"SIAR_CA_BUNDLE no es legible por el proceso: {bundle}"
        )
    return bundle


def _siar_get(url, **kwargs):
    """GET exclusivo para SIAR con validación TLS siempre activa."""
    if "verify" in kwargs:
        raise TypeError("_siar_get gestiona internamente el parámetro verify")
    kwargs["verify"] = _siar_verify_value()
    return requests.get(url, **kwargs)


def normalizar_fecha(valor):
    """Acepta YYYY-MM-DD y DD/MM/YYYY; devuelve date."""
    if isinstance(valor, date):
        return valor
    texto = str(valor or "").strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(texto, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Fecha inválida: {texto!r}. Usa DD/MM/YYYY o YYYY-MM-DD.")


def fecha_iso(valor):
    return normalizar_fecha(valor).isoformat()


def formatear_fecha_es(valor):
    return normalizar_fecha(valor).strftime("%d/%m/%Y")


def _siar_timestamp_to_civil_date(valor):
    """Convierte timestamps administrativos SIAR a fecha civil Europe/Madrid.

    SIAR ha devuelto históricamente medianoches locales serializadas como
    timestamps UTC sin sufijo de zona (p. ej. 22:00/23:00 del día anterior).
    Para preservar la fecha civil, los timestamps sin zona se interpretan como
    UTC y se convierten a Europe/Madrid antes de extraer la fecha.
    """
    ts = pd.to_datetime(str(valor), errors="raise")
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(SIAR_CIVIL_TIMEZONE).date()


def get_leyenda():
    path = os.path.join(REPO_PATH, "estaciones_baleares.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="error").fillna("-")
        return {"columnas": df.columns.tolist(), "filas": df.values.tolist()}
    except Exception as exc:
        return {"error": str(exc)}


def send_telegram(token, chat_id, text):
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
    except requests.RequestException:
        pass


def _csv_path(estacion):
    return os.path.join(DATOS_PATH, f"{estacion}_datos_completos.csv")


def _read_dates(path):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path, usecols=["Fecha"], on_bad_lines="error")
    fechas = pd.to_datetime(df["Fecha"], errors="coerce").dropna()
    return set(fechas.dt.date.tolist())


def _missing_ranges(fechas, start, end):
    """Devuelve rangos inclusivos de días sin dato entre start y end."""
    if start > end:
        return []
    existentes = set(fechas)
    ranges = []
    cursor = start
    while cursor <= end:
        while cursor <= end and cursor in existentes:
            cursor += timedelta(days=1)
        if cursor > end:
            break
        ini = cursor
        while cursor <= end and cursor not in existentes:
            cursor += timedelta(days=1)
        ranges.append((ini, cursor - timedelta(days=1)))
    return ranges


def _count_gap_days(fechas, start, end):
    if start > end:
        return 0
    total = (end - start).days + 1
    return total - sum(1 for f in fechas if start <= f <= end)


def analyze_station(estacion, target_date):
    estacion = str(estacion).strip().upper()
    target = normalizar_fecha(target_date)
    path = _csv_path(estacion)

    if not os.path.exists(path):
        return {
            "estacion": estacion,
            "exists": False,
            "inicio": None,
            "fin": None,
            "filas": 0,
            "dias_unicos": 0,
            "huecos": 0,
            "pendientes": None,
            "pendiente_desde": None,
            "pendiente_hasta": target.isoformat(),
            "rangos_pendientes": [],
            "estado": "SIN CSV",
        }

    df = pd.read_csv(path, usecols=["Fecha"], on_bad_lines="error")
    fechas_dt = pd.to_datetime(df["Fecha"], errors="coerce").dropna().dt.date
    fechas = set(fechas_dt.tolist())
    if not fechas:
        return {
            "estacion": estacion,
            "exists": True,
            "inicio": None,
            "fin": None,
            "filas": len(df),
            "dias_unicos": 0,
            "huecos": 0,
            "pendientes": None,
            "pendiente_desde": None,
            "pendiente_hasta": target.isoformat(),
            "rangos_pendientes": [],
            "estado": "CSV SIN FECHAS",
        }

    inicio = min(fechas)
    fin = max(fechas)
    limite_gaps = min(fin, target)
    huecos = _count_gap_days(fechas, inicio, limite_gaps) if inicio <= limite_gaps else 0

    rangos = _missing_ranges(fechas, inicio, target) if inicio <= target else []
    pendientes = sum((b - a).days + 1 for a, b in rangos)
    pendiente_desde = rangos[0][0].isoformat() if rangos else None

    if target < inicio:
        estado = "OBJETIVO ANTERIOR"
    elif pendientes == 0:
        estado = "COMPLETA"
    elif huecos > 0 and fin < target:
        estado = "CON HUECOS + PENDIENTE"
    elif huecos > 0:
        estado = "CON HUECOS"
    else:
        estado = "PENDIENTE"

    # Para la interfaz mostramos solo una muestra; el worker procesa todos los rangos.
    muestra = [
        {"inicio": a.isoformat(), "fin": b.isoformat(), "dias": (b - a).days + 1}
        for a, b in rangos[:20]
    ]

    return {
        "estacion": estacion,
        "exists": True,
        "inicio": inicio.isoformat(),
        "fin": fin.isoformat(),
        "filas": len(df),
        "dias_unicos": len(fechas),
        "huecos": huecos,
        "pendientes": pendientes,
        "pendiente_desde": pendiente_desde,
        "pendiente_hasta": target.isoformat(),
        "rangos_pendientes": muestra,
        "rangos_pendientes_total": len(rangos),
        "estado": estado,
    }


def scan_repo(target_date=TARGET_DATE_DEFAULT):
    target = normalizar_fecha(target_date)
    os.makedirs(DATOS_PATH, exist_ok=True)
    stations = []
    for file in sorted(os.listdir(DATOS_PATH)):
        if not file.endswith("_datos_completos.csv"):
            continue
        estacion = file[: -len("_datos_completos.csv")]
        try:
            stations.append(analyze_station(estacion, target))
        except Exception:
            # La estación se omite de la tabla si el CSV está corrupto.
            continue
    return stations


def obtener_token_siar(nif, password):
    if not nif or not password:
        raise SiarAuthError("Faltan SIAR_NIF / SIAR_PASSWORD")
    try:
        r1 = _siar_get(
            f"{API_BASE}/Autenticacion/cifrarCadena",
            params={"cadena": nif},
            timeout=20,
        )
        r1.raise_for_status()
        r2 = _siar_get(
            f"{API_BASE}/Autenticacion/cifrarCadena",
            params={"cadena": password},
            timeout=20,
        )
        r2.raise_for_status()
        r3 = _siar_get(
            f"{API_BASE}/Autenticacion/obtenerToken",
            params={"Usuario": r1.text.strip(), "Password": r2.text.strip()},
            timeout=20,
        )
        r3.raise_for_status()
        token = r3.text.strip()
        if len(token) < 20:
            raise SiarAuthError("Token inválido devuelto por SIAR")
        return token
    except requests.RequestException as exc:
        raise SiarAuthError(f"Error obteniendo token: {exc}") from exc


def _message_looks_token_error(message):
    low = str(message or "").lower()
    return "token" in low or "claveapi" in low or "credencial" in low


def _message_looks_rate_error(message):
    low = str(message or "").lower()
    markers = (
        "rebasaría",
        "rebasaria",
        "límite",
        "limite",
        "cuota",
        "accesos por minuto",
        "accesos diarios",
        "registros por minuto",
        "registros diarios",
    )
    return any(m in low for m in markers)


def _parse_response_error(res):
    body_text = (res.text or "").strip()
    try:
        body = res.json()
    except ValueError:
        body = None

    if isinstance(body, dict) and body.get("MensajeRespuesta"):
        msg = str(body["MensajeRespuesta"])
        if _message_looks_token_error(msg):
            return {"error_token": msg}
        if _message_looks_rate_error(msg):
            return {"error_rate_limit": msg}
        return {"error": f"SIAR: {msg}"}

    if _message_looks_token_error(body_text):
        return {"error_token": body_text[:500]}
    if _message_looks_rate_error(body_text):
        return {"error_rate_limit": body_text[:500]}

    if res.status_code in (401, 403):
        # Un 401/403 sin mensaje claro se trata como autenticación, no como cuota.
        return {"error_token": f"HTTP {res.status_code}: {body_text[:500]}"}
    if res.status_code != 200:
        return {"error": f"HTTP {res.status_code}: {body_text[:500]}"}
    return None


TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}
SIAR_RETRY_BACKOFF_SECONDS = (5, 15, 30, 60)


def _is_transient_request_exception(exc):
    """Solo considera transitorios los fallos de conexión/TLS y timeout."""
    return isinstance(exc, (requests.Timeout, requests.ConnectionError))


def _wait_before_siar_retry(component, detail, intento):
    """Espera con backoff conservador antes del siguiente intento SIAR."""
    if intento >= API_RETRIES:
        return
    delay = SIAR_RETRY_BACKOFF_SECONDS[
        min(max(intento - 1, 0), len(SIAR_RETRY_BACKOFF_SECONDS) - 1)
    ]
    ts = datetime.now().strftime("%H:%M:%S")
    print(
        f"[{ts}] SIAR {component}: ⚠️ {detail} "
        f"(intento {intento}/{API_RETRIES}); reintentando en {delay}s.",
        flush=True,
    )
    time.sleep(delay)


def obtener_accesos(token):
    """Consulta Info/ACCESOS con reintentos solo ante fallos transitorios."""
    last_error = None

    for intento in range(1, API_RETRIES + 1):
        try:
            res = _siar_get(
                f"{API_BASE}/Info/ACCESOS",
                params={"token": token},
                timeout=20,
            )
        except requests.RequestException as exc:
            last_error = str(exc)
            if _is_transient_request_exception(exc) and intento < API_RETRIES:
                _wait_before_siar_retry("ACCESOS", last_error[:250], intento)
                continue
            return {"error": f"Error consultando ACCESOS: {last_error}"}

        error = _parse_response_error(res)
        if error:
            # Token y cuota son estados funcionales: no se reintentan aquí.
            if "error_token" in error or "error_rate_limit" in error:
                return error

            last_error = error.get("error", str(error))
            if res.status_code in TRANSIENT_HTTP_STATUSES and intento < API_RETRIES:
                _wait_before_siar_retry(
                    "ACCESOS",
                    f"HTTP {res.status_code}: {(res.text or '').strip()[:180]}",
                    intento,
                )
                continue

            if res.status_code in TRANSIENT_HTTP_STATUSES:
                return {
                    "error": (
                        f"ACCESOS no disponible tras {API_RETRIES} intento(s): "
                        f"{last_error}"
                    )
                }
            return error

        try:
            body = res.json()
        except ValueError:
            return {"error": f"ACCESOS no devolvió JSON: {(res.text or '')[:300]}"}

        datos = body.get("datos") or []
        if not isinstance(datos, list) or not datos:
            return {"error": "ACCESOS devolvió una respuesta sin datos"}
        info = datos[0]
        if not isinstance(info, dict):
            return {"error": "ACCESOS devolvió un formato inesperado"}
        return info

    return {
        "error": (
            f"Error consultando ACCESOS tras {API_RETRIES} intento(s): "
            f"{last_error or 'desconocido'}"
        )
    }


def obtener_estacion_info(token, codigo):
    """Devuelve la ficha de una estación desde Info/ESTACIONES.

    El manual oficial indica que este servicio devuelve Codigo,
    Fecha_Instalacion y Fecha_Baja (null si la estación sigue activa).
    """
    try:
        cuota = esperar_cuota(token, 1)
    except SiarDailyLimitError:
        raise
    if isinstance(cuota, dict) and "error_token" in cuota:
        return cuota
    if isinstance(cuota, dict) and "error" in cuota:
        return cuota

    try:
        res = _siar_get(
            f"{API_BASE}/Info/ESTACIONES",
            params={"token": token},
            timeout=30,
        )
    except requests.RequestException as exc:
        return {"error": f"Error consultando ESTACIONES: {exc}"}

    error = _parse_response_error(res)
    if error:
        return error

    try:
        body = res.json()
    except ValueError:
        return {"error": f"ESTACIONES no devolvió JSON: {(res.text or '')[:300]}"}

    datos = body.get("datos")
    if datos is None:
        datos = body.get("Datos")
    if not isinstance(datos, list):
        return {"error": "ESTACIONES devolvió un formato inesperado"}

    codigo_buscado = str(codigo).strip().upper()
    coincidencias = [
        item for item in datos
        if isinstance(item, dict) and str(item.get("Codigo", "")).strip().upper() == codigo_buscado
    ]
    if not coincidencias:
        return {"error": f"SIAR no encontró la estación {codigo_buscado} en Info/ESTACIONES"}

    info = coincidencias[0]
    fecha_instalacion_raw = info.get("Fecha_Instalacion")
    if not fecha_instalacion_raw:
        return {"error": f"SIAR no informó Fecha_Instalacion para {codigo_buscado}"}

    try:
        fecha_instalacion = _siar_timestamp_to_civil_date(fecha_instalacion_raw)
    except Exception as exc:
        return {"error": f"Fecha_Instalacion inválida para {codigo_buscado}: {fecha_instalacion_raw!r} ({exc})"}

    fecha_baja = None
    fecha_baja_raw = info.get("Fecha_Baja")
    if fecha_baja_raw not in (None, "", "null"):
        try:
            fecha_baja = _siar_timestamp_to_civil_date(fecha_baja_raw)
        except Exception as exc:
            return {"error": f"Fecha_Baja inválida para {codigo_buscado}: {fecha_baja_raw!r} ({exc})"}
        if fecha_baja < fecha_instalacion:
            return {"error": f"Fecha_Baja anterior a Fecha_Instalacion para {codigo_buscado}"}

    return {
        "info": info,
        "codigo": codigo_buscado,
        "fecha_instalacion": fecha_instalacion,
        "fecha_baja": fecha_baja,
    }


def _sleep_until_next_minute():
    ahora = time.time()
    restante = 60 - (ahora % 60)
    time.sleep(max(2, restante + 1))


def get_api_usage():
    """Obtiene los límites y contadores actuales sin exponer el token."""
    try:
        token = obtener_token_siar(SIAR_NIF, SIAR_PASSWORD)
    except SiarAuthError as exc:
        return {"error": str(exc)}
    info = obtener_accesos(token)
    if "error_token" in info:
        try:
            token = obtener_token_siar(SIAR_NIF, SIAR_PASSWORD)
            info = obtener_accesos(token)
        except SiarAuthError as exc:
            return {"error": str(exc)}
    if "error" in info or "error_token" in info:
        return info
    return {
        "peticiones_minuto": info.get("NumAccesosMinutoActual"),
        "max_peticiones_minuto": info.get("MaxAccesosMinuto"),
        "peticiones_dia": info.get("NumAccesosDiaActual"),
        "max_peticiones_dia": info.get("MaxAccesosDia"),
        "registros_minuto": info.get("RegistrosAcumuladosMinuto"),
        "max_registros_minuto": info.get("MaxRegistrosMinuto"),
        "registros_dia": info.get("RegistrosAcumuladosDia"),
        "max_registros_dia": info.get("MaxRegistrosDia"),
    }


def esperar_cuota(token, expected_records=1):
    """Consulta ACCESOS y espera solo cuando la cuota de minuto lo exige.
    Si se agota la cuota diaria, eleva SiarDailyLimitError para detener sin martillear la API.
    """
    info = obtener_accesos(token)
    if "error_token" in info:
        return info
    if "error" in info:
        return info

    max_req_min = int(info.get("MaxAccesosMinuto") or 0)
    req_min = int(info.get("NumAccesosMinutoActual") or 0)
    max_req_day = int(info.get("MaxAccesosDia") or 0)
    req_day = int(info.get("NumAccesosDiaActual") or 0)

    max_rec_min = int(info.get("MaxRegistrosMinuto") or 0)
    rec_min = int(info.get("RegistrosAcumuladosMinuto") or 0)
    max_rec_day = int(info.get("MaxRegistrosDia") or 0)
    rec_day = int(info.get("RegistrosAcumuladosDia") or 0)

    if max_req_day and req_day >= max_req_day:
        raise SiarDailyLimitError(
            f"Cuota diaria de peticiones agotada ({req_day}/{max_req_day})."
        )
    if max_rec_day and rec_day + expected_records > max_rec_day:
        raise SiarDailyLimitError(
            f"Cuota diaria de registros insuficiente ({rec_day}+{expected_records}>{max_rec_day})."
        )

    # Dejamos un hueco conservador de una petición para evitar caer exactamente sobre el límite.
    if max_req_min and req_min >= max_req_min - 1:
        _sleep_until_next_minute()
        return esperar_cuota(token, expected_records)
    if max_rec_min and rec_min + expected_records >= max_rec_min:
        _sleep_until_next_minute()
        return esperar_cuota(token, expected_records)

    return info


def fetch_data_rango(codigo, f_inicio, f_fin, token):
    """Descarga un bloque. Diferencia token inválido de cuota 403."""
    last_error = None
    expected_records = max(1, (normalizar_fecha(f_fin) - normalizar_fecha(f_inicio)).days + 1)

    for intento in range(1, API_RETRIES + 1):
        cuota = esperar_cuota(token, expected_records)
        if isinstance(cuota, dict) and "error_token" in cuota:
            return cuota
        if isinstance(cuota, dict) and "error" in cuota:
            return cuota

        try:
            res = _siar_get(
                f"{API_BASE}/Datos/{TIPO_DATOS}/{AMBITO}",
                params={
                    "token": token,
                    "Id": codigo,
                    "FechaInicial": f_inicio,
                    "FechaFinal": f_fin,
                    "DatosCalculados": "true",
                },
                timeout=60,
            )
        except requests.RequestException as exc:
            last_error = str(exc)
            if _is_transient_request_exception(exc) and intento < API_RETRIES:
                _wait_before_siar_retry(
                    f"Datos {codigo} {f_inicio}..{f_fin}",
                    last_error[:250],
                    intento,
                )
                continue
            return {"error": f"Error de red/API: {last_error}"}

        error = _parse_response_error(res)
        if error:
            if "error_token" in error or "error_rate_limit" in error:
                return error

            last_error = error.get("error", str(error))
            if res.status_code in TRANSIENT_HTTP_STATUSES and intento < API_RETRIES:
                _wait_before_siar_retry(
                    f"Datos {codigo} {f_inicio}..{f_fin}",
                    f"HTTP {res.status_code}: {(res.text or '').strip()[:180]}",
                    intento,
                )
                continue

            # 4xx/otros errores funcionales no se repiten indiscriminadamente.
            if res.status_code not in TRANSIENT_HTTP_STATUSES:
                return error
            continue

        try:
            body = res.json()
        except ValueError:
            return {"error": f"Respuesta no JSON: {(res.text or '')[:500]}"}

        data = body.get("datos")
        if data is None:
            data = body.get("Datos")
        data = data or []
        if not isinstance(data, list):
            return {"error": f"Formato inesperado de datos: {type(data).__name__}"}

        inicio_date = normalizar_fecha(f_inicio)
        fin_date = normalizar_fecha(f_fin)
        seen_dates = set()
        for row in data:
            if not isinstance(row, dict):
                return {"error": f"Fila SIAR no es objeto: {type(row).__name__}"}
            row["Estacion"] = codigo
            if "Fecha" not in row or not row["Fecha"]:
                return {"error": f"Fila SIAR sin Fecha en {f_inicio}..{f_fin}"}
            try:
                row_date = normalizar_fecha(str(row["Fecha"]).split("T", 1)[0])
            except ValueError as exc:
                return {"error": str(exc)}
            if row_date < inicio_date or row_date > fin_date:
                return {"error": f"SIAR devolvió una fecha fuera de rango: {row.get('Fecha')}"}
            if row_date in seen_dates:
                return {"error": f"SIAR devolvió más de un registro diario para {row_date.isoformat()}"}
            seen_dates.add(row_date)

        return data

    return {"error": f"Error de red/API tras {API_RETRIES} intentos: {last_error or 'desconocido'}"}


def _read_existing_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=CANONICAL_COLUMNS), CANONICAL_COLUMNS.copy()
    df = pd.read_csv(path, on_bad_lines="error")
    if "Fecha" not in df.columns:
        raise SiarDataError(f"CSV existente sin columna Fecha: {path}")
    return df, list(df.columns)


def _normalize_api_dataframe(datos, expected_columns):
    if not datos:
        return pd.DataFrame(columns=expected_columns)

    normalized = []
    for row in datos:
        item = dict(row)
        for src, dst in FIELD_ALIASES.items():
            if src in item and dst not in item:
                item[dst] = item.pop(src)
        normalized.append(item)

    df = pd.json_normalize(normalized)
    if "Fecha" not in df.columns:
        raise SiarDataError("SIAR devolvió datos sin Fecha")

    fechas = pd.to_datetime(df["Fecha"], errors="coerce")
    if fechas.isna().any():
        raise SiarDataError(f"SIAR devolvió {int(fechas.isna().sum())} fila(s) con Fecha inválida")
    df["Fecha"] = fechas.dt.strftime("%Y-%m-%d")

    for col in expected_columns:
        if col not in df.columns:
            df[col] = pd.NA

    return df[expected_columns].copy()


def _atomic_write_csv(df, path):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".siar_", suffix=".csv", dir=directory, text=True)
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        check = pd.read_csv(tmp_path, on_bad_lines="error")
        if list(check.columns) != list(df.columns):
            raise SiarDataError("La validación del CSV temporal detectó un cambio de esquema")
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _create_empty_canonical_csv(path):
    """Crea un CSV vacío con exactamente el esquema canónico de 23 columnas."""
    empty = pd.DataFrame(columns=CANONICAL_COLUMNS)
    _atomic_write_csv(empty, path)
    verified = pd.read_csv(path, on_bad_lines="error")
    if list(verified.columns) != CANONICAL_COLUMNS:
        raise SiarDataError("No se pudo crear el CSV canónico de estación nueva")
    return verified


def merge_and_save(path, datos, estacion, replace_existing=False):
    existing, columns = _read_existing_csv(path)
    incoming = _normalize_api_dataframe(datos, columns)
    if incoming.empty:
        return 0, None, existing

    incoming = incoming[
        incoming["Estacion"].astype(str).str.upper() == estacion.upper()
    ].copy()
    if incoming.empty:
        return 0, None, existing

    incoming_dates = set(incoming["Fecha"].tolist())
    if replace_existing:
        existing = existing[~existing["Fecha"].astype(str).isin(incoming_dates)].copy()
        new_rows = incoming
    else:
        existing_dates = set(
            pd.to_datetime(existing["Fecha"], errors="coerce")
            .dt.strftime("%Y-%m-%d")
            .dropna()
            .tolist()
        )
        new_rows = incoming[~incoming["Fecha"].isin(existing_dates)].copy()

    if new_rows.empty and not replace_existing:
        return 0, None, existing

    if existing.empty:
        combined = new_rows.copy()
    elif new_rows.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    combined["Fecha"] = pd.to_datetime(combined["Fecha"], errors="coerce")
    if combined["Fecha"].isna().any():
        raise SiarDataError("Se detectaron Fechas inválidas al combinar el CSV")
    combined.sort_values("Fecha", inplace=True)
    combined.drop_duplicates(subset=["Fecha"], keep="last" if replace_existing else "first", inplace=True)
    combined["Fecha"] = combined["Fecha"].dt.strftime("%Y-%m-%d")
    combined = combined[columns]

    _atomic_write_csv(combined, path)
    verified = pd.read_csv(path, on_bad_lines="error")
    if len(verified) != len(combined):
        raise SiarDataError("Fallo de persistencia: número de filas distinto tras escritura")

    if replace_existing:
        changed = len(new_rows)
        last_date = max(incoming["Fecha"].tolist()) if not incoming.empty else None
        return changed, last_date, verified

    persisted_dates = sorted(new_rows["Fecha"].tolist())
    return len(persisted_dates), persisted_dates[-1] if persisted_dates else None, verified


def _git_askpass_push(github_token, relative_paths=None):
    """Commit only the requested SIAR paths and publish them safely to GitHub.

    Remote changes are fetched before pushing. If origin/main has advanced,
    they are integrated with a normal merge. A real merge conflict aborts the
    merge and leaves the local SIAR commit/data intact. Network/TLS failures
    are retried conservatively.
    """
    if not github_token:
        return False, "Falta GITHUB_TOKEN"

    git_retry_delays = (0, 5, 15, 30)
    transient_markers = (
        "gnutls",
        "tls",
        "ssl",
        "timed out",
        "timeout",
        "connection reset",
        "connection refused",
        "connection aborted",
        "remote end hung up",
        "could not resolve",
        "network is unreachable",
        "temporary failure",
    )
    divergence_markers = (
        "fetch first",
        "non-fast-forward",
        "non fast forward",
        "tip of your current branch is behind",
        "remote contains work that you do not have locally",
    )
    max_reconciliations = 2

    with _GIT_LOCK:
        helper_path = None
        try:
            if relative_paths is None:
                relative_paths = ["datos_siar_baleares/"]

            common_git = {
                "cwd": REPO_PATH,
                "capture_output": True,
                "text": True,
            }

            # La configuración queda limitada a este repositorio.
            subprocess.run(
                ["git", "config", "--local", "user.email", "servidor@local.com"],
                check=True,
                **common_git,
            )
            subprocess.run(
                ["git", "config", "--local", "user.name", "DellServer-Admin"],
                check=True,
                **common_git,
            )
            subprocess.run(
                [
                    "git",
                    "remote",
                    "set-url",
                    "origin",
                    "https://github.com/lependens/Capitulo-1.git",
                ],
                check=True,
                **common_git,
            )

            env = os.environ.copy()
            with tempfile.NamedTemporaryFile(
                "w",
                prefix=".git-askpass-",
                delete=False,
            ) as helper:
                helper.write("#!/bin/sh\n")
                helper.write('case "$1" in\n')
                helper.write('  *Username*) echo "x-access-token";;\n')
                helper.write('  *Password*) printf "%s\\n" "$GITHUB_TOKEN_FOR_ASKPASS";;\n')
                helper.write("esac\n")
                helper_path = helper.name

            os.chmod(helper_path, 0o700)
            env.update(
                {
                    "GIT_ASKPASS": helper_path,
                    "GIT_TERMINAL_PROMPT": "0",
                    "GITHUB_TOKEN_FOR_ASKPASS": github_token,
                }
            )

            # El worker SOLO pone en staging las rutas que recibió.
            add = subprocess.run(
                ["git", "add", "--", *relative_paths],
                cwd=REPO_PATH,
                capture_output=True,
                text=True,
            )
            if add.returncode != 0:
                return False, f"Git Add Fail: {add.stderr.strip()[:300]}"

            status = subprocess.run(
                ["git", "status", "--porcelain", "--", *relative_paths],
                cwd=REPO_PATH,
                capture_output=True,
                text=True,
            )
            if status.returncode != 0:
                return False, f"Git Status Fail: {status.stderr.strip()[:300]}"

            # Primero conservamos en un commit local cualquier cambio SIAR.
            if status.stdout.strip():
                commit = subprocess.run(
                    [
                        "git",
                        "commit",
                        "-m",
                        f"Automático SIAR: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    ],
                    cwd=REPO_PATH,
                    capture_output=True,
                    text=True,
                )
                if commit.returncode != 0:
                    return False, f"Commit Fail: {commit.stderr.strip()[:300]}"

            def local_ahead_count():
                result = subprocess.run(
                    ["git", "rev-list", "--count", "origin/main..HEAD"],
                    cwd=REPO_PATH,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    return None
                try:
                    return int(result.stdout.strip() or "0")
                except ValueError:
                    return None

            def run_fetch():
                """Fetch origin/main with the same transient retry policy."""
                last_error = ""
                for intento, delay in enumerate(git_retry_delays, start=1):
                    if delay:
                        time.sleep(delay)

                    fetch = subprocess.run(
                        ["git", "fetch", "origin", "main"],
                        cwd=REPO_PATH,
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                    if fetch.returncode == 0:
                        return True, ""

                    last_error = (fetch.stderr or fetch.stdout or "").strip()
                    low_error = last_error.lower()

                    if intento < len(git_retry_delays) and any(
                        marker in low_error for marker in transient_markers
                    ):
                        continue

                    return False, last_error

                return False, last_error

            def integrate_remote():
                """Fetch origin/main and merge it when the remote is ahead."""
                ok, fetch_error = run_fetch()
                if not ok:
                    return False, f"Git Fetch Fail: {fetch_error[:400]}"

                counts = subprocess.run(
                    [
                        "git",
                        "rev-list",
                        "--left-right",
                        "--count",
                        "HEAD...origin/main",
                    ],
                    cwd=REPO_PATH,
                    capture_output=True,
                    text=True,
                )
                if counts.returncode != 0:
                    return False, (
                        "Git Divergence Check Fail: "
                        f"{counts.stderr.strip()[:300]}"
                    )

                try:
                    ahead, behind = map(int, counts.stdout.strip().split())
                except (ValueError, AttributeError):
                    return False, (
                        "Git Divergence Check Fail: "
                        f"salida inesperada {counts.stdout.strip()!r}"
                    )

                if behind == 0:
                    return True, ""

                merge = subprocess.run(
                    ["git", "merge", "--no-edit", "origin/main"],
                    cwd=REPO_PATH,
                    capture_output=True,
                    text=True,
                )
                if merge.returncode == 0:
                    return True, ""

                merge_error = (merge.stderr or merge.stdout or "").strip()

                # Si el merge no es resoluble automáticamente, nunca dejamos
                # el repositorio en estado de merge. El commit SIAR ya existe
                # y, por tanto, los datos locales siguen conservados.
                subprocess.run(
                    ["git", "merge", "--abort"],
                    cwd=REPO_PATH,
                    capture_output=True,
                    text=True,
                )

                return False, f"Merge Conflict: {merge_error[:400]}"

            # Si no hay cambios ni commits SIAR pendientes, no tocamos GitHub.
            ahead = local_ahead_count()
            if ahead is None:
                return False, "Git Ahead Check Fail"
            if ahead == 0:
                return True, "Sin cambios SIAR pendientes"

            # Antes de publicar, incorporamos cualquier trabajo que ya exista
            # en GitHub. Así los cambios manuales de documentación conviven
            # con los commits automáticos de los CSV.
            ok, msg = integrate_remote()
            if not ok:
                return False, msg

            reconciliations = 0

            while True:
                ahead = local_ahead_count()
                if ahead is None:
                    return False, "Git Ahead Check Fail"
                if ahead == 0:
                    return True, "Sin cambios SIAR pendientes"

                last_error = ""

                for intento, delay in enumerate(git_retry_delays, start=1):
                    if delay:
                        time.sleep(delay)

                    push = subprocess.run(
                        ["git", "push", "origin", "main"],
                        cwd=REPO_PATH,
                        capture_output=True,
                        text=True,
                        env=env,
                    )

                    if push.returncode == 0:
                        if intento == 1 and reconciliations == 0:
                            return True, "Subida OK a GitHub"
                        return True, (
                            "Subida OK a GitHub "
                            f"(intento {intento}/{len(git_retry_delays)})"
                        )

                    last_error = (push.stderr or push.stdout or "").strip()
                    low_error = last_error.lower()

                    # Otro usuario/proceso pudo hacer push después de nuestro
                    # fetch. Volvemos a sincronizar, pero limitamos el número
                    # de reconciliaciones para evitar bucles.
                    if any(marker in low_error for marker in divergence_markers):
                        if reconciliations >= max_reconciliations:
                            return False, f"Push Fail: {last_error[:400]}"

                        reconciliations += 1
                        ok, msg = integrate_remote()
                        if not ok:
                            return False, msg
                        break

                    if intento < len(git_retry_delays) and any(
                        marker in low_error for marker in transient_markers
                    ):
                        continue

                    return False, f"Push Fail: {last_error[:400]}"

                else:
                    return False, f"Push Fail: {last_error[:400]}"

                # Se ha producido una divergencia remota y se ha integrado.
                # Volvemos al bucle para intentar el push de nuevo.
                continue

        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"Git Fatal: {exc}"
        finally:
            if helper_path:
                try:
                    os.unlink(helper_path)
                except FileNotFoundError:
                    pass


def push_to_github(github_token, relative_paths=None):
    return _git_askpass_push(github_token, relative_paths)


def _make_chunks(ranges):
    for range_start, range_end in ranges:
        current = range_start
        while current <= range_end:
            chunk_end = min(current + timedelta(days=CHUNK_DAYS - 1), range_end)
            yield current, chunk_end
            current = chunk_end + timedelta(days=1)


def bg_download_task(estacion, f_inicio, f_fin, api_key, tg_token, tg_chat, gh_token, status_dict):
    """Compatibilidad con el endpoint antiguo: ahora sincroniza usando el objetivo final."""
    del api_key
    try:
        target = normalizar_fecha(f_fin or TARGET_DATE_DEFAULT)
    except ValueError as exc:
        ts = datetime.now().strftime("%H:%M:%S")
        status_dict[estacion] = {"msg": str(exc), "error": True, "ts": ts}
        return
    return sync_station(estacion, target, tg_token, tg_chat, gh_token, status_dict)


def sync_station(estacion, target_date, tg_token, tg_chat, gh_token, status_dict):
    estacion = str(estacion).strip().upper()
    target = normalizar_fecha(target_date)
    path = _csv_path(estacion)

    def update_log(msg, is_error=False, notify=False, extra=None):
        ts = datetime.now().strftime("%H:%M:%S")
        previous = status_dict.get(estacion, {}) or {}
        payload = {
            "msg": msg,
            "error": is_error,
            "ts": ts,
            "running": previous.get("running", True),
            "target": previous.get("target", target.isoformat()),
        }
        if extra:
            payload.update(extra)
        status_dict[estacion] = payload
        if notify or is_error:
            send_telegram(tg_token, tg_chat, f"[{ts}] {estacion}: {msg}")
        print(f"[{ts}] {estacion}: {msg}", flush=True)

    # Solo usamos este primer análisis para saber si hay CSV/fechas. El periodo
    # oficial se consulta siempre a SIAR antes de construir el plan definitivo.
    deltas_requested = analyze_station(estacion, target)
    new_station = (not deltas_requested["exists"]) or (
        deltas_requested["exists"] and not deltas_requested["inicio"]
    )
    effective_target = target
    fecha_instalacion = None
    fecha_baja = None
    station_info = None

    try:
        token = obtener_token_siar(SIAR_NIF, SIAR_PASSWORD)
    except SiarAuthError as exc:
        update_log(f"🔑 Auth SIAR: {exc}", True, notify=True)
        return

    git_paths = [os.path.relpath(path, REPO_PATH)]
    total_new = 0
    total_refreshed = 0
    chunks_done = 0
    chunks_since_push = 0

    def checkpoint_before_quota_exit(reason):
        """Protege en Git lo ya persistido antes de salir por cuota SIAR."""
        ok, git_msg = push_to_github(gh_token, git_paths)
        if ok:
            update_log(
                f"⏸️ {reason} Progreso local conservado. ☁️ Checkpoint final: {git_msg}. "
                "Continúa otro día.",
                True,
                notify=True,
                extra={
                    "analysis": analyze_station(estacion, effective_target),
                    "objetivo_solicitado": target.isoformat(),
                    "objetivo_efectivo": effective_target.isoformat(),
                    "fecha_instalacion": fecha_instalacion.isoformat() if fecha_instalacion else None,
                    "fecha_baja": fecha_baja.isoformat() if fecha_baja else None,
                },
            )
        else:
            update_log(
                f"⏸️ {reason} Progreso local conservado. ⚠️ Checkpoint Git falló: {git_msg}. "
                "El CSV local permanece intacto; continúa otro día.",
                True,
                notify=True,
                extra={
                    "analysis": analyze_station(estacion, effective_target),
                    "objetivo_solicitado": target.isoformat(),
                    "objetivo_efectivo": effective_target.isoformat(),
                    "fecha_instalacion": fecha_instalacion.isoformat() if fecha_instalacion else None,
                    "fecha_baja": fecha_baja.isoformat() if fecha_baja else None,
                },
            )
        return ok, git_msg

    # v2.3.2: el periodo oficial se consulta SIEMPRE, también para CSV existentes.
    update_log(
        "📍 Consultando Info/ESTACIONES para validar el periodo oficial de la estación.",
        notify=new_station,
    )
    try:
        info_result = obtener_estacion_info(token, estacion)
    except SiarDailyLimitError as exc:
        checkpoint_before_quota_exit(str(exc))
        return

    if isinstance(info_result, dict) and "error_token" in info_result:
        update_log("🔑 Token rechazado al consultar ESTACIONES; renovando…")
        try:
            token = obtener_token_siar(SIAR_NIF, SIAR_PASSWORD)
        except SiarAuthError as exc:
            update_log(f"⚠️ No se pudo renovar token: {exc}", True, notify=True)
            return
        try:
            info_result = obtener_estacion_info(token, estacion)
        except SiarDailyLimitError as exc:
            checkpoint_before_quota_exit(str(exc))
            return

    if isinstance(info_result, dict):
        if "error_rate_limit" in info_result:
            checkpoint_before_quota_exit(
                f"Cuota/límite SIAR al consultar estaciones: {info_result['error_rate_limit'][:250]}."
            )
            return
        if "error" in info_result:
            # Política conservadora: sin periodo oficial fiable no continuamos.
            update_log(f"⚠️ Info/ESTACIONES: {info_result['error']}", True, notify=True)
            return

    station_info = info_result
    fecha_instalacion = station_info["fecha_instalacion"]
    fecha_baja = station_info.get("fecha_baja")
    if fecha_baja and effective_target > fecha_baja:
        effective_target = fecha_baja
        update_log(
            f"ℹ️ La estación tiene Fecha_Baja {formatear_fecha_es(fecha_baja)}; "
            f"el objetivo efectivo se limita a {formatear_fecha_es(effective_target)}.",
            notify=True,
        )

    # A partir de aquí, todo análisis/plan usa el objetivo efectivo oficial.
    deltas = analyze_station(estacion, effective_target)

    if new_station:
        if not os.path.exists(path):
            try:
                _create_empty_canonical_csv(path)
            except (OSError, pd.errors.ParserError, SiarDataError, ValueError) as exc:
                update_log(f"⚠️ No se pudo crear el CSV canónico: {exc}", True, notify=True)
                return
            update_log(
                f"💾 CSV creado con el esquema canónico de {len(CANONICAL_COLUMNS)} columnas.",
                extra={
                    "analysis": analyze_station(estacion, effective_target),
                    "new_station": True,
                    "fecha_instalacion": fecha_instalacion.isoformat(),
                    "fecha_baja": fecha_baja.isoformat() if fecha_baja else None,
                    "objetivo_solicitado": target.isoformat(),
                    "objetivo_efectivo": effective_target.isoformat(),
                },
            )
        else:
            # Corrige únicamente un CSV vacío: no sobrescribe un CSV con datos.
            try:
                existing_df, existing_columns = _read_existing_csv(path)
            except (OSError, pd.errors.ParserError, SiarDataError, ValueError) as exc:
                update_log(f"⚠️ Error leyendo CSV existente: {exc}", True, notify=True)
                return
            if existing_df.empty and existing_columns != CANONICAL_COLUMNS:
                update_log(
                    "⚠️ El CSV vacío existente no usa el esquema canónico; no se sobrescribe automáticamente.",
                    True,
                    notify=True,
                )
                return

        deltas = analyze_station(estacion, effective_target)
        update_log(
            f"📌 Periodo SIAR de la estación: {formatear_fecha_es(fecha_instalacion)}"
            + (f" → {formatear_fecha_es(fecha_baja)}" if fecha_baja else " → activa")
            + f". Objetivo efectivo: {formatear_fecha_es(effective_target)}.",
            extra={
                "analysis": deltas,
                "new_station": True,
                "fecha_instalacion": fecha_instalacion.isoformat(),
                "fecha_baja": fecha_baja.isoformat() if fecha_baja else None,
                "objetivo_solicitado": target.isoformat(),
                "objetivo_efectivo": effective_target.isoformat(),
            },
        )

        if effective_target < fecha_instalacion:
            update_log(
                f"ℹ️ El objetivo {formatear_fecha_es(effective_target)} es anterior a la instalación "
                f"({formatear_fecha_es(fecha_instalacion)}). No existe un periodo descargable.",
                extra={
                    "analysis": deltas,
                    "new_station": True,
                    "complete": True,
                    "objetivo_solicitado": target.isoformat(),
                    "objetivo_efectivo": effective_target.isoformat(),
                },
            )
            ok, msg = push_to_github(gh_token, git_paths)
            if not ok:
                update_log(
                    f"❌ CSV creado localmente pero GitHub no pudo actualizarse: {msg}.",
                    True,
                    notify=True,
                    extra={"analysis": deltas, "complete": False},
                )
            else:
                update_log(
                    f"🏁 CSV creado y registrado en GitHub. No había fechas descargables antes de la instalación. ✅ {msg}",
                    notify=True,
                    extra={"analysis": deltas, "complete": True},
                )
            return
    else:
        if not deltas["inicio"]:
            update_log("⚠️ El CSV no contiene fechas válidas.", True, notify=True, extra={"analysis": deltas})
            return
        update_log(
            f"🔄 Objetivo solicitado {formatear_fecha_es(target)}; objetivo efectivo {formatear_fecha_es(effective_target)}. "
            f"Cobertura actual {formatear_fecha_es(deltas['inicio'])} → {formatear_fecha_es(deltas['fin'])}. "
            f"Pendientes: {deltas.get('pendientes', 0)} días.",
            notify=True,
            extra={
                "analysis": deltas,
                "fecha_instalacion": fecha_instalacion.isoformat(),
                "fecha_baja": fecha_baja.isoformat() if fecha_baja else None,
                "objetivo_solicitado": target.isoformat(),
                "objetivo_efectivo": effective_target.isoformat(),
            },
        )

    try:
        # v2.3.2: el plan siempre empieza en la Fecha_Instalacion oficial. Los
        # datos históricos anteriores no se borran automáticamente, pero tampoco
        # generan peticiones ni pendientes artificiales.
        fechas = _read_dates(path)
        if fecha_instalacion is not None:
            start_date = fecha_instalacion
        elif fechas:
            start_date = min(fechas)
        else:
            update_log(
                "⚠️ No se encontraron fechas válidas en el CSV y no hay fecha de instalación disponible.",
                True,
                notify=True,
            )
            return

        pending_ranges = _missing_ranges(fechas, start_date, effective_target)
        pending_chunks = list(_make_chunks(pending_ranges))
        update_log(
            f"📋 Plan: {len(pending_ranges)} rango(s) pendiente(s), {len(pending_chunks)} bloque(s) de hasta {CHUNK_DAYS} días.",
            extra={
                "analysis": analyze_station(estacion, effective_target),
                "blocks_total": len(pending_chunks),
                "blocks_done": 0,
                "new_station": new_station,
                "objetivo_solicitado": target.isoformat(),
                "objetivo_efectivo": effective_target.isoformat(),
                "fecha_instalacion": fecha_instalacion.isoformat() if fecha_instalacion else None,
                "fecha_baja": fecha_baja.isoformat() if fecha_baja else None,
            },
        )

        for current, chunk_end in pending_chunks:
            # Puede haber cambiado el archivo mientras esta tarea esperaba. Si el bloque ya está cubierto, saltamos.
            latest_dates = _read_dates(path)
            block_dates = set(
                current + timedelta(days=i)
                for i in range((chunk_end - current).days + 1)
            )
            if block_dates.issubset(latest_dates):
                chunks_done += 1
                update_log(
                    f"⏭️ Bloque ya cubierto localmente {formatear_fecha_es(current)} → {formatear_fecha_es(chunk_end)}; se omite.",
                    extra={
                        "analysis": analyze_station(estacion, effective_target),
                        "blocks_done": chunks_done,
                        "blocks_total": len(pending_chunks),
                    },
                )
                continue

            ini = current.isoformat()
            fin = chunk_end.isoformat()
            update_log(f"📡 SIAR {formatear_fecha_es(current)} → {formatear_fecha_es(chunk_end)}")

            refresh_attempts = 0
            while True:
                datos = fetch_data_rango(estacion, ini, fin, token)
                if isinstance(datos, dict) and "error_token" in datos:
                    refresh_attempts += 1
                    if refresh_attempts > 2:
                        update_log(
                            f"❌ SIAR sigue rechazando el token para {formatear_fecha_es(current)} → {formatear_fecha_es(chunk_end)}. "
                            "Se detiene para no repetir la misma petición.",
                            True,
                        )
                        return
                    update_log("🔑 Token rechazado por SIAR; renovando automáticamente…")
                    try:
                        token = obtener_token_siar(SIAR_NIF, SIAR_PASSWORD)
                    except SiarAuthError as exc:
                        update_log(f"⚠️ No se pudo renovar token: {exc}", True, notify=True)
                        return
                    continue
                break

            if isinstance(datos, dict):
                if "error_rate_limit" in datos:
                    checkpoint_before_quota_exit(
                        f"Cuota/límite SIAR: {datos['error_rate_limit'][:250]}."
                    )
                    return
                update_log(f"⚠️ Error API: {datos.get('error', str(datos))}", True, notify=True)
                return

            expected_days = (chunk_end - current).days + 1
            returned_dates = set()
            for row in datos:
                try:
                    returned_dates.add(normalizar_fecha(str(row["Fecha"]).split("T", 1)[0]))
                except Exception:
                    pass
            missing_in_response = expected_days - len(returned_dates)

            try:
                new_count, last_date, _ = merge_and_save(path, datos, estacion, replace_existing=False)
            except (OSError, pd.errors.ParserError, SiarDataError, ValueError) as exc:
                update_log(f"⚠️ Error guardando {ini} → {fin}: {exc}", True, notify=True)
                return

            total_new += new_count
            chunks_done += 1
            chunks_since_push += 1
            if datos:
                msg = f"💾 {len(datos)} SIAR / {new_count} nuevas. CSV hasta {last_date or 'sin cambios'}."
                if missing_in_response > 0:
                    msg += f" ⚠️ Respuesta parcial: faltan {missing_in_response} día(s) del bloque; quedarán pendientes para futuros intentos."
            else:
                msg = "💾 SIAR devolvió 0 registros para este bloque; el rango queda como hueco para futuros intentos."
            update_log(
                msg + f" Progreso: {chunks_done}/{len(pending_chunks)} bloques.",
                extra={
                    "analysis": analyze_station(estacion, effective_target),
                    "blocks_done": chunks_done,
                    "blocks_total": len(pending_chunks),
                    "response_days": len(returned_dates),
                    "expected_days": expected_days,
                },
            )

            if chunks_since_push >= PUSH_EVERY_CHUNKS:
                ok, msg = push_to_github(gh_token, git_paths)
                if not ok:
                    update_log(
                        f"⚠️ Git: {msg}. Datos locales conservados; se reintentará en próxima ejecución.",
                        True,
                        notify=True,
                    )
                    return
                update_log(f"☁️ Checkpoint GitHub tras {chunks_done} bloques: {msg}")
                chunks_since_push = 0

            if API_PAUSE_SECONDS:
                time.sleep(API_PAUSE_SECONDS)

        # Refresco de los últimos días para incorporar correcciones ya publicadas en SIAR.
        fechas = _read_dates(path)
        if fechas and REFRESH_RECENT_DAYS > 0 and effective_target >= min(fechas):
            refresh_start = max(min(fechas), effective_target - timedelta(days=REFRESH_RECENT_DAYS - 1))
            refresh_ranges = [(refresh_start, effective_target)] if refresh_start <= effective_target else []
            refresh_chunks = list(_make_chunks(refresh_ranges))
            if refresh_chunks:
                update_log(
                    f"🔁 Refresco de {formatear_fecha_es(refresh_start)} → {formatear_fecha_es(effective_target)} "
                    f"({len(refresh_chunks)} bloques) para captar correcciones de SIAR.",
                    extra={"analysis": analyze_station(estacion, effective_target)},
                )
                for current, chunk_end in refresh_chunks:
                    ini = current.isoformat()
                    fin = chunk_end.isoformat()
                    refresh_attempts = 0
                    while True:
                        datos = fetch_data_rango(estacion, ini, fin, token)
                        if isinstance(datos, dict) and "error_token" in datos:
                            refresh_attempts += 1
                            if refresh_attempts > 2:
                                update_log("❌ Token rechazado repetidamente durante el refresco; se detiene.", True, notify=True)
                                return
                            update_log("🔑 Token de refresco rechazado; renovando…")
                            try:
                                token = obtener_token_siar(SIAR_NIF, SIAR_PASSWORD)
                            except SiarAuthError as exc:
                                update_log(f"⚠️ No se pudo renovar token de refresco: {exc}", True, notify=True)
                                return
                            continue
                        break

                    if isinstance(datos, dict):
                        if "error_rate_limit" in datos:
                            checkpoint_before_quota_exit(
                                f"Refresco detenido por cuota SIAR: {datos['error_rate_limit'][:250]}."
                            )
                            return
                        update_log(f"⚠️ Error en refresco: {datos.get('error', str(datos))}", True, notify=True)
                        return
                    try:
                        changed, _, _ = merge_and_save(path, datos, estacion, replace_existing=True)
                    except (OSError, pd.errors.ParserError, SiarDataError, ValueError) as exc:
                        update_log(f"⚠️ Error guardando refresco: {exc}", True, notify=True)
                        return
                    total_refreshed += changed
                    chunks_since_push += 1
                    update_log(
                        f"🔁 Refresco {formatear_fecha_es(current)} → {formatear_fecha_es(chunk_end)}: {changed} filas actualizadas/insertadas.",
                        extra={"analysis": analyze_station(estacion, effective_target)},
                    )
                    if chunks_since_push >= PUSH_EVERY_CHUNKS:
                        ok, msg = push_to_github(gh_token, git_paths)
                        if not ok:
                            update_log(f"⚠️ Git tras refresco: {msg}. Datos locales conservados.", True, notify=True)
                            return
                        update_log(f"☁️ Checkpoint GitHub durante refresco: {msg}")
                        chunks_since_push = 0
                    if API_PAUSE_SECONDS:
                        time.sleep(API_PAUSE_SECONDS)

    except SiarDailyLimitError as exc:
        checkpoint_before_quota_exit(str(exc))
        return

    ok, msg = push_to_github(gh_token, git_paths)
    final_analysis = analyze_station(estacion, effective_target)
    extra_final = {
        "analysis": final_analysis,
        "objetivo_solicitado": target.isoformat(),
        "objetivo_efectivo": effective_target.isoformat(),
        "fecha_instalacion": fecha_instalacion.isoformat() if fecha_instalacion else None,
        "fecha_baja": fecha_baja.isoformat() if fecha_baja else None,
        "new_station": new_station,
    }
    if ok:
        if final_analysis.get("pendientes", 0) == 0:
            update_log(
                f"🏁 Completado hasta {formatear_fecha_es(effective_target)}. ➕ {total_new} nuevas, 🔁 {total_refreshed} refrescadas. ✅ {msg}",
                notify=True,
                extra={**extra_final, "complete": True},
            )
        else:
            update_log(
                f"🏁 Proceso terminado con {final_analysis.get('pendientes')} día(s) aún pendiente(s) o sin dato. "
                f"➕ {total_new} nuevas, 🔁 {total_refreshed} refrescadas. ✅ {msg}",
                notify=True,
                extra={**extra_final, "complete": False},
            )
    else:
        update_log(
            f"🏁 Descarga local completada. ➕ {total_new} nuevas, 🔁 {total_refreshed} refrescadas. ❌ Git: {msg}. "
            "Los CSV permanecen guardados.",
            True,
            notify=True,
            extra={**extra_final, "complete": False},
        )
