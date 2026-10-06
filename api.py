import io
import os
import re
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import ConfigurationError, OperationFailure
from pymongo.server_api import ServerApi

API_PREFIX = "/api/v1"
CODE_PATTERN = re.compile(r"^[A-Za-z0-9_!.,-]{3,24}$")
DEFAULT_MONGO_URI = "mongodb+srv://loktavolterra:RWnm7vIITjOMTuBg@cluster0.oupu0e6.mongodb.net/?appName=Cluster0"


def get_cors_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS")
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8001",
        "http://localhost:8001",
        "*"
    ]


def generate_random_short_string(length: int = 5) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789_!-."
    return "".join(secrets.choice(alphabet) for _ in range(length))


class SimulationCodeRequest(BaseModel):
    simulation_code: str = Field(min_length=3, max_length=24)


class PairingCodeResponse(BaseModel):
    pairing_code: str


class SessionInfoResponse(BaseModel):
    device_id: str
    session_id: str


class SimulationParameters(BaseModel):
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    delta: Optional[float] = None
    eta: Optional[float] = None
    zeta: Optional[float] = None
    epsilon: Optional[float] = None
    omega: Optional[float] = None
    delta_t: Optional[float] = None
    equation: Optional[int] = None
    k1: Optional[float] = None
    k2: Optional[float] = None
    sigma: Optional[float] = None
    with_invader: Optional[bool] = None
    stochastic: Optional[bool] = None
    fire_active: Optional[bool] = None
    fire_intensity: Optional[float] = None


class InitialConditions(BaseModel):
    preys: Optional[float] = None
    predators: Optional[float] = None
    invaders: Optional[float] = None


class SimulationStep(BaseModel):
    step: int
    time: float
    timestamp_utc: Optional[str] = None
    preys_count: Optional[float] = None
    predators_count: Optional[float] = None
    invaders_count: Optional[float] = None
    stopped_sim: bool = False


class SimulationRunSummary(BaseModel):
    sim_id: int
    total_steps: int
    first_timestamp_utc: Optional[str] = None
    last_timestamp_utc: Optional[str] = None
    prey_min: Optional[float] = None
    prey_max: Optional[float] = None
    predator_min: Optional[float] = None
    predator_max: Optional[float] = None
    prey_final: Optional[float] = None
    predator_final: Optional[float] = None


class SimulationRun(BaseModel):
    sim_id: int
    parameters: SimulationParameters
    initial_conditions: InitialConditions
    summary: SimulationRunSummary
    data: List[SimulationStep]


class SimulationResponseSummary(BaseModel):
    total_simulations: int
    total_steps: int
    event_types: List[str]
    first_timestamp_utc: Optional[str] = None
    last_timestamp_utc: Optional[str] = None


class SimulationResponse(BaseModel):
    simulation_code: str
    session: SessionInfoResponse
    summary: SimulationResponseSummary
    simulations: List[SimulationRun]


class UserSessionRequest(BaseModel):
    device_id: str
    session_id: str


app = FastAPI(title="Lotka-Volterra Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

mongo_uri = os.getenv("MONGODB_URI", DEFAULT_MONGO_URI)
mongo_db_name = os.getenv("MONGODB_DB", "LoktaVolterra")
try:
    _mongo_client = MongoClient(mongo_uri, server_api=ServerApi("1"), connect=False)
    db = _mongo_client[mongo_db_name]
except ConfigurationError:
    _mongo_client = None
    db = None


def get_db() -> Any:
    if db is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No se pudo configurar MongoDB. Revisa MONGODB_URI o la conectividad DNS "
                "del cluster MongoDB Atlas."
            ),
        )
    return db


def _to_builtin(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def validate_code_format(code: str) -> str:
    normalized = code.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="El codigo no puede estar vacio")
    if not CODE_PATTERN.match(normalized):
        raise HTTPException(status_code=400, detail="Formato de codigo invalido")
    return normalized


def get_session_info_by_code(code: str) -> Dict[str, Any]:
    session_info = get_db().temp_codes.find_one({"code": code})
    if not session_info:
        raise HTTPException(status_code=404, detail="Codigo inexistente o expirado")
    if not session_info.get("device_id") or not session_info.get("session_id"):
        raise HTTPException(status_code=422, detail="Codigo sin metadatos de sesion validos")
    return session_info


def get_logs_for_session(session_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    query = {
        "device_id": session_info["device_id"],
        "session_id": session_info["session_id"],
    }
    logs_list = list(get_db().logs.find(query))
    if not logs_list:
        raise HTTPException(status_code=404, detail="Sin datos para ese codigo")
    return logs_list


def build_event_dataframes(logs_list: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    event_types = sorted({log.get("event_type", "Otros") for log in logs_list})
    grouped_frames: Dict[str, pd.DataFrame] = {}
    for event_type in event_types:
        relevant_logs = [log for log in logs_list if log.get("event_type", "Otros") == event_type]
        df_filtered = pd.DataFrame(relevant_logs)
        if "_id" in df_filtered.columns:
            df_filtered = df_filtered.drop(columns=["_id"])
        df_filtered = df_filtered.dropna(axis=1, how="all")
        sheet_name = str(event_type)[:30].replace("/", "_")
        grouped_frames[sheet_name] = df_filtered
    return grouped_frames


def prepare_simulation_data(grouped_frames: Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "Iteration" not in grouped_frames:
        raise HTTPException(status_code=422, detail="No se encontro el evento Iteration")
    if "Parameters" not in grouped_frames:
        raise HTTPException(status_code=422, detail="No se encontro el evento Parameters")

    df_iter = grouped_frames["Iteration"].copy()
    df_params = grouped_frames["Parameters"].copy()

    if df_iter.empty:
        raise HTTPException(status_code=404, detail="Resultado vacio: Iteration sin filas")

    if "stopped_sim" not in df_iter.columns:
        raise HTTPException(status_code=422, detail="Iteration no contiene la columna stopped_sim")

    df_iter["sim_id"] = (df_iter["stopped_sim"].shift(1) == True).cumsum()  # noqa: E712
    if "event_subtype" in df_params.columns:
        df_start_params = df_params[df_params["event_subtype"] == "start"].reset_index(drop=True)
    else:
        df_start_params = pd.DataFrame()
    return df_iter, df_params, df_start_params


def build_steps(run_df: pd.DataFrame) -> List[SimulationStep]:
    timestamps = pd.to_datetime(run_df.get("timestamp_utc"), errors="coerce", utc=True)
    if timestamps.notna().any():
        first_ts = timestamps.dropna().iloc[0]
        elapsed = (timestamps - first_ts).dt.total_seconds().ffill().fillna(0.0)
    else:
        elapsed = pd.Series([float(i) for i in range(len(run_df))])

    steps: List[SimulationStep] = []
    for idx, (_, row) in enumerate(run_df.reset_index(drop=True).iterrows()):
        steps.append(
            SimulationStep(
                step=idx,
                time=float(elapsed.iloc[idx]),
                timestamp_utc=_to_builtin(row.get("timestamp_utc")),
                preys_count=_to_builtin(row.get("preys_count")),
                predators_count=_to_builtin(row.get("predators_count")),
                invaders_count=_to_builtin(row.get("invaders_count")),
                stopped_sim=bool(row.get("stopped_sim", False)),
            )
        )
    return steps


def build_simulation_runs(df_iter: pd.DataFrame, df_start_params: pd.DataFrame) -> List[SimulationRun]:
    runs: List[SimulationRun] = []
    sim_ids = list(df_iter["sim_id"].dropna().unique())

    for run_index, sim_id in enumerate(sim_ids):
        run_df = df_iter[df_iter["sim_id"] == sim_id].reset_index(drop=True)
        if run_df.empty:
            continue

        params_row = {}
        if run_index < len(df_start_params):
            params_row = {k: _to_builtin(v) for k, v in df_start_params.iloc[run_index].to_dict().items()}

        steps = build_steps(run_df)
        prey_series = run_df.get("preys_count", pd.Series(dtype=float))
        predator_series = run_df.get("predators_count", pd.Series(dtype=float))

        run_summary = SimulationRunSummary(
            sim_id=int(sim_id),
            total_steps=len(run_df),
            first_timestamp_utc=_to_builtin(run_df.iloc[0].get("timestamp_utc")),
            last_timestamp_utc=_to_builtin(run_df.iloc[-1].get("timestamp_utc")),
            prey_min=_to_builtin(prey_series.min()) if not prey_series.empty else None,
            prey_max=_to_builtin(prey_series.max()) if not prey_series.empty else None,
            predator_min=_to_builtin(predator_series.min()) if not predator_series.empty else None,
            predator_max=_to_builtin(predator_series.max()) if not predator_series.empty else None,
            prey_final=_to_builtin(prey_series.iloc[-1]) if not prey_series.empty else None,
            predator_final=_to_builtin(predator_series.iloc[-1]) if not predator_series.empty else None,
        )

        run = SimulationRun(
            sim_id=int(sim_id),
            parameters=SimulationParameters(
                alpha=params_row.get("alpha"),
                beta=params_row.get("beta"),
                gamma=params_row.get("gamma"),
                delta=params_row.get("delta"),
                eta=params_row.get("eta"),
                zeta=params_row.get("zeta"),
                epsilon=params_row.get("epsilon"),
                omega=params_row.get("omega"),
                delta_t=params_row.get("delta_t"),
                equation=params_row.get("equation"),
                k1=params_row.get("k1"),
                k2=params_row.get("k2"),
                sigma=params_row.get("sigma"),
                with_invader=params_row.get("with_invader"),
                stochastic=params_row.get("stochastic"),
                fire_active=params_row.get("fire_active"),
                fire_intensity=params_row.get("fire_intensity"),
            ),
            initial_conditions=InitialConditions(
                preys=params_row.get("preys"),
                predators=params_row.get("predators"),
                invaders=params_row.get("invaders"),
            ),
            summary=run_summary,
            data=steps,
        )
        runs.append(run)

    if not runs:
        raise HTTPException(status_code=404, detail="Resultado vacio tras procesar Iteration")
    return runs


def build_simulation_response(code: str) -> SimulationResponse:
    checked_code = validate_code_format(code)
    session_info = get_session_info_by_code(checked_code)
    logs_list = get_logs_for_session(session_info)
    grouped_frames = build_event_dataframes(logs_list)
    df_iter, _, df_start_params = prepare_simulation_data(grouped_frames)
    runs = build_simulation_runs(df_iter, df_start_params)

    summary = SimulationResponseSummary(
        total_simulations=len(runs),
        total_steps=int(len(df_iter)),
        event_types=list(grouped_frames.keys()),
        first_timestamp_utc=_to_builtin(df_iter.iloc[0].get("timestamp_utc")),
        last_timestamp_utc=_to_builtin(df_iter.iloc[-1].get("timestamp_utc")),
    )

    return SimulationResponse(
        simulation_code=checked_code,
        session=SessionInfoResponse(
            device_id=str(session_info["device_id"]),
            session_id=str(session_info["session_id"]),
        ),
        summary=summary,
        simulations=runs,
    )


def build_excel_report(df_iter: pd.DataFrame, df_params: pd.DataFrame, df_start_params: pd.DataFrame) -> io.BytesIO:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_iter.to_excel(writer, index=False, sheet_name="Iteration")
        df_params.to_excel(writer, index=False, sheet_name="Parameters")

        workbook = writer.book
        sheet_it = writer.sheets["Iteration"]

        start_row_excel = 1
        sim_ids = df_iter["sim_id"].unique()

        for i, sim_id in enumerate(sim_ids):
            if i >= len(df_start_params):
                break

            data_sim = df_iter[df_iter["sim_id"] == sim_id]
            end_row_excel = start_row_excel + len(data_sim) - 1

            p = df_start_params.iloc[i]
            a, b, k1, k2, eq = p["alpha"], p["beta"], p["k1"], p["k2"], p["equation"]
            g, d = p["gamma"], p["delta"]

            col_null = 25
            row_null = i * 5

            if eq == 2:
                pred_max1 = k1 / a if a != 0 else k1
                prey_max2 = k2 / b if b != 0 else k2

                sheet_it.write_row(row_null, col_null, ["NC_Prey_X", "NC_Prey_Y", "NC_Pred_X", "NC_Pred_Y"])
                sheet_it.write_row(row_null + 1, col_null, [k1, 0, 0, k2])
                sheet_it.write_row(row_null + 2, col_null, [0, pred_max1, prey_max2, 0])

            chart_evol = workbook.add_chart({"type": "line"})
            chart_evol.add_series(
                {
                    "name": "N2",
                    "categories": ["Iteration", start_row_excel, 2, end_row_excel, 2],
                    "values": ["Iteration", start_row_excel, 4, end_row_excel, 4],
                    "line": {"color": "blue"},
                }
            )
            chart_evol.add_series(
                {
                    "name": "N1",
                    "values": ["Iteration", start_row_excel, 5, end_row_excel, 5],
                    "line": {"color": "red"},
                }
            )

            chart_evol.set_title({"name": f"Evolucion Sim {i + 1} (a:{a}, b:{b}, g:{g}, d:{d})"})
            chart_evol.set_x_axis({"name": "Tiempo"})
            chart_evol.set_y_axis({"name": "Poblacion"})
            sheet_it.insert_chart(f"L{2 + (i * 18)}", chart_evol)

            chart_phase = workbook.add_chart({"type": "scatter", "subtype": "smooth"})
            chart_phase.add_series(
                {
                    "name": "Trayectoria",
                    "categories": ["Iteration", start_row_excel, 5, end_row_excel, 5],
                    "values": ["Iteration", start_row_excel, 4, end_row_excel, 4],
                    "line": {"color": "black", "width": 1.25},
                }
            )
            if eq == 2:
                chart_phase.add_series(
                    {
                        "name": "Isoclina N2",
                        "categories": ["Iteration", row_null + 1, col_null, row_null + 2, col_null],
                        "values": ["Iteration", row_null + 1, col_null + 1, row_null + 2, col_null + 1],
                        "line": {"color": "#32CD32", "dash_type": "dash"},
                    }
                )
                chart_phase.add_series(
                    {
                        "name": "Isoclina N1",
                        "categories": ["Iteration", row_null + 1, col_null + 2, row_null + 2, col_null + 2],
                        "values": ["Iteration", row_null + 1, col_null + 3, row_null + 2, col_null + 3],
                        "line": {"color": "#DC3C3C", "dash_type": "dash"},
                    }
                )

            chart_phase.set_title({"name": f"Plano de Fase Sim {i + 1}"})
            chart_phase.set_x_axis({"name": "N1"})
            chart_phase.set_y_axis({"name": "N2"})
            sheet_it.insert_chart(f"U{2 + (i * 18)}", chart_phase)

            start_row_excel = end_row_excel + 1

        sheet_it.set_column("Z:AC", None, None, {"hidden": True})

    output.seek(0)
    return output


@app.on_event("startup")
def setup_db() -> None:
    if db is None:
        return
    database = get_db()
    try:
        database.temp_codes.create_index("created_at", expireAfterSeconds=300000)
    except OperationFailure as exc:
        if exc.code == 85:
            database.temp_codes.drop_index("created_at_1")
            database.temp_codes.create_index("created_at", expireAfterSeconds=300000)


@app.get("/health")
def health_check() -> Dict[str, str]:
    if db is None:
        return {"status": "degraded"}
    return {"status": "ok"}


@app.post("/download-code", response_model=PairingCodeResponse)
def generate_code(user_data: UserSessionRequest) -> PairingCodeResponse:
    code = generate_random_short_string(5)
    get_db().temp_codes.insert_one(
        {
            "code": code,
            "device_id": user_data.device_id,
            "session_id": user_data.session_id,
            "created_at": datetime.now(timezone.utc),
        }
    )
    return PairingCodeResponse(pairing_code=code)


@app.post("/log_entry")
def log_entry(log_data: Dict[str, Any]) -> Dict[str, str]:
    try:
        result = get_db().logs.insert_one(log_data)
        return {"inserted_id": str(result.inserted_id), "status": "success"}
    except Exception:
        raise HTTPException(status_code=500, detail="Error al insertar en MongoDB")


@app.get(f"{API_PREFIX}/simulations/{{code}}", response_model=SimulationResponse)
def get_simulation_by_code(code: str) -> SimulationResponse:
    try:
        return build_simulation_response(code)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Error durante la simulacion")


@app.post(f"{API_PREFIX}/simulations", response_model=SimulationResponse)
def get_simulation_from_body(request: SimulationCodeRequest) -> SimulationResponse:
    return get_simulation_by_code(request.simulation_code)


@app.get("/download-logs/{code}")
def download_logs_excel(code: str) -> StreamingResponse:
    checked_code = validate_code_format(code)
    session_info = get_session_info_by_code(checked_code)
    logs_list = get_logs_for_session(session_info)
    grouped_frames = build_event_dataframes(logs_list)
    df_iter, df_params, df_start_params = prepare_simulation_data(grouped_frames)
    output = build_excel_report(df_iter, df_params, df_start_params)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=logs_{checked_code}.xlsx"},
    )


@app.get(f"{API_PREFIX}/simulations/{{code}}/excel")
def download_logs_excel_v1(code: str) -> StreamingResponse:
    return download_logs_excel(code)
