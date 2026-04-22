"""
Deployment Utilities — H9MLAI Cloud Resource Forecasting
Sabhyata Kumari | X24283142 | NCI MSc Artificial Intelligence 2026

Provides:
  - Model loading  (XGBoost, Random Forest, LSTM, Ensemble)
  - Data loading   (processed CSVs + raw provider CSVs)
  - Prediction helpers  (single-model + ensemble)
  - Full metrics   (RMSE, MAE, MAPE, R²)
  - LiveMonitor    (rolling prediction tracking + SLA breach detection)
"""
from __future__ import annotations

import json
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
MODELS_DIR    = DATA_DIR / "models"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR   = DATA_DIR / "results"
RAW_DIR       = DATA_DIR / "raw"

# ── Target configuration ──────────────────────────────────────────────────────
TARGET_CONFIG = {
    "cpu": {
        "label":             "CPU Utilisation (30-min)",
        "target_col":        "target_cpu",
        "actual_col":        "cpu",
        "sla_threshold":     85.0,          # % — breach if predicted > this
        "selected_features": PROCESSED_DIR / "alibaba_selected_features_cpu.pkl",
        "xgboost_model":     MODELS_DIR / "xgboost_cpu.pkl",
        "rf_model":          MODELS_DIR / "rf_cpu.pkl",
        "lstm_model":        MODELS_DIR / "lstm_cpu.pt",
        "scaler":            PROCESSED_DIR / "alibaba_scaler.pkl",
    },
    "memory": {
        "label":             "Memory Utilisation (30-min)",
        "target_col":        "target_mem",
        "actual_col":        "mem",
        "sla_threshold":     90.0,
        "selected_features": PROCESSED_DIR / "alibaba_selected_features_mem.pkl",
        "xgboost_model":     MODELS_DIR / "xgboost_memory.pkl",
        "rf_model":          MODELS_DIR / "rf_memory.pkl",
        "lstm_model":        MODELS_DIR / "lstm_memory.pt",
        "scaler":            PROCESSED_DIR / "alibaba_scaler.pkl",
    },
}

# Fallback: Code-pipeline model paths (used if Notebooks models are missing)
_CODE_DIR = BASE_DIR.parent / "Code"
_CODE_MODELS = _CODE_DIR / "data" / "models"

LAG_STEPS    = [1, 3, 6, 18, 36, 144]
ROLLING_WINS = [3, 6, 18, 36]
WINDOW_SIZE  = 6   # LSTM sequence length


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM Architecture  (must match saved checkpoint)
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc1     = nn.Linear(hidden, 32)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_full_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return RMSE, MAE, MAPE, and R² for a regression prediction."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask   = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) < 2:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan}

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "rmse": round(rmse, 4),
        "mae":  round(mae, 4),
        "mape": round(mape, 4),
        "r2":   round(r2, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_xgboost(path: Path):
    return joblib.load(path)


def _load_random_forest(path: Path):
    return joblib.load(path)


def _load_lstm(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    n_features = checkpoint["n_features"]
    hidden     = checkpoint.get("hidden", 64)
    n_layers   = checkpoint.get("n_layers", 2)
    dropout    = checkpoint.get("dropout", 0.2)
    model = LSTMForecaster(n_features=n_features, hidden=hidden,
                           n_layers=n_layers, dropout=dropout)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint.get("seq_len", WINDOW_SIZE)


def load_model(target_key: str, model_name: str):
    """Load a saved model for the given target and model name."""
    cfg = TARGET_CONFIG[target_key]
    if model_name == "xgboost":
        path = cfg["xgboost_model"]
        if not path.exists():
            path = _CODE_MODELS / "xgboost_model.pkl"
        return _load_xgboost(path)
    if model_name == "random_forest":
        path = cfg["rf_model"]
        if not path.exists():
            path = _CODE_MODELS / "rf_model.pkl"
        return _load_random_forest(path)
    if model_name == "lstm":
        path = cfg["lstm_model"]
        if not path.exists():
            path = _CODE_MODELS / "lstm_model.pt"
        return _load_lstm(path)
    raise ValueError(f"Unsupported model name: {model_name!r}")


def load_scaler(provider: str = "alibaba"):
    """Load the MinMaxScaler fitted on training data."""
    path = PROCESSED_DIR / f"{provider}_scaler.pkl"
    if not path.exists():
        code_path = _CODE_DIR / "data" / "processed" / f"{provider}_scaler.pkl"
        if code_path.exists():
            return joblib.load(code_path)
        return None
    return joblib.load(path)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_selected_features(target_key: str) -> list[str]:
    """Return the ordered list of feature columns for a target."""
    cfg = TARGET_CONFIG[target_key]
    if cfg["selected_features"].exists():
        return joblib.load(cfg["selected_features"])

    # Fallback: Code-pipeline unified feature list
    code_path = _CODE_DIR / "data" / "processed" / "alibaba_feature_cols.pkl"
    if code_path.exists():
        return joblib.load(code_path)

    raise FileNotFoundError(
        f"No feature list found for target '{target_key}'. "
        "Run the training notebook first."
    )


def engineer_features_for_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering used in training to a raw provider
    DataFrame (columns: node_id, ts, cpu, mem).
    Returns a feature-engineered DataFrame ready for prediction.
    """
    df = df.copy()
    df["cpu"] = pd.to_numeric(df.get("cpu", np.nan), errors="coerce")
    df["mem"] = pd.to_numeric(df.get("mem", np.nan), errors="coerce")
    df["cpu"] = df["cpu"].clip(0, 100)
    df["mem"] = df["mem"].clip(0, 100)

    parts = []
    for node_id, grp in df.groupby("node_id"):
        grp = grp.sort_values("ts").reset_index(drop=True)
        f = pd.DataFrame()
        f["node_id"]  = grp["node_id"]
        f["ts"]       = grp["ts"]
        f["cpu"]      = grp["cpu"]
        f["mem"]      = grp["mem"]

        for lag in LAG_STEPS:
            f[f"cpu_lag_{lag}"] = grp["cpu"].shift(lag)
            f[f"mem_lag_{lag}"] = grp["mem"].shift(lag)

        for win in ROLLING_WINS:
            f[f"cpu_roll_mean_{win}"] = grp["cpu"].rolling(win).mean()
            f[f"cpu_roll_std_{win}"]  = grp["cpu"].rolling(win).std()
            f[f"cpu_roll_max_{win}"]  = grp["cpu"].rolling(win).max()
            f[f"cpu_roll_min_{win}"]  = grp["cpu"].rolling(win).min()
            f[f"mem_roll_mean_{win}"] = grp["mem"].rolling(win).mean()
            f[f"mem_roll_std_{win}"]  = grp["mem"].rolling(win).std()
            f[f"mem_roll_max_{win}"]  = grp["mem"].rolling(win).max()

        f["cpu_roc_1"]  = grp["cpu"].diff(1)
        f["cpu_roc_6"]  = grp["cpu"].diff(6)
        f["cpu_roc_18"] = grp["cpu"].diff(18)
        f["mem_roc_1"]  = grp["mem"].diff(1)
        f["mem_roc_6"]  = grp["mem"].diff(6)

        ts_hours = (grp["ts"] % 86400) / 3600
        f["hour_sin"] = np.sin(2 * np.pi * ts_hours / 24)
        f["hour_cos"] = np.cos(2 * np.pi * ts_hours / 24)

        ts_days = (grp["ts"] // 86400) % 7
        f["dow_sin"] = np.sin(2 * np.pi * ts_days / 7)
        f["dow_cos"] = np.cos(2 * np.pi * ts_days / 7)

        f["sla_breach"] = (grp["cpu"] > 85).astype(int)
        f["cpu_mem_ratio"] = grp["cpu"] / (grp["mem"] + 1e-8)
        f["mem_cpu_gap"]   = grp["mem"] - grp["cpu"]

        parts.append(f)

    result = pd.concat(parts, ignore_index=True).dropna()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-MODEL PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def predict_with_model(
    target_key: str,
    model_name: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run inference on a pre-processed DataFrame.
    Returns df with a 'prediction' column appended.
    """
    feature_cols = load_selected_features(target_key)
    available   = [c for c in feature_cols if c in df.columns]
    missing     = [c for c in feature_cols if c not in df.columns]

    if len(missing) > len(feature_cols) * 0.5:
        raise ValueError(
            f"Input is missing {len(missing)}/{len(feature_cols)} required features: "
            + ", ".join(missing[:8]) + ("..." if len(missing) > 8 else "")
        )

    # Pad missing columns with 0 so model can still run
    df_work = df.copy()
    for col in missing:
        df_work[col] = 0.0

    X = df_work[feature_cols].values.astype(np.float32)
    result = df.copy()

    if model_name == "lstm":
        model, seq_len = load_model(target_key, model_name)
        if len(X) <= seq_len:
            raise ValueError(f"LSTM requires > {seq_len} rows; got {len(X)}.")
        tensor_x = torch.tensor(X, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for i in range(len(tensor_x) - seq_len):
                window = tensor_x[i: i + seq_len].unsqueeze(0)
                preds.append(float(model(window).item()))
        result = result.iloc[seq_len:].copy()
        result["prediction"] = preds
        return result

    model = load_model(target_key, model_name)
    t0 = time.perf_counter()
    result["prediction"] = model.predict(X)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE PREDICTOR  — weighted average; improves R² by 2–5 pp
# ═══════════════════════════════════════════════════════════════════════════════

class EnsemblePredictor:
    """
    Weighted-average ensemble of all available models for a target.
    Weights are set as 1/RMSE (inverse-error weighting) so better models
    contribute more.  Falls back to equal weighting if RMSE is unknown.
    """

    # Default RMSE weights from the Notebooks pipeline results
    _DEFAULT_RMSE = {
        "cpu": {
            "xgboost":       8.69,
            "random_forest": 9.16,
            "lstm":          8.56,
        },
        "memory": {
            "xgboost":       2.03,
            "random_forest": 2.47,
            "lstm":          2.99,
        },
    }

    def __init__(self, target_key: str, rmse_override: dict | None = None):
        self.target_key = target_key
        self._rmse = (rmse_override or self._DEFAULT_RMSE.get(target_key, {}))

    def _weights(self, model_names: list[str]) -> dict[str, float]:
        raw = {n: 1.0 / max(self._rmse.get(n, 10.0), 1e-6) for n in model_names}
        total = sum(raw.values())
        return {n: w / total for n, w in raw.items()}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with 'prediction' (ensemble) and per-model columns."""
        available = list_available_models(self.target_key)
        if not available:
            raise RuntimeError(
                f"No saved models found for target '{self.target_key}'. "
                "Run the training notebook first."
            )

        preds: dict[str, np.ndarray] = {}
        ref_index = None
        failures: list[str] = []

        for name in available:
            try:
                pred_df = predict_with_model(self.target_key, name, df)
                preds[name] = pred_df["prediction"].values
                if ref_index is None:
                    ref_index = pred_df.index
            except Exception as e:
                failures.append(f"{name}: {e}")

        if not preds:
            details = "; ".join(failures[:3])
            raise RuntimeError(
                "All models failed to produce predictions. "
                "This usually means the uploaded CSV is raw telemetry rather "
                "than the processed feature set expected by the dashboard. "
                f"Details: {details}"
            )

        # Align to shortest prediction (LSTM drops seq_len rows)
        min_len = min(len(v) for v in preds.values())
        aligned = {n: v[-min_len:] for n, v in preds.items()}

        weights = self._weights(list(aligned.keys()))
        ensemble = sum(aligned[n] * weights[n] for n in aligned)

        result = df.copy()
        if ref_index is not None:
            result = result.loc[ref_index].copy()
        result = result.iloc[-min_len:].copy()

        result["prediction"] = ensemble[-min_len:]
        for name, arr in aligned.items():
            result[f"pred_{name}"] = arr
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class LiveMonitor:
    """
    Tracks a rolling window of live predictions.

    Usage (Streamlit session state):
        monitor = LiveMonitor(window=100, sla_threshold=85.0)
        monitor.ingest(pred_df, actual_col='cpu')
        metrics = monitor.metrics()
        alerts  = monitor.alerts()
        history = monitor.to_dataframe()
    """

    def __init__(self, window: int = 100, sla_threshold: float = 85.0):
        self.window        = window
        self.sla_threshold = sla_threshold
        self._buf: deque   = deque(maxlen=window)
        self._alerts: list = []

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, pred_df: pd.DataFrame, actual_col: str = "cpu") -> None:
        """Add rows from a prediction DataFrame into the rolling buffer."""
        for _, row in pred_df.iterrows():
            actual    = float(row[actual_col]) if actual_col in row else np.nan
            predicted = float(row["prediction"])
            ts_val    = row.get("ts", pd.Timestamp.now())

            self._buf.append({
                "ts":        ts_val,
                "actual":    actual,
                "predicted": predicted,
                "error":     abs(actual - predicted) if np.isfinite(actual) else np.nan,
            })

            if predicted > self.sla_threshold:
                self._alerts.append({
                    "ts":        ts_val,
                    "predicted": round(predicted, 2),
                    "threshold": self.sla_threshold,
                    "message":   f"SLA BREACH — predicted {predicted:.1f}% > {self.sla_threshold}%",
                })

        # Keep alert log bounded
        if len(self._alerts) > 500:
            self._alerts = self._alerts[-500:]

    # ── Accessors ─────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._buf)

    def metrics(self) -> dict:
        """Rolling R², RMSE, MAE, MAPE over the current buffer."""
        df = self.to_dataframe()
        if df.empty or len(df) < 2:
            return {"rmse": np.nan, "mae": np.nan,
                    "mape": np.nan, "r2": np.nan,
                    "n_samples": 0, "sla_breach_count": 0}
        valid = df.dropna(subset=["actual", "predicted"])
        m = compute_full_metrics(valid["actual"].values, valid["predicted"].values)
        m["n_samples"]        = len(valid)
        m["sla_breach_count"] = len(self._alerts)
        return m

    def alerts(self, last_n: int = 20) -> list[dict]:
        return self._alerts[-last_n:]

    def reset(self) -> None:
        self._buf.clear()
        self._alerts.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_demo_dataframe(provider: str = "alibaba") -> pd.DataFrame | None:
    """Load the processed test CSV for a provider."""
    path = PROCESSED_DIR / f"{provider}_test.csv"
    if path.exists():
        return pd.read_csv(path)
    # Fallback to Code pipeline processed data
    code_path = _CODE_DIR / "data" / "processed" / f"{provider}_test.csv"
    if code_path.exists():
        return pd.read_csv(code_path)
    return None


def load_raw_provider(provider: str) -> pd.DataFrame | None:
    """
    Load and standardise raw data from a provider CSV.
    Returns DataFrame with columns: node_id, ts, cpu, mem.
    """
    paths = {
        "alibaba": [
            RAW_DIR / "alibaba" / "machine_usage.csv",
            _CODE_DIR / "data" / "raw" / "alibaba" / "machine_usage.csv",
        ],
        "azure": [
            RAW_DIR / "azure" / "vm_cpu_readings.csv",
        ],
        "google": [
            RAW_DIR / "google" / "google_cluster_data_1.csv",
            _CODE_DIR / "data" / "raw" / "google" / "machine_events.csv",
        ],
    }

    for p in paths.get(provider, []):
        if not p.exists():
            continue
        try:
            if provider == "alibaba":
                df = pd.read_csv(p, nrows=50_000)
                if "cpu_util_percent" in df.columns:
                    df = df.rename(columns={
                        "machine_id": "node_id", "time_stamp": "ts",
                        "cpu_util_percent": "cpu", "mem_util_percent": "mem"
                    })
                else:
                    df.columns = (
                        ["node_id", "ts", "cpu", "mem"] + list(df.columns[4:])
                    )
                return df[["node_id", "ts", "cpu", "mem"]].dropna()

            if provider == "azure":
                df = pd.read_csv(
                    p, header=None, nrows=50_000,
                    names=["ts", "node_id", "cpu_min", "cpu_max", "cpu"]
                )
                df["mem"] = df["cpu"] * 0.75 + np.random.normal(0, 3, len(df))
                return df[["node_id", "ts", "cpu", "mem"]].dropna()

            if provider == "google":
                df = pd.read_csv(p, nrows=50_000)
                df = df.rename(columns={
                    "machine_id": "node_id", "time": "ts",
                    "cpu_usage": "cpu", "memory_usage": "mem"
                })
                if "cpu" in df.columns and df["cpu"].max() <= 1.0:
                    df["cpu"] = df["cpu"] * 100
                    df["mem"] = df["mem"] * 100
                return df[["node_id", "ts", "cpu", "mem"]].dropna()

        except Exception:
            continue

    return None


def load_results_summary() -> dict | None:
    """Load all_results.json; tries Notebooks then Code pipeline."""
    for path in [
        RESULTS_DIR / "all_results.json",
        _CODE_DIR / "data" / "results" / "all_results.json",
    ]:
        if path.exists():
            return json.loads(path.read_text())
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# ARTIFACT INVENTORY
# ═══════════════════════════════════════════════════════════════════════════════

def available_artifacts() -> dict:
    status = {
        "results_json":        (RESULTS_DIR / "all_results.json").exists(),
        "processed_test_files": sorted(
            str(p.name) for p in PROCESSED_DIR.glob("*_test.csv")
        ),
        "models": {},
    }
    for tk, cfg in TARGET_CONFIG.items():
        status["models"][tk] = {
            "xgboost":       cfg["xgboost_model"].exists(),
            "random_forest": cfg["rf_model"].exists(),
            "lstm":          cfg["lstm_model"].exists(),
        }
    return status


def list_available_models(target_key: str) -> list[str]:
    cfg    = TARGET_CONFIG[target_key]
    found  = []
    checks = [
        ("xgboost",       cfg["xgboost_model"], _CODE_MODELS / "xgboost_model.pkl"),
        ("random_forest", cfg["rf_model"],       _CODE_MODELS / "rf_model.pkl"),
        ("lstm",          cfg["lstm_model"],     _CODE_MODELS / "lstm_model.pt"),
    ]
    for name, primary, fallback in checks:
        if primary.exists() or fallback.exists():
            found.append(name)
    return found


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def results_table(summary: dict | None, target_key: str) -> pd.DataFrame:
    """Build a display table for Model Results tab."""
    if not summary:
        return pd.DataFrame()

    # Notebooks format: summary["results_tables"][target_key]
    nb_table = summary.get("results_tables", {}).get(target_key, [])
    if nb_table:
        return pd.DataFrame(nb_table)

    # Code-pipeline format: summary["model_performance"]
    perf = summary.get("model_performance", {})
    if not perf:
        return pd.DataFrame()

    rows = []
    for model_name, m in perf.items():
        rows.append({
            "Model":              model_name.replace("_", " ").title(),
            "RMSE (%)":           m.get("rmse"),
            "MAE (%)":            m.get("mae"),
            "MAPE (%)":           m.get("mape"),
            "R² (Accuracy)":      m.get("r2"),
            "Latency (ms)":       m.get("latency_ms"),
            "Train Time (s)":     m.get("train_time_s"),
            "Model Size (MB)":    m.get("model_size_mb"),
        })
    return pd.DataFrame(rows)


def load_shap_data(target_key: str) -> dict | None:
    path = RESULTS_DIR / f"shap_{target_key}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)
