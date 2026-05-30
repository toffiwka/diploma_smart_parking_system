"""
Parking Monitor — Flask Backend с SQLite
=========================================
БД хранит:
  - snapshots : каждое обновление от камеры (история)
  - logs      : события подключений и ошибок
"""

from flask import Flask, request, jsonify, render_template
from datetime import datetime
import sqlite3
import os

app = Flask(__name__)

# ─────────────────────────────────────────────
# Путь к файлу БД (рядом с app.py)
# ─────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "parking.db")


# ─────────────────────────────────────────────
# Метаданные парковок (статичные)
# ─────────────────────────────────────────────
PARKING_META = {
    "parking_1": {"name": "Кумисбекова 4",    "address": "ул. Кумисбекова 4, Астана",    "lat": 51.163706, "lon": 71.401294},
    "parking_2": {"name": "Тәуелсіздік 47А",  "address": "пр. Тәуелсіздік 47А, Астана", "lat": 51.13339,  "lon": 71.465989},
}

# ─────────────────────────────────────────────
# Кэш текущего состояния (обновляется при POST /update)
# При перезапуске подгружается из БД
# ─────────────────────────────────────────────
parking_data = {
    pid: {**meta, "free": 0, "occupied": 0, "total": 0, "last_update": None}
    for pid, meta in PARKING_META.items()
}


# ══════════════════════════════════════════════
# РАБОТА С БД
# ══════════════════════════════════════════════

def get_db():
    """Открыть соединение с БД."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Создать таблицы если их нет."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                parking_id  TEXT    NOT NULL,
                free        INTEGER NOT NULL,
                occupied    INTEGER NOT NULL,
                total       INTEGER NOT NULL,
                recorded_at TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS logs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                level      TEXT NOT NULL,
                message    TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_parking
                ON snapshots(parking_id, recorded_at);
        """)
    print("[DB] База данных инициализирована.")


def log_event(level: str, message: str):
    """Записать событие в таблицу logs."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO logs (level, message, created_at) VALUES (?, ?, ?)",
            (level, message, datetime.now().isoformat())
        )


def load_latest_from_db():
    """
    При старте подгрузить последние значения из БД,
    чтобы после перезапуска не показывались нули.
    """
    with get_db() as conn:
        for pid in parking_data:
            row = conn.execute("""
                SELECT free, occupied, total, recorded_at
                FROM snapshots WHERE parking_id = ?
                ORDER BY id DESC LIMIT 1
            """, (pid,)).fetchone()
            if row:
                parking_data[pid]["free"]        = row["free"]
                parking_data[pid]["occupied"]    = row["occupied"]
                parking_data[pid]["total"]       = row["total"]
                parking_data[pid]["last_update"] = row["recorded_at"][11:19]
                print(f"[DB] Восстановлено {pid}: free={row['free']}, occupied={row['occupied']}")


# ══════════════════════════════════════════════
# МАРШРУТЫ
# ══════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────────
# POST /update — приём данных от OpenCV
# ─────────────────────────────────────────────
@app.route("/update", methods=["POST"])
def update():
    data = request.get_json(silent=True)

    if not data:
        log_event("WARNING", "POST /update — пустое тело запроса")
        return jsonify({"error": "Нет JSON в теле запроса"}), 400

    parking_id = data.get("parking_id")
    free       = data.get("free")
    occupied   = data.get("occupied")

    if parking_id not in parking_data:
        log_event("WARNING", f"Неизвестный parking_id: {parking_id}")
        return jsonify({"error": f"Неизвестный parking_id: {parking_id}"}), 404

    if free is None or occupied is None:
        log_event("WARNING", f"{parking_id} — отсутствуют поля free/occupied")
        return jsonify({"error": "Поля 'free' и 'occupied' обязательны"}), 400

    free, occupied = int(free), int(occupied)
    total = free + occupied
    now   = datetime.now()

    # Обновить кэш
    parking_data[parking_id].update({
        "free":        free,
        "occupied":    occupied,
        "total":       total,
        "last_update": now.strftime("%H:%M:%S")
    })

    # Записать снимок в БД
    with get_db() as conn:
        conn.execute(
            "INSERT INTO snapshots (parking_id, free, occupied, total, recorded_at) VALUES (?, ?, ?, ?, ?)",
            (parking_id, free, occupied, total, now.isoformat())
        )

    log_event("INFO", f"{parking_id}: free={free}, occupied={occupied}")
    print(f"[UPDATE] {parking_id}: free={free}, occupied={occupied}")
    return jsonify({"status": "ok", "parking_id": parking_id}), 200


# ─────────────────────────────────────────────
# GET /status — текущее состояние (для polling)
# ─────────────────────────────────────────────
@app.route("/status")
def status():
    return jsonify(parking_data)


# ─────────────────────────────────────────────
# GET /history?parking_id=parking_1&limit=100
# История снимков для графиков
# ─────────────────────────────────────────────
@app.route("/history")
def history():
    parking_id = request.args.get("parking_id")
    limit      = min(int(request.args.get("limit", 100)), 1000)

    if parking_id and parking_id not in parking_data:
        return jsonify({"error": "Неизвестный parking_id"}), 404

    query  = "SELECT parking_id, free, occupied, total, recorded_at FROM snapshots"
    params = []

    if parking_id:
        query  += " WHERE parking_id = ?"
        params.append(parking_id)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()

    return jsonify([dict(r) for r in reversed(rows)])


# ─────────────────────────────────────────────
# GET /logs?limit=50 — последние события
# ─────────────────────────────────────────────
@app.route("/logs")
def logs():
    limit = min(int(request.args.get("limit", 50)), 500)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT level, message, created_at FROM logs ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return jsonify([dict(r) for r in rows])


# ══════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════
if __name__ == "__main__":
    init_db()
    load_latest_from_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
