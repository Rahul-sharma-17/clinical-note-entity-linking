"""Local FastAPI coder review interface."""

from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import BENCHMARK_METRICS_PATH
from src.data_loader import load_dataset
from src.pipeline import load_pipeline

STATIC = Path(__file__).resolve().parent / "static"


class CustomNote(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(application: FastAPI):
    dataset = load_dataset()
    application.state.notes = {
        note_id: dataset.notes[note_id] for note_id in sorted(dataset.splits["validation"])
    }
    application.state.pipeline = load_pipeline(dataset)
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def page() -> FileResponse:
    return FileResponse(
        STATIC / "review.html", media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/notes")
def validation_ids(request: Request) -> JSONResponse:
    return JSONResponse(list(request.app.state.notes), headers={"Cache-Control": "no-store"})


@app.get("/api/metrics")
def benchmark_metrics() -> FileResponse:
    if not BENCHMARK_METRICS_PATH.is_file():
        raise HTTPException(status_code=404, detail="No saved validation benchmark.")
    return FileResponse(
        BENCHMARK_METRICS_PATH, media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


def predict_note(request: Request, text: str, note_id: str | None) -> JSONResponse:
    hits = request.app.state.pipeline.predict(text)
    return JSONResponse(
        {"note_id": note_id, "text": text, "mentions": [asdict(hit) for hit in hits]},
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/note")
def validation_note(request: Request, note_id: str) -> JSONResponse:
    notes = request.app.state.notes
    if note_id not in notes:
        raise HTTPException(status_code=404, detail="Validation note not found.")
    return predict_note(request, notes[note_id], note_id)


@app.post("/api/predict")
def custom_note(request: Request, payload: CustomNote) -> JSONResponse:
    return predict_note(request, payload.text, None)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()

