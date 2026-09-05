from fastapi import FastAPI

app = FastAPI(title="Locus Voice Gateway V2", docs_url=None, redoc_url=None)


@app.get("/ws/v2/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "realtime", "version": "0.1.0"}
