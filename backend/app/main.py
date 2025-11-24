from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.dataset_router import router as dataset_router
from app.routers.eda_router import router as eda_router
from app.routers.models_router import router as models_router
from app.routers.preprocess_router import router as preprocess_router
from app.routers.report_router import router as report_router
from app.routers.issues_router import router as issues_router

app = FastAPI(title="AutoML Backend")


# CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(dataset_router, prefix="/api/dataset", tags=["Dataset"])
app.include_router(eda_router, prefix="/api/eda", tags=["EDA"])
app.include_router(preprocess_router, prefix="/api/preprocess", tags=["Preprocessing"])
app.include_router(models_router, prefix="/api/models", tags=["Models"])
app.include_router(report_router, prefix="/api/report", tags=["Report"])
app.include_router(issues_router, prefix="/api/issues", tags=["Issues"])


@app.get("/")
def root():
    return {"message": "Welcome to the AutoML Backend!"}