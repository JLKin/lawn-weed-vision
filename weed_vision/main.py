"""FastAPI application factory; one model instance and serialized inference per worker."""

from collections.abc import Callable
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, JSONResponse

from . import __version__
from .config import Settings
from .errors import APIError, error_body
from .images import decode_image
from .middleware import RequestContextMiddleware
from .schemas import ErrorResponse, HealthResponse, SegmentationResponse
from .segmenter import Segmenter, create_segmenter
from .mapping_api import mapping_router

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    segmenter_factory: Callable[[Settings], Segmenter] | None = None,
) -> FastAPI:
    config = settings if settings is not None else Settings()
    factory = segmenter_factory or create_segmenter

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.segmenter = None
        application.state.inference_lock = Lock()
        try:
            application.state.segmenter = factory(config)
            await run_in_threadpool(application.state.segmenter.load)
        except Exception:
            # Remain available for /health diagnostics, never return a fake success.
            logger.exception("Model startup failed; check model path, dependencies, and device.")
            application.state.segmenter = None
        yield
        application.state.segmenter = None

    application = FastAPI(
        title="智能除草机视觉模型接口",
        version=__version__,
        description=(
            "YOLOv5 实例分割服务。默认 COCO 通用模型仅用于部署测试，尚未针对草坪和杂草训练。"
            "完成标定前，ground_position_m 始终为空。"
        ),
        docs_url="/api-docs",
        lifespan=lifespan,
        redirect_slashes=False,
    )
    application.add_middleware(RequestContextMiddleware, max_request_bytes=config.max_request_bytes)
    web_dir = Path(__file__).with_name("web")
    application.mount("/static", StaticFiles(directory=web_dir), name="static")
    application.include_router(mapping_router(config))

    @application.get("/docs", include_in_schema=False)
    def chinese_test_page() -> FileResponse:
        return FileResponse(web_dir / "index.html", headers={"Cache-Control": "no-cache"})

    @application.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        return JSONResponse(
            error_body(request.state.request_id, exc.code, exc.message),
            status_code=exc.status_code,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        details = [
            {"loc": list(error["loc"]), "type": error["type"], "message": error["msg"]}
            for error in exc.errors()
        ]
        return JSONResponse(
            error_body(request.state.request_id, "validation_error", "Invalid request fields.", details),
            status_code=422,
        )

    @application.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        # Parser errors can contain input data, so use a stable public description.
        messages = {400: "Malformed request.", 404: "Endpoint not found.", 405: "Method not allowed."}
        return JSONResponse(
            error_body(request.state.request_id, "http_error", messages.get(exc.status_code, "Request failed.")),
            status_code=exc.status_code,
            headers=exc.headers,
        )

    @application.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        request_id = request.state.request_id
        logger.error("Unhandled request failure request_id=%s", request_id, exc_info=exc)
        return JSONResponse(
            error_body(request_id, "internal_error", "An internal error occurred; check server logs using request_id."),
            status_code=500,
            headers={"X-Request-ID": request_id},
        )

    @application.get(
        "/health", summary="检查模型状态", tags=["模型状态"],
        response_model=HealthResponse, responses={503: {"model": HealthResponse, "description": "模型未就绪"}},
    )
    def health(request: Request, response: Response) -> HealthResponse:
        segmenter = getattr(request.app.state, "segmenter", None)
        ready = segmenter is not None and segmenter.ready
        response.status_code = 200 if ready else 503
        return HealthResponse(
            status="ok" if ready else "unavailable",
            model_ready=ready,
            model_version=segmenter.model_version if segmenter else config.model_version or Path(config.model_path).name,
            model_path=Path(config.model_path).name,
            detail=None if ready else "Model is unavailable. Check startup logs and restart after correcting configuration.",
        )

    @application.post(
        "/v1/weed-segmentation",
        summary="上传图片并进行实例分割",
        tags=["图像识别"],
        response_model=SegmentationResponse,
        responses={code: {"model": ErrorResponse} for code in (400, 413, 415, 422, 500, 503)},
    )
    def weed_segmentation(
        request: Request,
        file: Annotated[UploadFile, File(description="待识别图片，支持 JPEG、PNG 或 WebP。")],
        camera_id: Annotated[
            str, Form(
                min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$",
                description="相机编号：以英文字母或数字开头，其余可使用字母、数字、下划线、点、冒号和连字符。",
            )
        ],
        capture_timestamp_ns: Annotated[
            int | None, Form(ge=0, le=2**63 - 1, description="可选的采集时间戳，单位为纳秒；省略时返回空值。")
        ] = None,
    ) -> SegmentationResponse:
        # A synchronous route runs in FastAPI's thread pool; health stays responsive.
        try:
            data = file.file.read(config.max_upload_bytes + 1)
        finally:
            file.file.close()
        image = decode_image(data, file.content_type, config)
        try:
            segmenter = request.app.state.segmenter
            if segmenter is None or not segmenter.ready:
                raise APIError(503, "model_unavailable", "The segmentation model is unavailable. Check /health.")
            try:
                with request.app.state.inference_lock:
                    # Excludes image decoding and lock wait; includes model preprocessing,
                    # inference, CPU synchronization, and conversion to response objects.
                    start = perf_counter()
                    detections = segmenter.predict(image)
                    inference_ms = (perf_counter() - start) * 1000
            except Exception as exc:
                logger.exception("Inference failed request_id=%s", request.state.request_id)
                raise APIError(500, "inference_failed", "Model inference failed; check server logs using request_id.") from exc
            return SegmentationResponse(
                request_id=request.state.request_id,
                camera_id=camera_id,
                capture_timestamp_ns=capture_timestamp_ns,
                model_version=segmenter.model_version,
                image_width=image.width,
                image_height=image.height,
                inference_ms=round(inference_ms, 3),
                detections=detections,
            )
        finally:
            image.close()

    return application


app = create_app()
