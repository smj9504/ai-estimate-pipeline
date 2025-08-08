# src/main.py
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import os
from pathlib import Path

# 프로젝트 내부 모듈들
from src.models.model_interface import ModelOrchestrator
from src.processors.result_merger import ResultMerger
from src.validators.estimation_validator import ComprehensiveValidator
from src.models.data_models import ProjectData
from src.phases.phase_manager import PhaseManager
from src.utils.logger import get_logger, log_error

# Token tracking imports
try:
    from src.api.tracking_endpoints import tracking_router
    from src.tracking.tracked_orchestrator import TrackedModelOrchestrator
    TRACKING_AVAILABLE = True
except ImportError as e:
    TRACKING_AVAILABLE = False
    tracking_router = None
    TrackedModelOrchestrator = None

app = FastAPI(title="Reconstruction Estimator", version="2.0.0")

# Logger 설정
logger = get_logger('main')

# Phase Manager 인스턴스 (전역)
phase_manager = PhaseManager()

# Include tracking router if available
if TRACKING_AVAILABLE and tracking_router:
    app.include_router(tracking_router)
    logger.info("Token tracking endpoints registered")
else:
    logger.warning("Token tracking endpoints not available")

logger.info("FastAPI 서버 초기화 완료")

# 템플릿과 정적 파일 설정
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))

# 정적 파일이 있다면
try:
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")
except Exception:
    pass  # 정적 파일 디렉토리가 없어도 일단 넘어감

class EstimateRequest(BaseModel):
    json_data: Dict[str, Any]
    models_to_use: List[str] = ["gpt4", "claude", "gemini"]

class PhaseRequest(BaseModel):
    session_id: Optional[str] = None
    phase_number: int
    input_data: Optional[Dict[str, Any]] = None
    model_to_use: Optional[str] = "gemini"  # Phase 0용
    models_to_use: Optional[List[str]] = None  # Phase 1-6용
    
class PhaseApprovalRequest(BaseModel):
    session_id: str
    phase_number: int
    approved: bool = True
    modified_data: Optional[Dict[str, Any]] = None

class EstimateResponse(BaseModel):
    success: bool
    merged_result: Dict[str, Any] = None
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = {}
    error_message: str = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """홈페이지 - JSON 업로드 인터페이스"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline(request: Request):
    """Phase 기반 파이프라인 인터페이스"""
    return templates.TemplateResponse("pipeline.html", {"request": request})

@app.get("/usage", response_class=HTMLResponse)
async def usage_dashboard(request: Request):
    """Token usage dashboard interface"""
    if not TRACKING_AVAILABLE:
        # Show simple message if tracking not available
        return JSONResponse({
            "error": "Token tracking system is not available. Please check the installation."
        }, status_code=503)
    
    return templates.TemplateResponse("usage_dashboard.html", {"request": request})

@app.post("/api/estimate/merge", response_model=EstimateResponse)
async def merge_estimates(request: EstimateRequest):
    """
    메인 API - 여러 모델 결과를 병합하여 최종 견적 생성
    """
    try:
        logger.info(f"견적 요청 시작: {request.models_to_use}")
        
        # 1. 프롬프트 로드 (실제 프롬프트는 파일에서 읽어오거나 설정에서 가져옴)
        base_prompt = """You are a Senior Reconstruction Estimating Specialist in the DMV area. Your task is to generate a detailed reconstruction estimate by meticulously analyzing and simultaneously validating the provided complex project data. The input data is a single, consolidated JSON file that already contains all room-specific details.

Instructions:
1. Analyze and Validate Data: Iterate through each room object in the rooms array. Carefully examine the material, work_scope, measurements, demo_scope(already demo'd), and additional_notes data for each room.

2. Define and Apply the "Remove & Replace" Strategy:
   - Trigger: This strategy applies to any material where the work_scope is explicitly set to "Remove & Replace".
   - Already Demolished: The quantities listed in the demo_scope object represent work that is already complete. Do not include any removal cost for these items.
   - Remaining Material Removal: For the material being replaced, any existing material that was not already removed must be explicitly included in the work scope as a removal task.
   - New Installation: The new material must be installed for the entire calculated surface area or perimeter.

3. Process demo_scope: Quantities in demo_scope are considered completed work. While no removal cost is applied, if the work is a surface finish like drywall, installation of new finishes is required.

4. Apply Additional Rules:
   - High Ceiling Premium: If a room's height is over 9 feet, apply a high ceiling premium to wall and ceiling work.
   - Special Tasks: Include all tasks from the additional_notes object.

Please provide your response in a structured format with clear work scope breakdown for each room."""
        
        # 2. 각 모델에 병렬로 프롬프트 전송
        # Use tracked orchestrator if available, otherwise fallback to regular
        if TRACKING_AVAILABLE and TrackedModelOrchestrator:
            orchestrator = TrackedModelOrchestrator(
                enable_tracking=True, 
                phase="estimate_merge"
            )
            logger.info("Using tracked orchestrator for token monitoring")
        else:
            orchestrator = ModelOrchestrator()
            logger.info("Using regular orchestrator (tracking unavailable)")
        
        available_models = orchestrator.get_available_models()
        
        # 요청된 모델 중 사용 가능한 것만 필터링
        models_to_use = [model for model in request.models_to_use if model in available_models]
        
        if not models_to_use:
            return EstimateResponse(
                success=False,
                error_message="요청된 모델들을 사용할 수 없습니다. API 키를 확인하세요."
            )
        
        logger.info(f"사용 가능한 모델: {models_to_use}")
        
        # 모델 병렬 실행 (tracked or regular)
        if TRACKING_AVAILABLE and hasattr(orchestrator, 'run_parallel_tracked'):
            model_results = await orchestrator.run_parallel_tracked(
                prompt=base_prompt,
                json_data=request.json_data,
                model_names=models_to_use
            )
        else:
            model_results = await orchestrator.run_parallel(
                prompt=base_prompt, 
                json_data=request.json_data, 
                model_names=models_to_use
            )
        
        if not model_results:
            return EstimateResponse(
                success=False,
                error_message="모든 모델 호출이 실패했습니다."
            )
        
        logger.info(f"모델 실행 완료: {len(model_results)}개 결과")
        
        # 3. 결과 병합
        merger = ResultMerger()
        merged_result = merger.merge_results(model_results)
        
        logger.info(f"병합 완료: 신뢰도 {merged_result.overall_confidence:.2f}")
        
        # 4. 검증 (선택적)
        try:
            validator = ComprehensiveValidator()
            project_data = ProjectData.from_json_list(request.json_data)
            validation_result = validator.validate_merged_estimate(merged_result, project_data)
            
            # 검증 결과를 메타데이터에 추가
            validation_summary = {
                "validation_score": validation_result.score,
                "validation_issues": len(validation_result.issues),
                "validation_warnings": len(validation_result.warnings),
                "is_valid": validation_result.is_valid
            }
        except Exception as e:
            logger.warning(f"검증 중 오류 (계속 진행): {e}")
            validation_summary = {"validation_error": str(e)}
        
        # 5. 응답 생성
        return EstimateResponse(
            success=True,
            merged_result={
                "project_info": merged_result.project_info,
                "rooms": merged_result.rooms,
                "total_work_items": merged_result.total_work_items,
                "summary_stats": merged_result.summary_stats
            },
            confidence_score=merged_result.overall_confidence,
            metadata={
                "models_used": merged_result.metadata.models_used,
                "processing_time": f"{merged_result.metadata.processing_time_total:.1f}s",
                "consensus_level": merged_result.metadata.consensus_level,
                "confidence_level": merged_result.metadata.confidence_level.value,
                "manual_review_required": merged_result.metadata.manual_review_required,
                "deviation_metrics": merged_result.metadata.deviation_metrics,
                "validation": validation_summary
            }
        )
        
    except Exception as e:
        logger.error(f"견적 처리 중 오류: {e}")
        log_error('main', e, {'models': request.models_to_use})
        import traceback
        traceback.print_exc()
        
        return EstimateResponse(
            success=False,
            error_message=f"처리 중 오류 발생: {str(e)}"
        )

@app.post("/api/pipeline/start")
async def start_pipeline(request: PhaseRequest):
    """
    파이프라인 시작 - Phase 0부터 실행
    """
    try:
        # Phase 0 입력 데이터 검증
        if request.phase_number != 0:
            return JSONResponse({
                "success": False,
                "error": "파이프라인은 Phase 0부터 시작해야 합니다"
            }, status_code=400)
        
        if not request.input_data:
            return JSONResponse({
                "success": False,
                "error": "Phase 0 입력 데이터가 필요합니다"
            }, status_code=400)
        
        # 파이프라인 시작
        session_id = await phase_manager.start_pipeline(
            initial_data=request.input_data,
            start_phase=0,
            model_to_use=request.model_to_use or "gemini"
        )
        
        # Phase 0 결과 가져오기
        phase_result = phase_manager.get_phase_result(session_id, 0)
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "phase": 0,
            "result": phase_result.output_data,
            "requires_review": phase_result.metadata.get('requires_review', False)
        })
        
    except Exception as e:
        logger.error(f"파이프라인 시작 오류: {e}")
        log_error('main', e, {'phase': 0})
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/api/phase/execute")
async def execute_phase(request: PhaseRequest):
    """
    특정 Phase 실행
    """
    try:
        # 세션 확인
        if not request.session_id:
            return JSONResponse({
                "success": False,
                "error": "session_id가 필요합니다"
            }, status_code=400)
        
        # Phase 실행
        result = await phase_manager.execute_phase(
            session_id=request.session_id,
            phase_number=request.phase_number,
            input_data=request.input_data,
            model_to_use=request.model_to_use,
            models_to_use=request.models_to_use
        )
        
        return JSONResponse({
            "success": True,
            "session_id": request.session_id,
            "phase": request.phase_number,
            "result": result.output_data,
            "requires_review": result.metadata.get('requires_review', False)
        })
        
    except Exception as e:
        logger.error(f"Phase 실행 오류: {e}")
        log_error('main', e, {'phase': request.phase_number, 'session': request.session_id})
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/api/phase/approve")
async def approve_phase(request: PhaseApprovalRequest):
    """
    Phase 결과 승인/수정
    """
    try:
        # Phase 결과 승인
        success = phase_manager.approve_phase_result(
            session_id=request.session_id,
            phase_number=request.phase_number,
            modified_data=request.modified_data
        )
        
        if not success:
            return JSONResponse({
                "success": False,
                "error": "Phase 결과를 찾을 수 없습니다"
            }, status_code=404)
        
        return JSONResponse({
            "success": True,
            "message": f"Phase {request.phase_number} 승인 완료",
            "can_continue": phase_manager._can_continue_to_next(request.session_id)
        })
        
    except Exception as e:
        logger.error(f"Phase 승인 오류: {e}")
        log_error('main', e, {'phase': request.phase_number, 'session': request.session_id})
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/api/phase/continue")
async def continue_phase(request: Dict[str, Any]):
    """
    다음 Phase로 진행
    """
    try:
        session_id = request.get('session_id')
        models_to_use = request.get('models_to_use')
        
        if not session_id:
            return JSONResponse({
                "success": False,
                "error": "session_id가 필요합니다"
            }, status_code=400)
        
        logger.info(f"Phase 진행 시작 - 세션: {session_id}, 모델: {models_to_use}")
        
        # 다음 Phase 실행 - 완료될 때까지 대기
        result = await phase_manager.continue_to_next_phase(
            session_id=session_id,
            models_to_use=models_to_use
        )
        
        if not result:
            logger.info(f"모든 Phase 완료됨 - 세션: {session_id}")
            return JSONResponse({
                "success": True,
                "message": "모든 Phase가 완료되었습니다",
                "completed": True
            })
        
        logger.info(f"Phase {result.phase_number} 완료 - 세션: {session_id}, 성공: {result.output_data.get('success', False)}")
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "phase": result.phase_number,
            "result": result.output_data,
            "requires_review": result.metadata.get('requires_review', False)
        })
        
    except Exception as e:
        logger.error(f"Phase 진행 오류: {e}")
        log_error('main', e, {'session': session_id if 'session_id' in locals() else 'unknown'})
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/pipeline/status/{session_id}")
async def get_pipeline_status(session_id: str):
    """
    파이프라인 진행 상태 조회
    """
    try:
        status = phase_manager.get_pipeline_status(session_id)
        
        if 'error' in status:
            return JSONResponse({
                "success": False,
                "error": status['error']
            }, status_code=404)
        
        return JSONResponse({
            "success": True,
            "status": status
        })
        
    except Exception as e:
        logger.error(f"상태 조회 오류: {e}")
        log_error('main', e, {'session': session_id})
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/api/pipeline/save/{session_id}")
async def save_pipeline(session_id: str):
    """
    파이프라인 세션 저장
    """
    try:
        file_path = phase_manager.save_session(session_id)
        
        return JSONResponse({
            "success": True,
            "message": "세션 저장 완료",
            "file_path": file_path
        })
        
    except Exception as e:
        logger.error(f"세션 저장 오류: {e}")
        log_error('main', e, {'session': session_id})
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/api/upload-json")
async def upload_json(file: UploadFile = File(...)):
    """JSON 파일 업로드 처리"""
    try:
        content = await file.read()
        json_data = json.loads(content.decode('utf-8'))
        
        # 기본 검증
        if not isinstance(json_data, list):
            raise ValueError("JSON should be a list format")
            
        return JSONResponse({
            "success": True,
            "message": "JSON uploaded successfully",
            "data": json_data
        })
        
    except json.JSONDecodeError:
        return JSONResponse({
            "success": False,
            "error": "Invalid JSON format"
        }, status_code=400)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/health")
async def health_check():
    """헬스체크 엔드포인트"""
    try:
        # 모델 상태 확인
        orchestrator = ModelOrchestrator()
        available_models = orchestrator.get_available_models()
        api_key_status = orchestrator.validate_api_keys()
        
        return {
            "status": "healthy", 
            "version": "1.0.0",
            "available_models": available_models,
            "api_keys_configured": {
                "openai": api_key_status.get('openai', False),
                "anthropic": api_key_status.get('anthropic', False), 
                "google": api_key_status.get('google', False)
            },
            "total_models_available": len(available_models)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "version": "1.0.0"
        }

@app.get("/api/models/status")
async def models_status():
    """모델별 상태 확인"""
    try:
        orchestrator = ModelOrchestrator()
        api_keys = orchestrator.validate_api_keys()
        
        status = {}
        for model_type, has_key in api_keys.items():
            if has_key:
                status[model_type] = {
                    "available": True,
                    "api_key_configured": True,
                    "status": "ready"
                }
            else:
                status[model_type] = {
                    "available": False,
                    "api_key_configured": False,
                    "status": "missing_api_key"
                }
        
        return {
            "models": status,
            "total_available": sum(1 for s in status.values() if s["available"])
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "models": {}
        }

@app.get("/api/test-data/load")
async def load_test_data():
    """
    테스트 데이터 로드 - /test_data 경로의 샘플 파일들 반환
    """
    try:
        import os
        from pathlib import Path
        
        # 프로젝트 루트에서 test_data 경로
        project_root = Path(__file__).parent.parent
        test_data_dir = project_root / "test_data"
        
        # 파일 경로들
        measurement_file = test_data_dir / "sample_measurement.json"
        demo_file = test_data_dir / "sample_demo.json"
        intake_file = test_data_dir / "sample_intake_form.txt"
        
        # 파일 존재 확인
        if not measurement_file.exists():
            raise FileNotFoundError(f"Measurement file not found: {measurement_file}")
        if not demo_file.exists():
            raise FileNotFoundError(f"Demo file not found: {demo_file}")
        if not intake_file.exists():
            raise FileNotFoundError(f"Intake form file not found: {intake_file}")
        
        # JSON 파일 읽기
        with open(measurement_file, 'r', encoding='utf-8') as f:
            measurement_data = json.load(f)
        
        with open(demo_file, 'r', encoding='utf-8') as f:
            demo_data = json.load(f)
        
        # Intake form 텍스트 파일 읽기
        with open(intake_file, 'r', encoding='utf-8') as f:
            intake_text = f.read().strip()
        
        return JSONResponse({
            "success": True,
            "data": {
                "measurement_data": measurement_data,
                "demo_data": demo_data,
                "intake_text": intake_text
            },
            "message": "Test data loaded successfully"
        })
        
    except FileNotFoundError as e:
        logger.error(f"테스트 데이터 파일 없음: {e}")
        return JSONResponse({
            "success": False,
            "error": f"Test data file not found: {str(e)}"
        }, status_code=404)
        
    except json.JSONDecodeError as e:
        logger.error(f"테스트 데이터 JSON 파싱 오류: {e}")
        return JSONResponse({
            "success": False,
            "error": f"Invalid JSON in test data file: {str(e)}"
        }, status_code=400)
        
    except Exception as e:
        logger.error(f"테스트 데이터 로드 오류: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    logger.info("서버 시작 중...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)