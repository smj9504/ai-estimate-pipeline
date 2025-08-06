# src/main.py
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any
import json
import os
from pathlib import Path

# 프로젝트 내부 모듈들
from src.models.model_interface import ModelOrchestrator
from src.processors.result_merger import ResultMerger
from src.validators.estimation_validator import ComprehensiveValidator
from src.models.data_models import ProjectData

app = FastAPI(title="Reconstruction Estimator", version="1.0.0")

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

@app.post("/api/estimate/merge", response_model=EstimateResponse)
async def merge_estimates(request: EstimateRequest):
    """
    메인 API - 여러 모델 결과를 병합하여 최종 견적 생성
    """
    try:
        print(f"견적 요청 시작: {request.models_to_use}")
        
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
        orchestrator = ModelOrchestrator()
        available_models = orchestrator.get_available_models()
        
        # 요청된 모델 중 사용 가능한 것만 필터링
        models_to_use = [model for model in request.models_to_use if model in available_models]
        
        if not models_to_use:
            return EstimateResponse(
                success=False,
                error_message="요청된 모델들을 사용할 수 없습니다. API 키를 확인하세요."
            )
        
        print(f"사용 가능한 모델: {models_to_use}")
        
        # 모델 병렬 실행
        model_results = await orchestrator.run_parallel(
            base_prompt, 
            request.json_data, 
            models_to_use
        )
        
        if not model_results:
            return EstimateResponse(
                success=False,
                error_message="모든 모델 호출이 실패했습니다."
            )
        
        print(f"모델 실행 완료: {len(model_results)}개 결과")
        
        # 3. 결과 병합
        merger = ResultMerger()
        merged_result = merger.merge_results(model_results)
        
        print(f"병합 완료: 신뢰도 {merged_result.overall_confidence:.2f}")
        
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
            print(f"검증 중 오류 (계속 진행): {e}")
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
        print(f"견적 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
        
        return EstimateResponse(
            success=False,
            error_message=f"처리 중 오류 발생: {str(e)}"
        )

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)