# src/models/data_models.py
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from enum import Enum

class WorkScopeType(str, Enum):
    REMOVE_REPLACE = "Remove & Replace"
    PAINT_ONLY = "Paint"
    INSTALL_NEW = "Install New"
    REPAIR = "Repair"
    NO_WORK = ""

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# 기본 측정 데이터
class Measurements(BaseModel):
    height: float = 0.0
    wall_area_sqft: float = 0.0
    ceiling_area_sqft: float = 0.0
    floor_area_sqft: float = 0.0
    walls_and_ceiling_area_sqft: float = 0.0
    flooring_area_sy: float = 0.0
    ceiling_perimeter_lf: float = 0.0
    floor_perimeter_lf: float = 0.0
    openings: List[Dict[str, Any]] = Field(default_factory=list)

# 재료 정보
class Materials(BaseModel):
    Floor: str = ""
    wall: str = ""
    ceiling: str = ""
    Baseboard: str = ""
    Quarter_Round: str = Field(alias="Quarter Round", default="")

# 작업 범위
class WorkScope(BaseModel):
    Flooring: str = ""
    Wall: str = ""
    Ceiling: str = ""
    Baseboard: str = ""
    Quarter_Round: str = Field(alias="Quarter Round", default="")
    Paint_Scope: str = Field(alias="Paint Scope", default="")

# 이미 철거된 항목
class DemoScope(BaseModel):
    ceiling_drywall_sqft: float = Field(alias="Ceiling Drywall(sq_ft)", default=0)
    wall_drywall_sqft: float = Field(alias="Wall Drywall(sq_ft)", default=0)

# 추가 노트
class AdditionalNotes(BaseModel):
    protection: List[str] = Field(default_factory=list)
    detach_reset: List[str] = Field(default_factory=list)

# 방 정보
class Room(BaseModel):
    name: str
    material: Materials
    work_scope: WorkScope
    measurements: Measurements
    demo_scope: DemoScope = Field(alias="demo_scope(already demo'd)")
    additional_notes: AdditionalNotes

# 층별 정보
class Floor(BaseModel):
    location: str
    rooms: List[Room]

# 프로젝트 정보
class JobsiteInfo(BaseModel):
    Jobsite: str = ""
    occupancy: str = ""
    company: Dict[str, Any] = Field(default_factory=dict)

# 전체 프로젝트 데이터
class ProjectData(BaseModel):
    jobsite_info: JobsiteInfo
    floors: List[Floor]
    
    @classmethod
    def from_json_list(cls, json_list: List[Dict[str, Any]]) -> 'ProjectData':
        """JSON 리스트에서 ProjectData 객체 생성"""
        if len(json_list) < 2:
            raise ValueError("JSON list must contain at least jobsite info and one floor")
        
        jobsite_info = JobsiteInfo(**json_list[0])
        floors = [Floor(**floor_data) for floor_data in json_list[1:]]
        
        return cls(jobsite_info=jobsite_info, floors=floors)

# AI 모델 응답 데이터 구조
class WorkItem(BaseModel):
    task_name: str
    description: str
    necessity: str  # "필수", "옵션", "검토필요"
    quantity: float = 0.0
    unit: str = ""
    reasoning: str = ""
    room_name: str = ""
    
class ModelResponse(BaseModel):
    model_name: str
    room_estimates: List[Dict[str, Any]]
    processing_time: float = 0.0
    total_work_items: int = 0
    validation_checklist: Dict[str, bool] = Field(default_factory=dict)
    confidence_self_assessment: float = 0.0
    raw_response: str = ""

# 병합 결과 메타데이터
class MergeMetadata(BaseModel):
    models_used: List[str]
    consensus_level: float  # 0-1, 모델간 합의 정도
    deviation_metrics: Dict[str, float] = Field(default_factory=dict)
    outlier_flags: List[str] = Field(default_factory=list)
    processing_time_total: float = 0.0
    confidence_level: ConfidenceLevel
    manual_review_required: bool = False
    safety_margin_applied: float = 0.0

# 최종 병합 결과
class MergedEstimate(BaseModel):
    project_info: JobsiteInfo
    rooms: List[Dict[str, Any]]
    total_work_items: int
    overall_confidence: float
    metadata: MergeMetadata
    
    # 요약 통계
    summary_stats: Dict[str, Any] = Field(default_factory=lambda: {
        "total_rooms": 0,
        "total_sqft_flooring": 0,
        "total_sqft_walls": 0,
        "total_sqft_ceiling": 0,
        "high_ceiling_rooms": 0
    })

# 설정 모델
class ModelWeights(BaseModel):
    gpt4: float = 0.35
    claude: float = 0.35
    gemini: float = 0.30
    
    def normalize(self) -> 'ModelWeights':
        """가중치 합이 1이 되도록 정규화"""
        total = self.gpt4 + self.claude + self.gemini
        if total == 0:
            return ModelWeights()  # 기본값 사용
        
        return ModelWeights(
            gpt4=self.gpt4 / total,
            claude=self.claude / total,
            gemini=self.gemini / total
        )

class DeviationThresholds(BaseModel):
    quantity: float = 0.10
    labor: float = 0.20
    timeline: float = 0.15

class SafetyMargins(BaseModel):
    low_variance: float = 0.05
    high_variance: float = 0.10

class ConsensusRules(BaseModel):
    minimum_agreement: int = 2
    outlier_threshold: float = 3.0  # 3-시그마 룰

class ValidationSettings(BaseModel):
    mode: str = "balanced"  # strict, balanced, lenient

class AppConfig(BaseModel):
    model_weights: ModelWeights = Field(default_factory=ModelWeights)
    deviation_thresholds: DeviationThresholds = Field(default_factory=DeviationThresholds)
    safety_margins: SafetyMargins = Field(default_factory=SafetyMargins)
    consensus: ConsensusRules = Field(default_factory=ConsensusRules)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)

# 유틸리티 함수들
def calculate_confidence_level(consensus_score: float) -> ConfidenceLevel:
    """합의 점수를 기반으로 신뢰도 레벨 계산"""
    if consensus_score >= 0.8:
        return ConfidenceLevel.HIGH
    elif consensus_score >= 0.6:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW

def is_high_ceiling(height: float) -> bool:
    """높은 천장 여부 판단 (9피트 초과)"""
    return height > 9.0

def requires_remove_replace(work_scope: str) -> bool:
    """Remove & Replace 로직 적용 필요 여부"""
    return work_scope.strip() == WorkScopeType.REMOVE_REPLACE