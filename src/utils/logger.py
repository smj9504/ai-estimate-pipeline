# src/utils/logger.py
"""
중앙화된 로깅 시스템
- 환경 변수 기반 로그 레벨 설정
- 색상 코딩된 콘솔 출력
- 파일 로깅 옵션
- 구조화된 로그 포맷
"""
import logging
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum

class LogLevel(Enum):
    """로그 레벨 정의"""
    DEBUG = logging.DEBUG      # 10 - 상세한 디버그 정보
    INFO = logging.INFO        # 20 - 일반 정보
    WARNING = logging.WARNING  # 30 - 경고
    ERROR = logging.ERROR      # 40 - 에러
    CRITICAL = logging.CRITICAL # 50 - 심각한 에러

class ColorCodes:
    """콘솔 출력용 색상 코드"""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    
    @staticmethod
    def colorize(text: str, color: str) -> str:
        """텍스트에 색상 적용"""
        if os.name == 'nt':  # Windows에서는 색상 코드 제한적
            try:
                import colorama
                colorama.init()
            except ImportError:
                return text
        return f"{color}{text}{ColorCodes.RESET}"

class CustomFormatter(logging.Formatter):
    """커스텀 로그 포매터 - 색상 및 구조화된 출력"""
    
    # 로그 레벨별 색상
    COLORS = {
        logging.DEBUG: ColorCodes.CYAN,
        logging.INFO: ColorCodes.GREEN,
        logging.WARNING: ColorCodes.YELLOW,
        logging.ERROR: ColorCodes.RED,
        logging.CRITICAL: ColorCodes.MAGENTA
    }
    
    def __init__(self, use_colors: bool = True, debug_mode: bool = False):
        self.use_colors = use_colors
        self.debug_mode = debug_mode
        
        if debug_mode:
            # 디버그 모드: 상세 정보 포함
            format_str = "[%(asctime)s] [%(name)s:%(lineno)d] [%(levelname)s] %(message)s"
        else:
            # 일반 모드: 간결한 출력
            format_str = "[%(asctime)s] [%(levelname)s] %(message)s"
        
        super().__init__(format_str, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record):
        # 원본 레벨명 저장
        original_levelname = record.levelname
        
        if self.use_colors and record.levelno in self.COLORS:
            # 레벨명에 색상 적용
            record.levelname = ColorCodes.colorize(
                f"{record.levelname:8}",
                self.COLORS[record.levelno]
            )
            
            # 에러/크리티컬 메시지는 빨간색으로
            if record.levelno >= logging.ERROR:
                record.msg = ColorCodes.colorize(str(record.msg), ColorCodes.RED)
        
        formatted = super().format(record)
        
        # 원본 레벨명 복원
        record.levelname = original_levelname
        
        return formatted

class AIEstimateLogger:
    """AI Estimate Pipeline 전용 로거"""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.debug_mode = self._get_debug_mode()
            self.log_level = self._get_log_level()
            self.log_dir = Path("logs")
            self.log_dir.mkdir(exist_ok=True)
            
            # 로그 파일 설정
            self.log_file = self.log_dir / f"ai_estimate_{datetime.now().strftime('%Y%m%d')}.log"
    
    def _get_debug_mode(self) -> bool:
        """환경 변수에서 디버그 모드 확인"""
        debug_env = os.environ.get('DEBUG', '').lower()
        return debug_env in ('true', '1', 'yes', 'on')
    
    def _get_log_level(self) -> int:
        """환경 변수에서 로그 레벨 확인"""
        if self.debug_mode:
            return logging.DEBUG
        
        level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return level_map.get(level_str, logging.INFO)
    
    def get_logger(self, name: str) -> logging.Logger:
        """모듈별 로거 생성/반환"""
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        
        # 기존 핸들러 제거
        logger.handlers = []
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_formatter = CustomFormatter(
            use_colors=True,
            debug_mode=self.debug_mode
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러 (에러 이상만)
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.ERROR)
        file_formatter = CustomFormatter(
            use_colors=False,
            debug_mode=True  # 파일에는 항상 상세 정보
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 중복 로깅 방지
        logger.propagate = False
        
        self._loggers[name] = logger
        return logger
    
    def log_json(self, logger_name: str, level: str, message: str, data: Any):
        """JSON 데이터를 포함한 구조화된 로깅"""
        logger = self.get_logger(logger_name)
        
        if self.debug_mode:
            # 디버그 모드에서는 JSON을 예쁘게 출력
            try:
                if isinstance(data, (dict, list)):
                    json_str = json.dumps(data, indent=2, ensure_ascii=False)
                    formatted_message = f"{message}\n{ColorCodes.colorize(json_str, ColorCodes.BLUE)}"
                else:
                    formatted_message = f"{message}: {data}"
            except Exception:
                formatted_message = f"{message}: {data}"
        else:
            # 일반 모드에서는 간결하게
            formatted_message = f"{message}"
        
        level_method = getattr(logger, level.lower(), logger.info)
        level_method(formatted_message)
    
    def log_phase_start(self, phase_number: int, phase_name: str, **kwargs):
        """Phase 시작 로깅"""
        logger = self.get_logger('phase_manager')
        logger.info(f"{'='*50}")
        logger.info(f"Phase {phase_number}: {phase_name} 시작")
        
        if self.debug_mode and kwargs:
            logger.debug(f"Phase 파라미터: {json.dumps(kwargs, indent=2, ensure_ascii=False)}")
    
    def log_phase_end(self, phase_number: int, phase_name: str, success: bool, duration: float = None):
        """Phase 종료 로깅"""
        logger = self.get_logger('phase_manager')
        
        status = ColorCodes.colorize("성공", ColorCodes.GREEN) if success else ColorCodes.colorize("실패", ColorCodes.RED)
        msg = f"Phase {phase_number}: {phase_name} {status}"
        
        if duration:
            msg += f" (소요시간: {duration:.2f}초)"
        
        logger.info(msg)
        logger.info(f"{'='*50}")
    
    def log_model_call(self, model_name: str, prompt_length: int, response_time: float = None):
        """AI 모델 호출 로깅"""
        logger = self.get_logger('model_interface')
        
        logger.info(f"모델 호출: {model_name}")
        
        if self.debug_mode:
            logger.debug(f"  - 프롬프트 길이: {prompt_length:,} 문자")
            if response_time:
                logger.debug(f"  - 응답 시간: {response_time:.2f}초")
    
    def log_error(self, module: str, error: Exception, context: Dict[str, Any] = None):
        """에러 로깅"""
        logger = self.get_logger(module)
        
        logger.error(f"에러 발생: {type(error).__name__}: {str(error)}")
        
        if self.debug_mode:
            import traceback
            logger.debug(f"스택 트레이스:\n{traceback.format_exc()}")
            
            if context:
                logger.debug(f"컨텍스트: {json.dumps(context, indent=2, ensure_ascii=False)}")
    
    def log_validation(self, validator_name: str, result: bool, details: Dict[str, Any] = None):
        """검증 결과 로깅"""
        logger = self.get_logger('validator')
        
        if result:
            logger.info(f"✓ {validator_name} 검증 통과")
        else:
            logger.warning(f"✗ {validator_name} 검증 실패")
        
        if self.debug_mode and details:
            logger.debug(f"검증 상세: {json.dumps(details, indent=2, ensure_ascii=False)}")

# 싱글톤 인스턴스
logger_instance = AIEstimateLogger()

# 편의 함수들
def get_logger(name: str) -> logging.Logger:
    """로거 인스턴스 반환"""
    return logger_instance.get_logger(name)

def log_json(logger_name: str, level: str, message: str, data: Any):
    """JSON 로깅"""
    logger_instance.log_json(logger_name, level, message, data)

def log_phase_start(phase_number: int, phase_name: str, **kwargs):
    """Phase 시작 로깅"""
    logger_instance.log_phase_start(phase_number, phase_name, **kwargs)

def log_phase_end(phase_number: int, phase_name: str, success: bool, duration: float = None):
    """Phase 종료 로깅"""
    logger_instance.log_phase_end(phase_number, phase_name, success, duration)

def log_model_call(model_name: str, prompt_length: int, response_time: float = None):
    """모델 호출 로깅"""
    logger_instance.log_model_call(model_name, prompt_length, response_time)

def log_error(module: str, error: Exception, context: Dict[str, Any] = None):
    """에러 로깅"""
    logger_instance.log_error(module, error, context)

# 사용 예시
if __name__ == "__main__":
    # 환경 변수 설정 예시
    os.environ['DEBUG'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    # 로거 사용
    logger = get_logger('test_module')
    
    logger.debug("디버그 메시지")
    logger.info("정보 메시지")
    logger.warning("경고 메시지")
    logger.error("에러 메시지")
    logger.critical("심각한 에러")
    
    # JSON 로깅
    log_json('test_module', 'info', "JSON 데이터", {"key": "value", "number": 123})
    
    # Phase 로깅
    log_phase_start(0, "Generate Scope of Work", model="gpt4")
    log_phase_end(0, "Generate Scope of Work", True, 3.5)
    
    # 모델 호출 로깅
    log_model_call("gpt4", 5000, 2.3)
    
    # 에러 로깅
    try:
        raise ValueError("테스트 에러")
    except Exception as e:
        log_error('test_module', e, {"context": "test"})