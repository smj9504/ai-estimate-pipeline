# src/api/progress_api.py
"""
AI 모델 진행 상황 추적을 위한 WebSocket/SSE API
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
import queue
import threading

router = APIRouter(prefix="/api/progress", tags=["progress"])

class ProgressManager:
    """진행 상황 관리 클래스"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.progress_queue = queue.Queue()
        self.progress_history = {}
        self.current_session = None
        
    async def connect(self, websocket: WebSocket):
        """WebSocket 연결"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        """WebSocket 연결 해제"""
        self.active_connections.remove(websocket)
        
    async def broadcast(self, message: Dict[str, Any]):
        """모든 연결된 클라이언트에 메시지 브로드캐스트"""
        message_str = json.dumps(message)
        
        # WebSocket으로 전송
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except:
                disconnected.append(connection)
        
        # 연결이 끊긴 클라이언트 제거
        for conn in disconnected:
            self.active_connections.remove(conn)
    
    def add_progress(self, model_name: str, status: str, message: str, progress: float, details: Dict[str, Any] = None):
        """진행 상황 추가"""
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.current_session,
            'model_name': model_name,
            'status': status,
            'message': message,
            'progress': progress,
            'details': details or {}
        }
        
        # 큐에 추가 (비동기 처리용)
        self.progress_queue.put(progress_data)
        
        # 히스토리 저장
        if model_name not in self.progress_history:
            self.progress_history[model_name] = []
        self.progress_history[model_name].append(progress_data)
        
        return progress_data
    
    def start_session(self, session_id: str):
        """새 세션 시작"""
        self.current_session = session_id
        self.progress_history.clear()
        
    def get_session_summary(self) -> Dict[str, Any]:
        """현재 세션 요약"""
        summary = {
            'session_id': self.current_session,
            'models': {},
            'overall_progress': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        for model_name, history in self.progress_history.items():
            if history:
                last_update = history[-1]
                summary['models'][model_name] = {
                    'status': last_update['status'],
                    'progress': last_update['progress'],
                    'message': last_update['message'],
                    'update_count': len(history)
                }
        
        # 전체 진행률 계산
        if summary['models']:
            total_progress = sum(m['progress'] for m in summary['models'].values())
            summary['overall_progress'] = total_progress / len(summary['models'])
        
        return summary

# 전역 진행 상황 관리자
progress_manager = ProgressManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 엔드포인트 - 실시간 진행 상황 전송"""
    await progress_manager.connect(websocket)
    
    try:
        while True:
            # 클라이언트로부터 메시지 대기 (연결 유지용)
            data = await websocket.receive_text()
            
            # ping/pong 처리
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        progress_manager.disconnect(websocket)

@router.get("/stream")
async def progress_stream():
    """Server-Sent Events (SSE) 스트림 - WebSocket 대안"""
    
    async def event_generator():
        """이벤트 생성기"""
        while True:
            try:
                # 큐에서 진행 상황 가져오기
                if not progress_manager.progress_queue.empty():
                    progress_data = progress_manager.progress_queue.get()
                    
                    # SSE 형식으로 전송
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    
                    # 브로드캐스트 (WebSocket 클라이언트용)
                    await progress_manager.broadcast(progress_data)
                
                # 짧은 대기
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"SSE 스트림 오류: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

@router.post("/start/{session_id}")
async def start_progress_session(session_id: str):
    """새 진행 상황 추적 세션 시작"""
    progress_manager.start_session(session_id)
    
    return {
        "status": "success",
        "session_id": session_id,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/summary")
async def get_progress_summary():
    """현재 세션의 진행 상황 요약"""
    return progress_manager.get_session_summary()

@router.get("/history/{model_name}")
async def get_model_history(model_name: str):
    """특정 모델의 진행 상황 히스토리"""
    history = progress_manager.progress_history.get(model_name, [])
    
    return {
        "model_name": model_name,
        "history": history,
        "count": len(history)
    }

# 진행 상황 콜백 함수 (모델에서 사용)
def create_progress_callback(model_name: str):
    """모델용 진행 상황 콜백 생성"""
    
    def callback(update_data: Dict[str, Any]):
        """콜백 함수"""
        progress_manager.add_progress(
            model_name=model_name,
            status=update_data.get('status', 'unknown'),
            message=update_data.get('message', ''),
            progress=update_data.get('progress', 0),
            details=update_data.get('details', {})
        )
    
    return callback