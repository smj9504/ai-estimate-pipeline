// web/static/js/progress_monitor.js
/**
 * AI 모델 진행 상황 모니터링 JavaScript
 */

class ProgressMonitor {
    constructor() {
        this.ws = null;
        this.eventSource = null;
        this.useWebSocket = true; // WebSocket 우선 사용
        this.progressData = {};
        this.sessionId = null;
    }

    /**
     * 진행 상황 모니터링 시작
     */
    start(sessionId) {
        this.sessionId = sessionId;
        
        // 세션 시작 API 호출
        fetch(`/api/progress/start/${sessionId}`, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log('Progress session started:', data);
            });

        // WebSocket 또는 SSE 연결
        if (this.useWebSocket) {
            this.connectWebSocket();
        } else {
            this.connectSSE();
        }
    }

    /**
     * WebSocket 연결
     */
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/progress/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.showConnectionStatus('connected');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleProgressUpdate(data);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showConnectionStatus('error');
            
            // WebSocket 실패 시 SSE로 폴백
            this.useWebSocket = false;
            this.connectSSE();
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.showConnectionStatus('disconnected');
        };
        
        // 주기적인 ping 전송 (연결 유지)
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send('ping');
            }
        }, 30000);
    }

    /**
     * Server-Sent Events 연결 (WebSocket 대안)
     */
    connectSSE() {
        this.eventSource = new EventSource('/api/progress/stream');
        
        this.eventSource.onopen = () => {
            console.log('SSE connected');
            this.showConnectionStatus('connected');
        };
        
        this.eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleProgressUpdate(data);
        };
        
        this.eventSource.onerror = (error) => {
            console.error('SSE error:', error);
            this.showConnectionStatus('error');
        };
    }

    /**
     * 진행 상황 업데이트 처리
     */
    handleProgressUpdate(data) {
        console.log('Progress update:', data);
        
        // 모델별 데이터 저장
        if (!this.progressData[data.model_name]) {
            this.progressData[data.model_name] = [];
        }
        this.progressData[data.model_name].push(data);
        
        // UI 업데이트
        this.updateProgressUI(data);
        
        // 상태별 처리
        switch(data.status) {
            case 'preparing':
                this.showModelStatus(data.model_name, '준비 중...', 'info');
                break;
            case 'sending_request':
                this.showModelStatus(data.model_name, 'API 요청 전송 중...', 'info');
                break;
            case 'waiting_response':
                this.showModelStatus(data.model_name, '응답 대기 중...', 'warning');
                break;
            case 'processing_response':
                this.showModelStatus(data.model_name, '응답 처리 중...', 'info');
                break;
            case 'completed':
                this.showModelStatus(data.model_name, '완료!', 'success');
                this.showModelDetails(data.model_name, data.details);
                break;
            case 'error':
                this.showModelStatus(data.model_name, `오류: ${data.message}`, 'danger');
                break;
        }
    }

    /**
     * 진행률 UI 업데이트
     */
    updateProgressUI(data) {
        const modelName = data.model_name;
        const progress = data.progress || 0;
        
        // 진행률 바 컨테이너 찾기 또는 생성
        let container = document.getElementById(`progress-${modelName}`);
        if (!container) {
            container = this.createProgressContainer(modelName);
        }
        
        // 진행률 바 업데이트
        const progressBar = container.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
            progressBar.textContent = `${Math.round(progress)}%`;
            
            // 상태에 따른 색상 변경
            progressBar.className = 'progress-bar progress-bar-striped progress-bar-animated';
            if (data.status === 'completed') {
                progressBar.classList.add('bg-success');
                progressBar.classList.remove('progress-bar-animated');
            } else if (data.status === 'error') {
                progressBar.classList.add('bg-danger');
                progressBar.classList.remove('progress-bar-animated');
            } else {
                progressBar.classList.add('bg-info');
            }
        }
        
        // 메시지 업데이트
        const messageElement = container.querySelector('.progress-message');
        if (messageElement) {
            messageElement.textContent = data.message;
        }
        
        // 세부 정보 업데이트
        if (data.details) {
            const detailsElement = container.querySelector('.progress-details');
            if (detailsElement) {
                detailsElement.innerHTML = this.formatDetails(data.details);
            }
        }
    }

    /**
     * 진행률 컨테이너 생성
     */
    createProgressContainer(modelName) {
        const container = document.createElement('div');
        container.id = `progress-${modelName}`;
        container.className = 'model-progress-container mb-3';
        container.innerHTML = `
            <h5>${this.formatModelName(modelName)}</h5>
            <div class="progress mb-2">
                <div class="progress-bar progress-bar-striped progress-bar-animated bg-info" 
                     role="progressbar" 
                     style="width: 0%" 
                     aria-valuenow="0" 
                     aria-valuemin="0" 
                     aria-valuemax="100">0%</div>
            </div>
            <div class="progress-message text-muted small">대기 중...</div>
            <div class="progress-details text-muted small mt-1"></div>
        `;
        
        // 진행 상황 영역에 추가
        const progressArea = document.getElementById('progress-area');
        if (progressArea) {
            progressArea.appendChild(container);
        }
        
        return container;
    }

    /**
     * 모델 이름 포맷팅
     */
    formatModelName(modelName) {
        const names = {
            'gpt-4': 'GPT-4',
            'claude-3-sonnet': 'Claude 3 Sonnet',
            'gemini-pro': 'Gemini Pro'
        };
        return names[modelName] || modelName;
    }

    /**
     * 세부 정보 포맷팅
     */
    formatDetails(details) {
        let html = '<div class="d-flex flex-wrap gap-2">';
        
        if (details.response_time) {
            html += `<span class="badge bg-secondary">응답 시간: ${details.response_time.toFixed(2)}초</span>`;
        }
        if (details.response_size) {
            html += `<span class="badge bg-secondary">응답 크기: ${this.formatBytes(details.response_size)}</span>`;
        }
        if (details.work_items !== undefined) {
            html += `<span class="badge bg-primary">작업 항목: ${details.work_items}개</span>`;
        }
        if (details.rooms !== undefined) {
            html += `<span class="badge bg-primary">방: ${details.rooms}개</span>`;
        }
        if (details.chunks_received) {
            html += `<span class="badge bg-info">청크: ${details.chunks_received}개</span>`;
        }
        
        html += '</div>';
        return html;
    }

    /**
     * 바이트 포맷팅
     */
    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    /**
     * 모델 상태 표시
     */
    showModelStatus(modelName, message, type) {
        const statusElement = document.getElementById(`status-${modelName}`);
        if (statusElement) {
            statusElement.className = `alert alert-${type} small`;
            statusElement.textContent = message;
        }
    }

    /**
     * 모델 세부 정보 표시
     */
    showModelDetails(modelName, details) {
        const detailsElement = document.getElementById(`details-${modelName}`);
        if (detailsElement && details) {
            let html = '<ul class="list-unstyled small">';
            for (const [key, value] of Object.entries(details)) {
                html += `<li><strong>${key}:</strong> ${value}</li>`;
            }
            html += '</ul>';
            detailsElement.innerHTML = html;
        }
    }

    /**
     * 연결 상태 표시
     */
    showConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            let badge = '';
            switch(status) {
                case 'connected':
                    badge = '<span class="badge bg-success">연결됨</span>';
                    break;
                case 'disconnected':
                    badge = '<span class="badge bg-warning">연결 끊김</span>';
                    break;
                case 'error':
                    badge = '<span class="badge bg-danger">오류</span>';
                    break;
                default:
                    badge = '<span class="badge bg-secondary">대기</span>';
            }
            statusElement.innerHTML = badge;
        }
    }

    /**
     * 진행 상황 요약 가져오기
     */
    async getSummary() {
        try {
            const response = await fetch('/api/progress/summary');
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Failed to get progress summary:', error);
            return null;
        }
    }

    /**
     * 모니터링 중지
     */
    stop() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        this.progressData = {};
        this.showConnectionStatus('disconnected');
    }

    /**
     * 진행 상황 초기화
     */
    reset() {
        this.progressData = {};
        const progressArea = document.getElementById('progress-area');
        if (progressArea) {
            progressArea.innerHTML = '';
        }
    }
}

// 전역 인스턴스 생성
const progressMonitor = new ProgressMonitor();