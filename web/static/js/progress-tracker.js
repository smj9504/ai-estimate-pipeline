// Dynamic Progress Tracker for AI Estimate Pipeline
class ProgressTracker {
    constructor() {
        this.startTime = null;
        this.currentStep = 0;
        this.totalSteps = 4;
        this.modelStatus = {
            'gpt4': { status: 'waiting', progress: 0, message: '' },
            'claude': { status: 'waiting', progress: 0, message: '' },
            'gemini': { status: 'waiting', progress: 0, message: '' }
        };
        this.steps = [
            { name: 'Sending requests to AI models', icon: '📤', duration: 2000 },
            { name: 'Collecting responses', icon: '📥', duration: 3000 },
            { name: 'Merging results with confidence scoring', icon: '🔄', duration: 2500 },
            { name: 'Validating output data', icon: '✅', duration: 1500 }
        ];
        this.intervalId = null;
        this.modelIntervalId = null;
    }

    start(selectedModels) {
        this.startTime = Date.now();
        this.currentStep = 0;
        this.selectedModels = selectedModels;
        
        // Initialize model status for selected models
        selectedModels.forEach(model => {
            if (this.modelStatus[model]) {
                this.modelStatus[model] = { status: 'pending', progress: 0, message: 'Preparing...' };
            }
        });
        
        this.updateDisplay();
        this.startStepProgress();
        this.startModelProgress();
    }

    startStepProgress() {
        let stepStartTime = Date.now();
        
        this.intervalId = setInterval(() => {
            const currentStepDuration = this.steps[this.currentStep].duration;
            const elapsed = Date.now() - stepStartTime;
            const progress = Math.min((elapsed / currentStepDuration) * 100, 100);
            
            this.updateStepProgress(progress);
            
            if (progress >= 100) {
                this.completeCurrentStep();
                this.currentStep++;
                
                if (this.currentStep < this.totalSteps) {
                    stepStartTime = Date.now();
                } else {
                    this.complete();
                }
            }
        }, 50);
    }

    startModelProgress() {
        this.modelIntervalId = setInterval(() => {
            this.selectedModels.forEach((model, index) => {
                const modelData = this.modelStatus[model];
                const delay = index * 500; // Stagger model starts
                const elapsed = Date.now() - this.startTime - delay;
                
                if (elapsed > 0) {
                    if (modelData.progress < 100) {
                        // Simulate progress
                        modelData.progress = Math.min(modelData.progress + Math.random() * 15, 100);
                        
                        // Update status based on progress
                        if (modelData.progress < 30) {
                            modelData.status = 'connecting';
                            modelData.message = 'Connecting to API...';
                        } else if (modelData.progress < 60) {
                            modelData.status = 'processing';
                            modelData.message = 'Processing request...';
                        } else if (modelData.progress < 90) {
                            modelData.status = 'analyzing';
                            modelData.message = 'Analyzing data...';
                        } else if (modelData.progress >= 100) {
                            modelData.status = 'completed';
                            modelData.message = 'Response received';
                        }
                        
                        this.updateModelDisplay(model);
                    }
                }
            });
        }, 200);
    }

    updateDisplay() {
        const container = document.querySelector('.processing-indicator');
        if (!container) return;
        
        const elapsed = this.getElapsedTime();
        const estimated = this.getEstimatedTime();
        
        container.innerHTML = `
            <h4>Phase Processing in Progress</h4>
            
            <div class="time-tracking">
                <div class="time-item">
                    <span class="time-label">Elapsed</span>
                    <span class="time-value">${elapsed}</span>
                </div>
                <div class="time-item">
                    <span class="time-label">Estimated</span>
                    <span class="time-value">${estimated}</span>
                </div>
            </div>
            
            <div class="overall-progress">
                <div class="progress-header">
                    <span>Overall Progress</span>
                    <span class="progress-percentage">0%</span>
                </div>
                <div class="progress" style="height: 30px;">
                    <div class="progress-bar progress-bar-striped progress-bar-animated" 
                         role="progressbar" style="width: 0%">
                    </div>
                </div>
            </div>
            
            <div class="models-status">
                <h5>AI Models Status</h5>
                <div class="model-cards">
                    ${this.getModelCardsHTML()}
                </div>
            </div>
            
            <div class="steps-progress">
                <h5>Processing Steps</h5>
                <div class="steps-list">
                    ${this.getStepsHTML()}
                </div>
            </div>
            
            <style>
            .processing-indicator {
                padding: 30px;
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border-radius: 12px;
                border: 2px solid #0066cc;
            }
            
            .time-tracking {
                display: flex;
                justify-content: center;
                gap: 40px;
                margin: 20px 0;
            }
            
            .time-item {
                text-align: center;
            }
            
            .time-label {
                display: block;
                font-size: 0.85em;
                color: #6c757d;
                margin-bottom: 5px;
            }
            
            .time-value {
                display: block;
                font-size: 1.5em;
                font-weight: bold;
                color: #0066cc;
                font-family: 'Courier New', monospace;
            }
            
            .overall-progress {
                margin: 30px 0;
            }
            
            .progress-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 10px;
                font-weight: 600;
            }
            
            .progress-percentage {
                color: #0066cc;
            }
            
            .models-status {
                margin: 30px 0;
            }
            
            .model-cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            
            .model-card {
                background: white;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
                transition: all 0.3s ease;
            }
            
            .model-card.active {
                border-color: #0066cc;
                box-shadow: 0 0 10px rgba(0, 102, 204, 0.2);
            }
            
            .model-card.completed {
                border-color: #28a745;
                background: #f0fff4;
            }
            
            .model-name {
                font-weight: bold;
                margin-bottom: 8px;
                font-size: 1.1em;
            }
            
            .model-status {
                font-size: 0.85em;
                color: #6c757d;
                margin-bottom: 10px;
            }
            
            .model-progress {
                height: 6px;
                background: #e9ecef;
                border-radius: 3px;
                overflow: hidden;
                margin-bottom: 5px;
            }
            
            .model-progress-bar {
                height: 100%;
                background: linear-gradient(90deg, #0066cc, #0052a3);
                transition: width 0.3s ease;
            }
            
            .model-progress-bar.completed {
                background: linear-gradient(90deg, #28a745, #218838);
            }
            
            .model-percentage {
                font-size: 0.8em;
                color: #0066cc;
                font-weight: bold;
            }
            
            .steps-progress {
                margin-top: 30px;
            }
            
            .steps-list {
                margin-top: 15px;
            }
            
            .step-item {
                display: flex;
                align-items: center;
                padding: 10px;
                background: white;
                border-radius: 6px;
                margin-bottom: 8px;
                border: 1px solid #e9ecef;
                transition: all 0.3s ease;
            }
            
            .step-item.active {
                background: #e7f3ff;
                border-color: #0066cc;
                transform: translateX(5px);
            }
            
            .step-item.completed {
                background: #f0fff4;
                border-color: #28a745;
            }
            
            .step-icon {
                font-size: 1.5em;
                margin-right: 15px;
            }
            
            .step-content {
                flex: 1;
            }
            
            .step-name {
                font-weight: 500;
                margin-bottom: 3px;
            }
            
            .step-status {
                font-size: 0.85em;
                color: #6c757d;
            }
            
            .step-indicator {
                margin-left: auto;
            }
            
            .spinner-small {
                width: 20px;
                height: 20px;
                border: 2px solid #f3f3f3;
                border-top: 2px solid #0066cc;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            .check-icon {
                color: #28a745;
                font-size: 1.5em;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.6; }
                100% { opacity: 1; }
            }
            
            .model-card.active .model-status {
                animation: pulse 1.5s infinite;
            }
            </style>
        `;
    }

    getModelCardsHTML() {
        const modelNames = {
            'gpt4': 'GPT-4',
            'claude': 'Claude',
            'gemini': 'Gemini'
        };
        
        return this.selectedModels.map(model => {
            const data = this.modelStatus[model];
            const statusClass = data.status === 'completed' ? 'completed' : 
                               data.status === 'waiting' ? '' : 'active';
            
            return `
                <div class="model-card ${statusClass}" id="model-${model}">
                    <div class="model-name">${modelNames[model]}</div>
                    <div class="model-status">${data.message || 'Waiting...'}</div>
                    <div class="model-progress">
                        <div class="model-progress-bar ${data.status === 'completed' ? 'completed' : ''}" 
                             style="width: ${data.progress}%"></div>
                    </div>
                    <div class="model-percentage">${Math.round(data.progress)}%</div>
                </div>
            `;
        }).join('');
    }

    getStepsHTML() {
        return this.steps.map((step, index) => {
            const isActive = index === this.currentStep;
            const isCompleted = index < this.currentStep;
            const statusClass = isCompleted ? 'completed' : (isActive ? 'active' : '');
            
            return `
                <div class="step-item ${statusClass}" id="step-${index}">
                    <div class="step-icon">${step.icon}</div>
                    <div class="step-content">
                        <div class="step-name">${step.name}</div>
                        <div class="step-status">
                            ${isCompleted ? 'Completed' : 
                              isActive ? 'In progress...' : 
                              'Pending'}
                        </div>
                    </div>
                    <div class="step-indicator">
                        ${isCompleted ? '<span class="check-icon">✓</span>' : 
                          isActive ? '<div class="spinner-small"></div>' : 
                          ''}
                    </div>
                </div>
            `;
        }).join('');
    }

    updateStepProgress(progress) {
        const overallProgress = ((this.currentStep / this.totalSteps) * 100) + 
                               (progress / this.totalSteps);
        
        const progressBar = document.querySelector('.overall-progress .progress-bar');
        const progressPercentage = document.querySelector('.progress-percentage');
        
        if (progressBar) {
            progressBar.style.width = `${overallProgress}%`;
        }
        if (progressPercentage) {
            progressPercentage.textContent = `${Math.round(overallProgress)}%`;
        }
        
        // Update time display
        const timeValue = document.querySelector('.time-value');
        if (timeValue) {
            timeValue.textContent = this.getElapsedTime();
        }
    }

    updateModelDisplay(model) {
        const card = document.getElementById(`model-${model}`);
        if (!card) return;
        
        const data = this.modelStatus[model];
        const statusEl = card.querySelector('.model-status');
        const progressBar = card.querySelector('.model-progress-bar');
        const percentageEl = card.querySelector('.model-percentage');
        
        if (statusEl) statusEl.textContent = data.message;
        if (progressBar) {
            progressBar.style.width = `${data.progress}%`;
            if (data.status === 'completed') {
                progressBar.classList.add('completed');
                card.classList.add('completed');
                card.classList.remove('active');
            } else if (data.status !== 'waiting') {
                card.classList.add('active');
            }
        }
        if (percentageEl) percentageEl.textContent = `${Math.round(data.progress)}%`;
    }

    completeCurrentStep() {
        const stepEl = document.getElementById(`step-${this.currentStep}`);
        if (stepEl) {
            stepEl.classList.remove('active');
            stepEl.classList.add('completed');
            const indicator = stepEl.querySelector('.step-indicator');
            if (indicator) {
                indicator.innerHTML = '<span class="check-icon">✓</span>';
            }
            const status = stepEl.querySelector('.step-status');
            if (status) {
                status.textContent = 'Completed';
            }
        }
    }

    getElapsedTime() {
        if (!this.startTime) return '00:00';
        const elapsed = Date.now() - this.startTime;
        const seconds = Math.floor(elapsed / 1000);
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`;
    }

    getEstimatedTime() {
        const totalDuration = this.steps.reduce((sum, step) => sum + step.duration, 0);
        const remainingSteps = this.steps.slice(this.currentStep);
        const remainingDuration = remainingSteps.reduce((sum, step) => sum + step.duration, 0);
        const seconds = Math.floor(remainingDuration / 1000);
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`;
    }

    complete() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        if (this.modelIntervalId) {
            clearInterval(this.modelIntervalId);
            this.modelIntervalId = null;
        }
        
        // Set all to 100%
        const progressBar = document.querySelector('.overall-progress .progress-bar');
        const progressPercentage = document.querySelector('.progress-percentage');
        
        if (progressBar) {
            progressBar.style.width = '100%';
            progressBar.classList.remove('progress-bar-animated');
        }
        if (progressPercentage) {
            progressPercentage.textContent = '100%';
        }
        
        // Mark all models as completed
        this.selectedModels.forEach(model => {
            this.modelStatus[model].progress = 100;
            this.modelStatus[model].status = 'completed';
            this.modelStatus[model].message = 'Response received';
            this.updateModelDisplay(model);
        });
    }

    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        if (this.modelIntervalId) {
            clearInterval(this.modelIntervalId);
            this.modelIntervalId = null;
        }
    }
}

// Export for use in pipeline.html
window.ProgressTracker = ProgressTracker;