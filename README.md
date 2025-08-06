# Reconstruction Estimator

Multi-model AI system for residential reconstruction estimates.

## Setup

프로젝트 설정 확인:
    ```
    python setup.py
    ```

requirements.txt 업데이트
    ```
    pipreqs . --force --encoding utf8
    ```

1. Create virtual environment:
    ```
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Configure environment:
    ```
    cp .env.example .env
    # Edit .env with your API keys
    ```

4. Run the application:
    ```
    python -m uvicorn src.main:app --reload
    ```

## Project Structure

reconstruction_estimator/
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 메인 애플리케이션
│   ├── models/
│   │   ├── __init__.py
│   │   └── data_models.py      # Pydantic 데이터 모델들
│   ├── processors/             # 결과 병합 로직 (다음에 구현)
│   ├── validators/             # 검증 로직 (다음에 구현)
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py    # 설정 로더
│       ├── statistical_utils.py # 통계 처리
│       ├── text_utils.py       # 텍스트 처리
│       └── validation_utils.py # 검증 유틸리티
├── web/
│   ├── templates/
│   │   └── index.html         # 웹 인터페이스
│   └── static/                # CSS, JS 파일
├── config/
│   └── settings.yaml          # 애플리케이션 설정
├── data/
│   └── samples/
│       └── sample_input.json  # 샘플 데이터
├── tests/                     # 테스트 파일들
├── requirements.txt           # Python 의존성
├── .env.example              # 환경변수 템플릿
├── .gitignore                # Git 무시 파일
├── README.md                 # 프로젝트 문서
├── setup.py                  # 프로젝트 설정 스크립트
├── run_server.py            # 서버 실행 스크립트
└── test_setup.py            # 기본 테스트


## Usage

1. Upload JSON data through web interface
2. System calls multiple AI models
3. Results are merged using consensus rules
4. Final estimate with confidence scores is generated

## Development

Run tests:
```
pytest
```
Format code:
```
black src/ tests/
```
