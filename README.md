# AI Investment Research Assistant

**먼저 코드로 최근 거래대금 기준 최대 500개 + 관심 종목들의 데이터를 API로 받아와, 이 중에서 기준에 맞는 종목들만 압축하고, LLM API를 통해 압축된 각 종목들의 근거를 찾아 GPT에게 최종 판단 검토까지 요구하는 개인 리서치 도구입니다.**

먼저 처음에는 코드로만 후보 리서치 프로그램을 만들고, 이 데이터들을 정리한 것을 프로그램이 출력해주면, 제가 직접 GPT에게 입력하여 최종적으로 매수/매도할 종목을 추천받는 프로그램을 만들었습니다. 
하지만 프로그램에서 출력해준 후보 종목을 GPT에 반복해서 입력하고 결과를 정리하는 과정에서 불편을 느꼈습니다. 그래서 이 과정을 gpt와 제미나이 API로 연결하고, **AI가 판단을 바꾼 이유와 최종 반영 결과를 함께 확인**할 수 있도록 프로그램을 새로 다시 만들었습니다.

첫번째 프로그램은 다음과 같았습니다. [첫 번째 ETF Radar 저장소 보기](https://github.com/petruslihm/first-etf-radar)

**해당 프로그램은 저의 두번째 프로그램이었고, 현재는 머신러닝의 기능을 활용하여 퀀트로 후보를 걸러내는 능력을 높인 세번째 프로그램을 개발중에 있고, 현재 개인적으로 사용중에 있습니다.**

[첫번째 프로그램-> 두번째(현재) 프로그램 문제 해결 사례](docs/CASE_STUDIES.md)

## 사용 흐름
0
준비
설정에서 스캔 대상 유니버스와 관심 종목을 설정하고, API 키를 입력하고, 카카오 계정을 연결합니다.

<img width="923" height="677" alt="image" src="https://github.com/user-attachments/assets/cf832dc1-7ccb-408e-9186-465c0fcf2f8f" />

1
Quant — 후보 선별
먼저 가격과 기술 지표를 API로 받아오고, 이 수치들을 분석해 설정된 기준에 맞는 종목들을 추려, 조사할 후보 종목과 초기 배분안을 만듭니다.

<img width="765" height="867" alt="image" src="https://github.com/user-attachments/assets/f3ac86ef-8b3c-4523-9690-18b0d9b11f6b" />

코드를 통해 정량적으로 선별된 후보 순위입니다.

2
Gemini — 근거 조사
후보 종목의 기업 정보와 위험 요인을 조사하고 초기 배분안을 보정합니다.

<img width="933" height="638" alt="image" src="https://github.com/user-attachments/assets/45e80d03-9273-400c-945e-bc2ec720c82b" />

<img width="928" height="578" alt="image" src="https://github.com/user-attachments/assets/5e9e6aa8-ecbe-4838-8e41-86bb9c86c3bb" />

제미나이 API에게 보내는 원문과 시스템 프롬프트 중 일부입니다. 

<img width="923" height="555" alt="image" src="https://github.com/user-attachments/assets/197236ca-e1dc-4304-b7bf-6b0bd6f0db0d" />

제미나이 API에게서 온 답변중 일부입니다.

3
GPT — 최종 판단
Quant 결과와 Gemini 조사 내용을 함께 검토해 종목별 행동, 순위, 목표 배분을 구조화된 형식으로 제시합니다.

<img width="971" height="670" alt="image" src="https://github.com/user-attachments/assets/e2ca73bd-e29b-4195-8c70-1e61d6632256" />


GPT API에게 보내는 프롬프트 중 일부입니다.

<img width="901" height="738" alt="image" src="https://github.com/user-attachments/assets/ddcd9a28-c3cd-4358-b0b7-619bc30bcbbb" />

GPT API에게서 온 답변중 일부입니다.

4
코드 검증 — 화면 반영
GPT 결과에 종목 누락, 잘못된 행동, 비정상적인 배분이 없는지 검사합니다. 종목별 배분 한도도 적용한 뒤 effective라는 최종 결과를 화면에 표시합니다.

5
각 단계의 판단과 화면 결과를 같은 입력 ID로 저장합니다. 이를 통해 사용자는 **어느 단계에서 행동·순위·배분이 바뀌었는지 비교**하고, 저장된 결과를 다시 열어볼 수 있습니다. GPT 실패 시에는 Gemini 반영 결과로 대체하고 그 출처를 표시하도록 설계했습니다. 

<img width="1457" height="557" alt="image" src="https://github.com/user-attachments/assets/67630f7c-0690-4240-a111-c9954c6ecf3b" />


## 부가적인 설계

### 1. API 연결 확인
API가 연결되지 않았는데 실행을 시킴으로써 시간과 토큰을 허비하는 문제점이 있었습니다. 저는 이를 최소화하기 위해, 먼저 가장 작은 단위의 토큰을 쓰는 매우 간단한 핑테스트를 통해 연결이 되었는지 확인하는 기능을 만들었습니다. 

<img width="767" height="637" alt="image" src="https://github.com/user-attachments/assets/45e816b4-b17e-4f7d-8ead-a1a1b9e26e7e" />

<img width="1007" height="797" alt="image" src="https://github.com/user-attachments/assets/c5551ac2-ebd6-4c60-b529-14691985c554" />

### 2. 실패와 모르는 값을 명확하게 표시

GPT 호출 실패나 잘못된 응답은 대체 결과의 출처와 함께 표시합니다. 평가할 수 없는 보유 자산은 **미보유 0으로 바꾸지 않고 미확정 값(`null`)으로 유지**해, 입력 누락을 정상 판단으로 오인하지 않도록 합니다.

### 3.

| 확인할 내용 | 코드 · 테스트 |
|---|---|
| 판단 검증·배분 제약 | [판단 처리](src/trading_system/final_decision.py) · [데모 테스트](tests/test_submission_demo.py) |
| 실패 처리·미확정 보유 | [추천 데이터](src/trading_system/recommendations.py) · [파이프라인 테스트](tests/test_v1_pipeline.py) |
| 가격 시점·출처 | [가격 데이터](src/trading_system/market/decision_data.py) · [시점 테스트](tests/test_price_provenance.py) |
| 단계별 저장·복원 | [기록 저장](src/trading_system/storage/ticks.py) · [구조 설명](docs/ARCHITECTURE.md) |
| 모델 평가·출처 표시 | [평가 무결성 테스트](tests/test_evaluation_integrity.py) · [평가 범위](docs/EVALUATION.md) |

</details>

### 4. 포트폴리오 기능

 개인의 금액 보안을 위해, Units는 사용자 정의 배분단위로, 주식 수나 달러 금액을 뜻하지 않습니다. 예를 들어 실제 금액이 1억원이고, A 종목 투자 금액이 2천만원이라면, 사용자는 총 Units을 100, A units을 20으로 설정할 수 있게 설계했습니다. 이때 포트폴리오에 들어가있는 종목은 무조건 후보에 포함되어 분석하도록 하고, 포트폴리오를 입력할때 해당 종목의 티커가 API로부터 받아온 목록에 있는지 확인하도록 하였습니다.
 
 <img width="936" height="697" alt="image" src="https://github.com/user-attachments/assets/f7252bf5-9391-4424-bf5d-78cfb412db9c" />

### 5. 분석 입력 고정(Freeze)과 중단 재개

GPT에 보낸 후보 종목·계좌·가격·조사 결과를 하나의 입력 묶음으로 저장합니다. 호출이 중단되면 이 입력을 그대로 사용해 이어가므로, 재시도 도중 바뀐 점수나 새로운 조사 결과가 이전 대화에 섞이지 않도록 합니다. 가격·보유 상태·배분 기준 등이 달라지면 같은 입력으로 재개하지 않고 다시 평가합니다.

### 6. 여러 머신러닝 모델과 온라인 학습

Ridge·LightGBM·LSTM·온라인 SGD로 5·10·20일 구간의 수익률을 예측하고 결과를 결합합니다. 미국 주식은 LambdaRank를 보조 순위 신호로 사용합니다. 결과를 확인할 수 있는 데이터가 쌓이면 온라인 모델을 갱신하고, 최근 오차를 바탕으로 모델과 예측 기간의 반영 비중을 조정합니다. 이미 학습에 사용한 결과는 별도 기록으로 관리해 중복 적용을 막습니다.

‘모델/학습’ 화면에서 모델별 평가 지표, 가중치 변화, 학습 기록을 확인할 수 있습니다. 평가 표본이 부족하거나 학습·평가 구간이 겹치면 지표를 보류합니다. 이 지표가 실제 투자 수익률을 뜻하지는 않습니다.

## 데모 실행

**[Windows 실행 파일 다운로드](https://github.com/petruslihm/ai-investment-research/releases/download/windows-launcher-v1/AI-Investment-Research.cmd)**

다운로드한 파일을 열면 **Python 3.12 준비 → 패키지 설치 → 실제 프로그램 실행 → 브라우저 열기**가 자동으로 진행됩니다. Windows 10/11 x64용이며, 첫 실행은 인터넷 연결과 설치 시간이 필요합니다. Python·Git·Node.js를 미리 설치할 필요는 없습니다.

**실제 리서치를 실행하려면 아래 API가 모두 필요합니다.** 프로그램이 열리면 설정 화면에 본인 또는 소속 조직에서 발급받은 키를 입력하세요.

| 필수 API | 입력할 값 | 용도 |
|---|---|---|
| Alpaca Market Data | API Key + Secret Key | 종목 시세·거래량 수집 |
| Google Gemini | Gemini API Key | 신규 주식 후보의 근거 조사·배분 보정 |
| OpenAI | OpenAI API Key | GPT 조사·최종 판단 검토 |

키가 없거나 사용 권한·할당량이 부족하면 전체 리서치를 실행할 수 없습니다. 실제 호출에는 제공자별 비용이 발생할 수 있습니다. SEC 공시 조회에는 별도로 연락처 이메일을 설정하며, 카카오 알림은 선택 기능입니다.

[상세 실행 안내](docs/RUNNING.md) — 필수 설정, 분석 순서, 저장 위치, 종료와 문제 해결.

**기술 스택:** Python · FastAPI/Jinja · DuckDB · pandas/NumPy · scikit-learn/LightGBM/PyTorch · pytest · uv
