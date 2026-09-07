# 현재 한계와 후속 검증 기준

이 프로젝트의 검증 대상은 리서치 workflow의 작동과 통제입니다. 테스트 통과, fixture 결과, 모델 진단 지표가 투자 성과나 LLM의 우월성을 입증하지는 않습니다. 아래는 현재 코드를 기준으로 확인한 한계와 앞으로 개선 여부를 판단할 기준입니다. **후속 검증 기준은 아직 완료한 기능이나 결과가 아닙니다.**

## 1. LLM 도입 효과를 검증하는 대조평가는 미완성입니다

현재 scan은 최초 Quant allocation 뒤 신규 주식 후보의 Gemini research 점수를 반영해 배분을 다시 계산할 수 있습니다. 이 결과도 `quant_only`라는 source로 저장하므로, 저장명만으로 순수 무LLM 대조군이라고 해석할 수 없습니다. Research 보정 전 결과는 독립된 비교 대상으로 보존되지 않습니다.

GPT는 서술형 포트폴리오 검토문을 생성합니다. `llm_final` 추천 형식과 저장·비교 지원 코드는 있지만 현행 scan은 이 구조화된 추천을 생성하지 않습니다. 따라서 “Quant와 GPT의 추천 성과를 자동 비교해 LLM 도입 효과를 측정한다”는 설명은 현재 구현보다 넓은 주장입니다.

`outcomes.py`는 만기가 지난 종목의 가격수익률 `(종료 가격 / 시작 가격) - 1`을 기록합니다. 매수·매도·보유 행동, 추천 unit, 현금, 거래비용을 반영하지 않습니다. 같은 tick·종목·horizon의 두 source라면 같은 종목 수익률이므로 이것만으로 override의 경제적 가치를 구분할 수 없습니다.

**후속 검증 기준:** 동일 입력·시점의 순수 Quant, research 반영, GPT 검토 결과를 구분해 보존하고, 행동·배분·현금·비용을 반영한 평가 대상을 먼저 정의합니다. 충분히 만기가 지난 forward 표본으로 비교하고 표본 수와 불확실성을 함께 공개해야 합니다. 리서치 시간 단축·누락 감소·사용자 수정률 같은 업무 효과도 별도로 측정해야 합니다. 현재 이런 개선 수치를 주장하지 않습니다.

근거: [`v1_cycle.py`](../src/trading_system/v1_cycle.py)의 `_run_allocation`, `stock_research`, `llm_recs`; [`allocation.py`](../src/trading_system/allocation.py)의 `research_sizing_multiplier`; [`outcomes.py`](../src/trading_system/outcomes.py)의 `record_matured_outcomes`; [`sec_llm.py`](../src/trading_system/sec_llm.py)의 `score_overrides`.

## 2. Research 최신성 검사와 GPT 재개는 다른 정책입니다

일반 research cache에는 45분 TTL과 마지막 종가 변화 3% 초과 시 무효화가 연결되어 있습니다. Portfolio signature와 market regime 변경 검사도 함수와 테스트에는 있지만, 현행 `research_ticker` 호출은 두 값을 전달하지 않습니다. 포트폴리오나 시장 상황이 달라지면 항상 즉시 무효화된다고 설명할 수 없습니다.

중단된 GPT 검토는 같은 ET 날짜에 저장한 후보·포트폴리오 package를 그대로 재사용합니다. 이때 SEC/Gemini 단계를 건너뛰며, 해당 package에 포함된 research에 45분 TTL이나 현재 가격·포트폴리오 검사를 다시 적용하지 않습니다. 따라서 같은 날짜 안에서도 오래된 입력으로 검토가 재개될 수 있습니다. ET 날짜 경계는 전날 package 재사용을 제한하지만 최신성 전체를 보장하지 않습니다.

**후속 검증 기준:** 재개 직전 입력 시각·가격·보유 맥락을 확인하고, 재사용 또는 새 조사 여부를 명시적인 규칙으로 결정합니다. TTL 경과, 가격 급변, 보유 변경, regime 변경, ET 날짜 전환을 포함하는 전체 scan 테스트로 실제 연결을 확인해야 합니다. “함수 테스트 통과”와 “운영 경로 적용”을 구분합니다.

근거: [`evidence_pack.py`](../src/trading_system/evidence_pack.py)의 `pack_is_fresh`; [`research_agent.py`](../src/trading_system/research_agent.py)의 `research_ticker`; [`openai_judge.py`](../src/trading_system/openai_judge.py)의 `load_pending_package`, `_load_resumable_turns`; [`v1_cycle.py`](../src/trading_system/v1_cycle.py)의 pending package 분기.

## 3. 재개·재시작 후 일부 상태 표시가 실제 수행 내용을 충분히 설명하지 못합니다

현재 scan은 OpenAI key가 없는 경우에도 후보가 있으면 GPT 호출 전에 pending package를 저장할 수 있습니다. 같은 ET 날짜의 다음 scan이 이를 불러오면 SEC/Gemini 조회를 생략합니다. 이 분기에서는 SEC와 extract 상태를 `AVAILABLE`로 설정하므로, 새 조회나 추출이 없었는데도 정상으로 보일 수 있습니다. 따라서 이 경로의 상태 배지만으로 외부 조회가 성공했다고 판단해서는 안 됩니다.

또한 `load_last_ui_snapshot`은 저장 결과에서 상태를 재구성할 때 actionable이 아니면 Quant를 `UNKNOWN`으로 표시할 수 있습니다. 재시작 전의 `NOT_CONFIGURED` 등 상세 상태가 그대로 복원되는 것은 아닙니다.

**후속 검증 기준:** 미설정, 신규 호출 성공, 이전 입력 재사용, 실패를 구분해 저장·복원합니다. Key 없는 첫 scan → 같은 날짜의 두 번째 scan, 중단 후 재개, process 재시작 사례에서 UI 상태가 실제 호출 이력과 일치해야 합니다.

근거: [`v1_cycle.py`](../src/trading_system/v1_cycle.py)의 `save_pending_package` 호출, pending package의 `sec_pack` 생성, `load_last_ui_snapshot`.

## 4. 시점 통제는 일부 구현되어 있지만 완전한 재현성을 보장하지 않습니다

과거 시점의 lookback feature, 주식/BTC의 서로 다른 만기, matured label 제한, 시간순 분할과 purge를 구현했습니다. `decision_epoch`도 모델 참조와 버전을 추천에 연결합니다. 다만 epoch 생성의 일부 값은 전체 상태를 담은 hash가 아닙니다. Online model 참조는 fitted 여부 중심이며 `ensemble_weights_hash`는 모델 hash 조합, `thresholds_hash`는 고정 버전 문자열입니다. 실제 가중치·계수·설정을 모두 고정해 당시 판단을 그대로 재현하는 체계는 아닙니다.

`chronological_split`은 고유 날짜가 매우 적으면 train/test를 같은 행으로 반환하는 예외가 있습니다. 이런 소표본 진단을 엄격한 out-of-sample 성과로 볼 수 없습니다. 현재 시점의 universe를 과거 데이터에 적용하는 경우에는 survivorship bias도 남습니다. 당시 구성 종목이나 이후 데이터 수정까지 완전히 복원한다고 주장하지 않습니다.

**후속 검증 기준:** 실제 모델 상태·가중치·설정·입력 snapshot을 포함하는 재현 실험을 정의합니다. 소표본에는 평가 불가 또는 in-sample임을 표시하고, train/test 날짜와 label 기간의 중첩을 검사해야 합니다. 과거 universe와 가격 조정 이력이 확보되지 않은 분석은 제한을 명시해야 합니다.

근거: [`features.py`](../src/trading_system/features.py), [`ml_engine.py`](../src/trading_system/ml_engine.py)의 `chronological_split`; [`models/registry.py`](../src/trading_system/models/registry.py)의 `DecisionEpochManifest`; [`v1_cycle.py`](../src/trading_system/v1_cycle.py)의 epoch 생성.

## 5. 출처와 반대 근거가 있어도 사실 검증을 대신하지 않습니다

Research는 지지·반대 근거, 출처, 미해결 질문을 구조화합니다. 그러나 `VERIFIED_SOURCE`는 URL이 출처 목록에 있거나 SEC URL인지 확인한 표시입니다. 원문을 독립적으로 읽어 주장의 사실성·인용 적합성·최신성을 검증했다는 의미가 아닙니다. 서로 다른 LLM을 사용해도 오류가 독립적이거나 줄어든다고 보장할 수 없습니다.

Research 배분 보정에 사용하는 품질 점수도 LLM의 자가평가입니다. 영향 범위를 제한했지만 검증된 신뢰도 확률은 아닙니다. 또한 구조화 추천용 `apply_portfolio_gate`는 함수와 테스트가 존재하나 현행 서술형 GPT 경로에는 연결되지 않습니다. 사람이 GPT 검토문의 수치와 근거를 확인해야 합니다.

**후속 검증 기준:** 원문·발행 시각·주장 단위의 인용 대응을 검토할 표본과 판정 기준을 정의합니다. 출처 없음, 원문과 불일치, 오래된 근거, 불확실한 사실을 별도로 집계하고, 출력의 사실 오류 및 사용자 수정률을 측정해야 합니다. 구조화 결과를 도입한다면 형식 검증뿐 아니라 실제 호출 경로의 제약 적용도 확인해야 합니다.

근거: [`judge_package.py`](../src/trading_system/judge_package.py)의 `verified_url_set`, `_mark_claim`; [`research_agent.py`](../src/trading_system/research_agent.py); [`allocation.py`](../src/trading_system/allocation.py)의 `research_sizing_multiplier`; [`portfolio_gate.py`](../src/trading_system/portfolio_gate.py).

## 6. 로컬 리서치 도구의 운영 범위

- **주문 실행 없음:** 추천·검토·알림을 제공하며 체결, 슬리피지, 실제 계좌 수익률을 검증하는 거래 시스템은 아닙니다.
- **Fixture와 live 구분:** Fixture는 경로 재현용 데이터입니다. 이를 이용한 모델 진단과 화면은 live 성과 근거가 아닙니다.
- **비용은 추정치:** 자동 Gemini/OpenAI 호출의 UTC 일일 기본 $15 기준은 다음 호출의 예상 비용을 확인하는 soft threshold입니다. 수동 scan은 자동 예산에 포함되지 않으며 실제 결제 상한을 보장하지 않습니다.
- **외부 서비스 의존:** 시장 데이터, SEC, LLM의 가용성·제공 범위·정보 품질에 의존합니다. Last-known-good은 최신 데이터와 같은 의미가 아닙니다.
- **단일 사용자 환경:** 로컬 사용을 전제로 합니다. 조직 내부의 민감정보를 외부 LLM API에 안전하게 전송하도록 승인·권한·보존 정책을 구현한 기업용 시스템은 아닙니다. Unit은 금액 입력을 줄이는 표현 방식이며 익명화나 정보 유출 방지 수단은 아닙니다.

[실행 흐름과 코드 안내](ARCHITECTURE.md) · [프로젝트 개요](../README.md)
