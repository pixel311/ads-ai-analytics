import os
import re
import json
from typing import Optional

import google.generativeai as genai
import pandas as pd
import streamlit as st
from google.cloud import bigquery

# -----------------------------
# 기본 설정
# -----------------------------
PROJECT_ID = "ads-analytics-project-493908"
DATASET_ID = "ads_analytics"

# 테이블 매핑 (캐시 테이블 사용 — Sheets 직접 참조 우회)
TABLE_MAP = {
    "kpi_lookup":          f"`{PROJECT_ID}.{DATASET_ID}.mart_ai_query_cache`",
    "channel_compare":     f"`{PROJECT_ID}.{DATASET_ID}.mart_ai_query_cache`",
    "product_compare":     f"`{PROJECT_ID}.{DATASET_ID}.mart_ai_query_cache`",
    "benchmark_compare":   f"`{PROJECT_ID}.{DATASET_ID}.mart_ai_query_cache`",
    "performance_check":   f"`{PROJECT_ID}.{DATASET_ID}.mart_ai_query_cache`",
    "top_performer_query": f"`{PROJECT_ID}.{DATASET_ID}.mart_ai_query_cache`",
    "recommendation":      f"`{PROJECT_ID}.{DATASET_ID}.fact_mix_recommendation_result`",
    "simulation":          f"`{PROJECT_ID}.{DATASET_ID}.mart_mix_simulation_cache`",
    "trend":               f"`{PROJECT_ID}.{DATASET_ID}.mart_channel_monthly_performance_cache`",
}

# 테이블별 허용 컬럼
COLUMN_MAP = {
    f"`{PROJECT_ID}.{DATASET_ID}.mart_ai_query_cache`": """
  advertiser_name, industry_name, channel, media_product, objective_type, primary_kpi,
  impressions, clicks, ctr, cpc, roas, spend_ratio,
  benchmark_ctr, benchmark_cpc, benchmark_roas,
  ctr_diff, cpc_diff, roas_diff""",

    f"`{PROJECT_ID}.{DATASET_ID}.fact_mix_recommendation_result`": """
  recommendation_id, advertiser_name, industry_name, objective_type,
  total_budget, recommended_mix, expected_ctr, expected_cpc, expected_roas,
  confidence_score, recommendation_reason, created_at""",

    f"`{PROJECT_ID}.{DATASET_ID}.mart_mix_simulation_cache`": """
  scenario_id, advertiser_name, industry_name, objective_type,
  total_budget, ratio_google, ratio_meta, budget_google, budget_meta,
  google_pred_ctr, google_pred_cpc, google_pred_roas,
  meta_pred_ctr, meta_pred_cpc, meta_pred_roas,
  predicted_roas, predicted_cpc, predicted_ctr""",

    f"`{PROJECT_ID}.{DATASET_ID}.mart_channel_monthly_performance_cache`": """
  month, advertiser_name, industry_name, channel, media_product,
  objective_type, primary_kpi, spend, impressions, clicks, conversions,
  avg_ctr, avg_cpc, avg_cvr, avg_cpa, avg_roas""",
}

ALLOWED_TABLES = {t.replace("`", "").lower() for t in TABLE_MAP.values()}

MODEL_NAME = "gemini-2.5-flash"
DEFAULT_LIMIT = 100
MAX_QUESTION_LENGTH = 300

st.set_page_config(page_title="광고 데이터 AI 분석", layout="wide")

# -----------------------------
# UI
# -----------------------------
st.title("📊 광고 데이터 AI 분석 MVP")
st.write("질문만 입력하면 AI가 SQL을 생성하고 BigQuery 결과를 보여줍니다.")
st.caption(
    "예시: ABC쇼핑 ROAS 알려줘 / Conversion 캠페인 CTR / benchmark 대비 성과 / "
    "5천만 예산 추천 믹스 / 채널별 월간 트렌드"
)

question = st.text_input(
    "질문 입력",
    placeholder="예: ABC쇼핑 5천만 예산 추천 믹스 / Google vs Meta 시나리오 / 채널별 월간 ROAS"
)

# -----------------------------
# 환경 변수 (로컬) / secrets (Streamlit Cloud)
# -----------------------------
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)

# -----------------------------
# 유틸 함수
# -----------------------------
def normalize_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^```\s*", "", sql)
    sql = re.sub(r"\s*```$", "", sql)
    sql = sql.strip().rstrip(";").strip()
    return sql


def contains_forbidden_keyword(sql: str) -> Optional[str]:
    forbidden_patterns = [
        r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bMERGE\b",
        r"\bDROP\b",   r"\bALTER\b",  r"\bTRUNCATE\b", r"\bCREATE\b",
        r"\bREPLACE\b", r"\bGRANT\b", r"\bREVOKE\b",
        r"\bEXECUTE\b", r"\bCALL\b",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, sql, flags=re.IGNORECASE):
            return pattern
    return None


def is_select_only(sql: str) -> bool:
    sql_stripped = sql.strip().lower()
    return sql_stripped.startswith("select") or sql_stripped.startswith("with")


def references_only_allowed_tables(sql: str) -> bool:
    sql_lower = sql.lower().replace("`", "")
    refs = re.findall(r"\b(?:from|join)\s+([a-zA-Z0-9_.-]+)", sql_lower)
    if not refs:
        return False
    for ref in refs:
        if ref not in ALLOWED_TABLES:
            return False
    return True


def enforce_limit(sql: str, limit: int = DEFAULT_LIMIT) -> str:
    if re.search(r"\blimit\s+\d+\b", sql, flags=re.IGNORECASE):
        return sql
    return f"{sql}\nLIMIT {limit}"


def validate_sql(sql: str) -> tuple[bool, str]:
    if not sql:
        return False, "SQL이 비어 있습니다."
    if not is_select_only(sql):
        return False, "SELECT 쿼리만 허용됩니다."
    forbidden = contains_forbidden_keyword(sql)
    if forbidden:
        return False, f"허용되지 않는 SQL 키워드: {forbidden}"
    if not references_only_allowed_tables(sql):
        return False, "허용된 테이블만 참조해야 합니다."
    return True, ""


# -----------------------------
# BigQuery 클라이언트 (견고한 예외 처리)
# -----------------------------
@st.cache_resource
def get_bq_client() -> bigquery.Client:
    """환경에 따라 BigQuery 클라이언트 반환 (로컬 / Streamlit Cloud 모두 지원)"""
    # Streamlit Cloud: secrets.toml의 gcp_service_account 사용
    try:
        if "gcp_service_account" in st.secrets:
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_info(
                dict(st.secrets["gcp_service_account"]),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            return bigquery.Client(project=PROJECT_ID, credentials=credentials)
    except Exception as e:
        raise RuntimeError(
            f"Streamlit Secrets의 gcp_service_account 설정 오류입니다. "
            f"Settings → Secrets에서 private_key 전체 내용을 확인해주세요.\n\n상세: {e}"
        )

    # 로컬: GOOGLE_APPLICATION_CREDENTIALS 환경변수 사용
    try:
        return bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        raise RuntimeError(
            "BigQuery 인증에 실패했습니다.\n"
            "- 로컬 실행: GOOGLE_APPLICATION_CREDENTIALS 환경변수 확인\n"
            "- Streamlit Cloud: Secrets에 [gcp_service_account] 섹션 추가 필요\n\n"
            f"상세: {e}"
        )


# -----------------------------
# Step 1: 템플릿 로드 (BigQuery)
# -----------------------------
@st.cache_data(ttl=300)
def load_templates() -> list[dict]:
    """mart_question_template_cache에서 템플릿 목록을 로드"""
    client = get_bq_client()
    sql = f"""
        SELECT
            template_id, template_name, question_type,
            required_filter_1, required_filter_2, required_filter_3,
            kpi_field, output_type, sql_group_by
        FROM `{PROJECT_ID}.{DATASET_ID}.mart_question_template_cache`
        ORDER BY template_id
    """
    df = client.query(sql).to_dataframe()
    return df.to_dict(orient="records")


@st.cache_data(ttl=300)
def load_dimension_values() -> dict[str, list[str]]:
    """dim_dimension_values_cache에서 모든 디멘션 마스터 값 로드"""
    client = get_bq_client()
    sql = f"""
        SELECT dim, value
        FROM `{PROJECT_ID}.{DATASET_ID}.dim_dimension_values_cache`
        ORDER BY dim, value
    """
    df = client.query(sql).to_dataframe()
    result: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        result.setdefault(row["dim"], []).append(row["value"])
    return result


def match_dimensions_with_gemini(
    user_question: str,
    dim_values: dict[str, list[str]],
) -> dict[str, Optional[str]]:
    """
    사용자 질문에서 광고주명·채널·미디어상품·목적·KPI를 추출 → 마스터 값과 매칭.
    Returns {dimension_name: matched_value or None}
    """
    if not dim_values:
        return {}

    summary_lines = []
    for dim, values in dim_values.items():
        summary_lines.append(f"- {dim}: {', '.join(values)}")
    summary = "\n".join(summary_lines)

    prompt = f"""
다음은 시스템에 등록된 디멘션별 마스터 값 목록이야:

{summary}

사용자 질문: "{user_question}"

위 질문에서 각 디멘션에 해당하는 값이 언급되었는지 판단하고,
언급됐다면 마스터 목록에서 가장 유사한 값을 찾아줘.

매칭 시 유의사항:
- 약어, 줄임말, 영문/한글 표기 차이 모두 고려 (예: "현대차"→"현대자동차", "kt"→"KT")
- 동의어, 변형 표현 모두 매칭 (예: "디멘드젠"→"DemandGen", "디맨드젠"→"DemandGen")
- 사용자 의도가 명확하지 않으면 null 반환

아래 JSON 형식으로만 응답해. 다른 말은 하지 마.
{{
  "advertiser_name": "정확한_값_또는_null",
  "channel": "정확한_값_또는_null",
  "media_product": "정확한_값_또는_null",
  "objective_type": "정확한_값_또는_null",
  "primary_kpi": "정확한_값_또는_null"
}}
""".strip()

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction="너는 광고 데이터 디멘션 매칭 시스템이다. JSON만 출력한다.",
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=0),
    )

    raw = response.text.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
        # null 문자열 → None 변환
        return {k: (v if v and v != "null" else None) for k, v in result.items()}
    except Exception:
        return {}


# -----------------------------
# Step 2: Gemini로 question_type 분류
# -----------------------------
def classify_with_gemini(user_question: str, templates: list[dict]) -> tuple[str, str]:
    template_summary = "\n".join([
        f"- template_id={t['template_id']}, question_type={t['question_type']}, "
        f"template_name={t['template_name']}, "
        f"filters={t['required_filter_1']}/{t['required_filter_2']}/{t['required_filter_3']}"
        for t in templates
    ])

    prompt = f"""
아래는 광고 데이터 분석 시스템의 질문 템플릿 목록이야.

{template_summary}

사용자 질문: "{user_question}"

위 템플릿 중 사용자 질문과 가장 잘 맞는 템플릿을 하나 골라서
아래 JSON 형식으로만 응답해. 다른 말은 하지 마.

{{"question_type": "...", "template_id": "..."}}
""".strip()

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction="너는 질문 분류 시스템이다. JSON만 출력한다.",
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=0),
    )

    raw = response.text.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
        return result.get("question_type", "kpi_lookup"), result.get("template_id", "T01")
    except Exception:
        return "kpi_lookup", "T01"


# -----------------------------
# Step 3: question_type → SQL 생성
# -----------------------------
def build_sql_prompt(
    user_question: str,
    question_type: str,
    template: dict,
    matched_dims: Optional[dict] = None,
    skip_dims: Optional[list] = None,
) -> str:
    """
    matched_dims: AI가 매칭한 디멘션 값들 (예: {"advertiser_name": "현대자동차"})
    skip_dims: 폴백 시 제외할 디멘션 리스트 (예: ["objective_type"])
    """
    table = TABLE_MAP.get(question_type, TABLE_MAP["kpi_lookup"])
    columns = COLUMN_MAP.get(table, "")
    group_by_hint = template.get("sql_group_by") or ""
    kpi_field = template.get("kpi_field") or ""

    matched_dims = matched_dims or {}
    skip_dims = skip_dims or []

    # 매칭된 디멘션 값을 명시적으로 SQL 규칙에 주입
    dim_rules = []
    rule_idx = 11
    for dim, value in matched_dims.items():
        if value and dim not in skip_dims:
            dim_rules.append(
                f"{rule_idx}. {dim} 필터는 반드시 정확한 값으로 사용할 것: "
                f"{dim} = '{value}'  "
                f"(사용자 질문의 표기와 다르더라도 이 값을 사용)"
            )
            rule_idx += 1

    if not any(matched_dims.values()):
        dim_rules.append(
            f"{rule_idx}. 디멘션 매칭이 안된 키워드는 LIKE '%키워드%' 패턴 사용 가능"
        )

    dim_hint = "\n" + "\n".join(dim_rules) if dim_rules else ""

    return f"""
너는 광고 데이터 분석가야.
사용자의 자연어 질문을 BigQuery Standard SQL로 변환해.

사용 가능한 테이블: {table}
사용 가능한 컬럼:{columns}

규칙:
1. BigQuery Standard SQL만 사용
2. SQL만 출력 (마크다운·설명 없이)
3. {table} 만 참조할 것
4. SELECT 문만 생성
5. 존재하지 않는 컬럼 사용 금지
6. 기본 LIMIT {DEFAULT_LIMIT}
7. 집계 필요 시 GROUP BY 사용
8. 권장 GROUP BY: {group_by_hint}
9. 핵심 KPI 컬럼: {kpi_field}
10. 날짜 컬럼이 없는 테이블에는 날짜 필터 사용 금지{dim_hint}

그룹화 키워드 매핑 (사용자가 "~별"로 요청하면 해당 컬럼으로 GROUP BY):
- "업종별"        → GROUP BY industry_name (이 경우 SELECT에도 industry_name 반드시 포함)
- "매체별/채널별" → GROUP BY channel
- "상품별"        → GROUP BY media_product
- "광고주별"      → GROUP BY advertiser_name
- "목적별"        → GROUP BY objective_type
- "월별"          → GROUP BY month (해당 컬럼 있을 때만)

집계 컬럼 처리:
- 집계 시 수치형 컬럼은 SUM 또는 AVG로 묶을 것 (impressions/clicks → SUM, ctr/cpc/roas → AVG)
- GROUP BY에 포함되지 않은 컬럼은 SELECT에서 제거할 것

사용자 질문: {user_question}
""".strip()


def generate_sql(
    user_question: str,
    question_type: str,
    template: dict,
    matched_dims: Optional[dict] = None,
    skip_dims: Optional[list] = None,
) -> str:
    prompt = build_sql_prompt(user_question, question_type, template, matched_dims, skip_dims)
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction="너는 SQL만 생성하는 분석 보조 시스템이다.",
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=0),
    )
    sql = response.text or ""
    return normalize_sql(sql)


# -----------------------------
# Step 4: AI 응답 요약
# -----------------------------
SUMMARY_PERSPECTIVE = {
    "kpi_lookup":          "광고주의 핵심 KPI 수치와 성과를 평가하는 관점",
    "channel_compare":     "매체별 KPI 차이와 효율 우위를 비교하는 관점",
    "product_compare":     "미디어 상품별 성과 격차와 투자 효율 관점",
    "benchmark_compare":   "내부 성과와 업계 benchmark 대비 강점/약점 관점",
    "performance_check":   "성과가 양호한지 미흡한지 평가하는 관점",
    "top_performer_query": "상위 성과 채널의 공통 특성과 교훈 관점",
    "recommendation":      "추천 믹스 선택 이유와 기대 효과를 설명하는 관점",
    "simulation":          "시나리오별 예상 KPI 차이와 트레이드오프 관점",
    "trend":               "시계열 추이와 변화 방향을 포착하는 관점",
}


def summarize_result(user_question: str, question_type: str, df: pd.DataFrame) -> str:
    perspective = SUMMARY_PERSPECTIVE.get(question_type, "데이터 분석 관점")

    sample_df = df.head(20)
    data_text = sample_df.to_string(index=False)
    row_count = len(df)

    prompt = f"""
너는 광고 데이터 분석가야. 아래 BigQuery 조회 결과를 광고주 담당자에게 설명해야 해.

[사용자 질문]
{user_question}

[분석 관점]
{perspective}

[조회 결과 — 총 {row_count}행 중 상위 {len(sample_df)}행]
{data_text}

위 데이터를 기반으로 아래 형식으로만 응답해. 마크다운이나 코드블록은 쓰지 말 것.

📊 핵심 요약
- (1줄)
- (1줄)
- (1줄)

💡 실행 제안
- (1~2줄 실행 가능한 액션)

각 불릿은 구체적 수치를 포함할 것. 추측이나 일반론은 배제할 것.
""".strip()

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction="너는 광고 데이터 인사이트 요약 전문가다.",
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.3),
    )
    return (response.text or "").strip()


# -----------------------------
# 결과 표시 레이블
# -----------------------------
TYPE_LABELS = {
    "kpi_lookup":          "📊 KPI 조회",
    "channel_compare":     "📊 매체별 KPI 비교",
    "product_compare":     "📊 상품별 KPI 비교",
    "benchmark_compare":   "📊 Benchmark 비교",
    "performance_check":   "📊 성과 평가",
    "top_performer_query": "📊 상위 성과 조회",
    "recommendation":      "🎯 미디어 믹스 추천",
    "simulation":          "🔬 예산 시나리오 시뮬레이션",
    "trend":               "📈 채널 월간 트렌드",
}


# -----------------------------
# 결과 표시 — 컬럼 포맷 정의
# -----------------------------
# 각 컬럼별 표시 포맷 정의 (원본 데이터는 변경 없음, 표시만 변경)
# - 비율(%)        : ctr 등 → 0.0008 → 0.08%
# - 배수(x)        : roas → 4.0 → 4.00x
# - 천단위 정수    : impressions, clicks 등 → 33,161,266
# - 금액(₩)        : cost, cpc 등 → ₩2,755
# - 차이(diff)     : 양수일 때 + 부호, 음수면 자동
PERCENT_COLUMNS = {
    "ctr", "ctr_diff",
    "benchmark_ctr",
    "spend_ratio",
}
MULTIPLIER_COLUMNS = {
    "roas", "roas_diff",
    "benchmark_roas",
    "expected_roas", "predicted_roas",
}
INTEGER_COLUMNS = {
    "impressions", "clicks", "conversions",
    "video_views",
    "campaign_count", "ad_group_count",
    "ratio_google", "ratio_meta",
}
CURRENCY_COLUMNS = {
    "cost", "cpc", "cpc_diff",
    "benchmark_cpc",
    "spend",
    "expected_cpc", "predicted_cpc",
    "total_budget", "budget_google", "budget_meta",
}


def build_column_config(df: pd.DataFrame) -> dict:
    """DataFrame 컬럼에 맞춰 Streamlit column_config 자동 생성"""
    config = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in PERCENT_COLUMNS:
            config[col] = st.column_config.NumberColumn(col, format="%.2f%%")
        elif col_lower in MULTIPLIER_COLUMNS:
            config[col] = st.column_config.NumberColumn(col, format="%.2fx")
        elif col_lower in INTEGER_COLUMNS:
            config[col] = st.column_config.NumberColumn(col, format="%d")
        elif col_lower in CURRENCY_COLUMNS:
            config[col] = st.column_config.NumberColumn(col, format="₩%d")
    return config


def display_dataframe(df: pd.DataFrame) -> None:
    """포맷이 적용된 DataFrame 표시 (CSV 다운로드는 원본 유지)"""
    # 비율 컬럼은 %로 표시하기 위해 100을 곱한 사본 사용
    display_df = df.copy()
    for col in display_df.columns:
        if col.lower() in PERCENT_COLUMNS and pd.api.types.is_numeric_dtype(display_df[col]):
            display_df[col] = display_df[col] * 100

    st.dataframe(
        display_df,
        use_container_width=True,
        column_config=build_column_config(display_df),
    )

# -----------------------------
# 실행
# -----------------------------
if question:
    if len(question.strip()) == 0:
        st.warning("질문을 입력해 주세요.")
        st.stop()

    if len(question) > MAX_QUESTION_LENGTH:
        st.warning(f"질문은 {MAX_QUESTION_LENGTH}자 이하로 입력해 주세요.")
        st.stop()

    if not api_key:
        st.error("GEMINI_API_KEY가 설정되지 않았습니다.")
        st.stop()

    genai.configure(api_key=api_key)

    try:
        templates = load_templates()
        dim_values = load_dimension_values()

        with st.spinner("질문 유형을 분류하고 있습니다..."):
            q_type, t_id = classify_with_gemini(question, templates)

        matched = next((t for t in templates if t["template_id"] == t_id), templates[0])

        label = TYPE_LABELS.get(q_type, "📊 조회")
        st.info(f"질문 유형: **{label}** (템플릿: {t_id} · {matched['template_name']})")

        # 멀티 디멘션 매칭 (광고주 / 채널 / 미디어 상품 / 목적 / KPI)
        with st.spinner("질문 키워드를 매칭하고 있습니다..."):
            matched_dims = match_dimensions_with_gemini(question, dim_values)

        # 매칭 결과 표시
        active_dims = {k: v for k, v in matched_dims.items() if v}
        if active_dims:
            badge_lines = []
            dim_label_map = {
                "advertiser_name": "광고주",
                "channel":         "채널",
                "media_product":   "미디어 상품",
                "objective_type":  "목적",
                "primary_kpi":     "KPI",
            }
            for dim, val in active_dims.items():
                badge_lines.append(f"{dim_label_map.get(dim, dim)}: **{val}**")
            st.caption("🎯 매칭된 조건 → " + " / ".join(badge_lines))

        # SQL 생성 + 단계별 폴백 (조건 완화)
        bq_client = get_bq_client()

        # 폴백 우선순위: objective_type → primary_kpi → media_product 순서로 제외
        # (너무 좁은 조건부터 점차 풀어줌)
        FALLBACK_ORDER = ["objective_type", "primary_kpi", "media_product", "channel"]

        # 오류 분류: 재시도 가능 vs 불가
        def is_retryable_error(err_msg: str) -> bool:
            """일시적 오류로 판단되어 재시도 가능한 오류인지 분류"""
            err_lower = err_msg.lower()
            retryable_keywords = [
                "rate limit", "429",                    # API 한도
                "timeout", "deadline exceeded",         # 타임아웃
                "no matching signature",                # AI가 생성한 SQL 타입 오류 (재생성 시 해결 가능)
                "unrecognized name",                    # AI가 잘못된 컬럼명 추측
                "syntax error",                         # 문법 오류 (재생성 시 해결 가능)
                "internal", "503", "500",               # 서버 일시 오류
            ]
            non_retryable_keywords = [
                "permission", "denied", "403",          # 권한 (재시도해도 안됨)
                "not found", "does not exist", "404",   # 리소스 없음
                "quota exceeded",                       # 할당량 초과 (대기 필요)
            ]
            for kw in non_retryable_keywords:
                if kw in err_lower:
                    return False
            for kw in retryable_keywords:
                if kw in err_lower:
                    return True
            return False  # 분류 불가 시 안전하게 재시도 안함

        df = None
        used_skip_dims: list = []
        attempted_queries: list = []
        retry_messages: list = []  # 사용자에게 안내할 재시도 메시지

        for skip_count in range(len(FALLBACK_ORDER) + 1):
            skip_dims = [
                d for d in FALLBACK_ORDER[:skip_count]
                if d in active_dims
            ]

            # 한 폴백 단계에서 최대 2회 재시도 (1차 + 1회 재생성)
            MAX_RETRY = 2
            success = False
            last_error = None

            for retry_n in range(MAX_RETRY):
                spinner_msg = (
                    "AI가 SQL을 생성하고 있습니다..." if (skip_count == 0 and retry_n == 0)
                    else f"조건을 완화하여 재조회 중 (제외: {', '.join(skip_dims)})..." if (skip_count > 0 and retry_n == 0)
                    else f"⏳ 일시 오류로 재시도 중... ({retry_n + 1}/{MAX_RETRY})"
                )
                with st.spinner(spinner_msg):
                    raw_sql = generate_sql(question, q_type, matched, matched_dims, skip_dims)
                    safe_sql = enforce_limit(raw_sql, DEFAULT_LIMIT)
                    is_valid, error_message = validate_sql(safe_sql)

                if not is_valid:
                    st.error(f"SQL 안전 규칙 미통과: {error_message}")
                    st.stop()

                try:
                    df = bq_client.query(safe_sql).to_dataframe()
                    attempted_queries.append((skip_dims, safe_sql))
                    success = True

                    # 재시도로 성공한 경우 사용자에게 알림
                    if retry_n > 0:
                        retry_messages.append(
                            f"ℹ️ 1차 시도에서 일시 오류가 발생하여 자동 재시도로 결과를 가져왔습니다."
                        )
                    break

                except Exception as q_err:
                    last_error = str(q_err)
                    attempted_queries.append((skip_dims, safe_sql))

                    # 재시도 가능 여부 판단
                    if not is_retryable_error(last_error):
                        # 재시도 불가 → 즉시 안내하고 종료
                        st.error(
                            f"❌ 다시 실행해도 해결되기 어려운 오류입니다.\n\n"
                            f"**오류 유형**: 권한·리소스·할당량 관련 영구 오류\n\n"
                            f"**상세**: {last_error}\n\n"
                            f"💡 관리자 또는 데이터 담당자에게 문의해 주세요."
                        )
                        st.stop()

                    # 재시도 가능 → 다음 retry 진행 (마지막 retry면 종료)
                    if retry_n == MAX_RETRY - 1:
                        # 모든 재시도 실패
                        retry_messages.append(
                            f"⚠️ 자동 재시도를 시도했으나 일시 오류가 계속 발생합니다. 잠시 후 다시 시도해 주세요."
                        )

            if not success:
                # 모든 재시도 실패 → 폴백 단계로 넘어가지 않고 종료
                st.error(
                    f"❌ {MAX_RETRY}회 재시도에도 쿼리가 성공하지 못했습니다.\n\n"
                    f"**상세 오류**: {last_error}"
                )
                st.stop()

            if df is not None and not df.empty:
                used_skip_dims = skip_dims
                break

            # 폴백할 조건이 더 없으면 종료
            remaining_to_skip = [
                d for d in FALLBACK_ORDER[skip_count:]
                if d in active_dims
            ]
            if not remaining_to_skip:
                break

        # 재시도 메시지 표시
        for msg in retry_messages:
            st.info(msg)

        # 최종 SQL 표시
        st.subheader("생성된 SQL")
        with st.expander("🔍 생성된 SQL 보기 (디버깅용)", expanded=False):
            for idx, (skip_d, sql) in enumerate(attempted_queries):
                if len(attempted_queries) > 1:
                    label_suffix = " (최종)" if idx == len(attempted_queries) - 1 else ""
                    st.caption(
                        f"시도 {idx+1}{label_suffix} — "
                        + (f"제외: {', '.join(skip_d)}" if skip_d else "전체 조건 적용")
                    )
                st.code(sql, language="sql")

        # 폴백 안내
        if used_skip_dims:
            skipped_labels = []
            dim_label_map = {
                "advertiser_name": "광고주",
                "channel":         "채널",
                "media_product":   "미디어 상품",
                "objective_type":  "목적",
                "primary_kpi":     "KPI",
            }
            for d in used_skip_dims:
                v = active_dims.get(d, "")
                skipped_labels.append(f"{dim_label_map.get(d, d)}({v})")
            st.warning(
                f"⚠️ 입력하신 조건 중 **{', '.join(skipped_labels)}** 에 해당하는 데이터를 "
                f"확인하지 못해 해당 조건을 제외하고 결과를 안내드립니다."
            )

        st.subheader("조회 결과")

        if df is None or df.empty:
            if active_dims:
                cond_lines = [f"{k}={v}" for k, v in active_dims.items()]
                st.info(
                    f"입력하신 조건({', '.join(cond_lines)})으로는 데이터를 찾지 못했습니다.\n\n"
                    f"💡 일부 조건을 제거하거나, 다른 광고주/기간으로 다시 질문해 주세요."
                )
            else:
                st.info("조회 결과가 없습니다. 질문 조건을 더 넓게 입력해 보세요.")
        else:
            if q_type == "recommendation" and "recommendation_reason" in df.columns:
                st.success("✅ 최적 미디어 믹스 추천 결과입니다.")
                highlight_cols = ["recommended_mix", "expected_roas", "confidence_score", "recommendation_reason"]
                display_dataframe(df[[c for c in highlight_cols if c in df.columns]])
                with st.expander("전체 컬럼 보기"):
                    display_dataframe(df)
            elif q_type == "simulation" and "predicted_roas" in df.columns:
                st.info("🔬 시나리오별 예상 KPI 결과입니다.")
                display_dataframe(df)
            else:
                display_dataframe(df)

            # AI 요약
            st.subheader("🧠 AI 인사이트 요약")
            with st.spinner("AI가 결과를 요약하고 있습니다..."):
                try:
                    summary = summarize_result(question, q_type, df)
                    if summary:
                        st.markdown(summary)
                    else:
                        st.caption("요약 결과가 비어있습니다.")
                except Exception as summary_err:
                    st.caption(f"요약 생성 중 일시 오류: {summary_err}")

            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="결과 CSV 다운로드",
                data=csv,
                file_name="query_result.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(
            f"오류가 발생했습니다. 질문을 더 구체적으로 입력하거나 "
            f"지원되는 KPI/필터 기준으로 다시 시도해 주세요.\n\n상세 오류: {e}"
        )

else:
    st.info("질문을 입력하면 실행됩니다.")
