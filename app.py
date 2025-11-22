import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ==============================
# 0. .env 로드 & API Key 준비
# ==============================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==============================
# 1. 차량 데이터 정의 (관심 차량 3대 예시)
# ==============================

RiskLevel = Literal["낮음", "보통", "높음"]


@dataclass
class CarSpec:
    """Mock 차량 스펙 데이터 (프로토타입용: 사용자가 찜/비교해 둔 후보 3대)"""
    id: int
    name: str
    segment: str          # 예: "소형 SUV", "중형 SUV"
    price: int            # 차량 가격 (만원 단위, 예: 1800 = 1,800만원)
    year: int             # 연식
    mileage_km: int       # 주행거리 (km)
    fuel_efficiency: float  # 연비 (km/L)
    base_insurance_year: int  # 기준 연간 보험료 (원)
    maintenance_risk: RiskLevel  # 정비 리스크 (낮음/보통/높음)
    base_depr_3y: float   # 3년 감가율 (예: 0.30 = 30%)


# 실제 서비스에선 사용자의 관심/비교 목록에서 가져온다고 가정하고,
# 프로토타입에서는 3대를 예시로 고정한다.
CAR_LIST: List[CarSpec] = [
    CarSpec(
        id=1,
        name="2019 소형 SUV A",
        segment="소형 SUV",
        price=1800,
        year=2019,
        mileage_km=60000,
        fuel_efficiency=12.0,
        base_insurance_year=900000,
        maintenance_risk="낮음",
        base_depr_3y=0.30,
    ),
    CarSpec(
        id=2,
        name="2018 소형 SUV B",
        segment="소형 SUV",
        price=1600,
        year=2018,
        mileage_km=80000,
        fuel_efficiency=11.0,
        base_insurance_year=950000,
        maintenance_risk="보통",
        base_depr_3y=0.35,
    ),
    CarSpec(
        id=3,
        name="2017 중형 SUV C",
        segment="중형 SUV",
        price=2200,
        year=2017,
        mileage_km=90000,
        fuel_efficiency=9.0,
        base_insurance_year=1100000,
        maintenance_risk="높음",
        base_depr_3y=0.45,
    ),
]

# ==============================
# 2. 재무/리스크 계산 함수
# ==============================


def calc_monthly_payment(price_krw: int, months: int = 36, annual_rate: float = 0.05) -> int:
    """원리금 귕등 상환 기준 월 납입액"""
    if months <= 0:
        return price_krw

    monthly_rate = annual_rate / 12.0
    if monthly_rate == 0:
        return price_krw // months

    numerator = monthly_rate * (1 + monthly_rate) ** months
    denominator = (1 + monthly_rate) ** months - 1
    monthly_payment = price_krw * numerator / denominator
    return int(monthly_payment)


def calc_monthly_fuel_cost(
    monthly_mileage_km: int,
    fuel_efficiency_km_per_l: float,
    fuel_price_per_l: int = 1800,
) -> int:
    if fuel_efficiency_km_per_l <= 0:
        return 0
    liters = monthly_mileage_km / fuel_efficiency_km_per_l
    return int(liters * fuel_price_per_l)


def calc_monthly_insurance_cost(base_insurance_year: int) -> int:
    return int(base_insurance_year / 12)


def calc_monthly_parking_toll_cost(has_paid_parking: bool) -> int:
    return 50000 if has_paid_parking else 0


def maintenance_risk_to_cost(risk: RiskLevel) -> int:
    if risk == "낮음":
        return 600000
    elif risk == "보통":
        return 1200000
    else:
        return 2000000


def calc_depreciation(price_krw: int, base_depr_3y: float):
    depr_amount = int(price_krw * base_depr_3y)
    depr_rate = base_depr_3y
    return depr_amount, depr_rate


def calc_safety_grade(
    monthly_vehicle_cost: int,
    monthly_income: int,
    depr_rate_3y: float,
    maintenance_risk: RiskLevel,
) -> str:
    """재무 안전도 등급 (A/B/C)"""
    if monthly_income <= 0:
        return "C"

    ratio = monthly_vehicle_cost / monthly_income

    if maintenance_risk == "낮음":
        risk_score = 1
    elif maintenance_risk == "보통":
        risk_score = 2
    else:
        return "C"  # 리스크 "높음"이면 무조건 C

    if ratio <= 0.15 and depr_rate_3y <= 0.35 and risk_score <= 2:
        return "A"
    elif ratio <= 0.20 and depr_rate_3y <= 0.45:
        return "B"
    else:
        return "C"


# ==============================
# 3. LLM 프롬프트 & 호출 (사용자 관점 JSON + 예산 가드레일)
# ==============================

def build_llm_prompt(
    monthly_income: int,
    monthly_fixed_expense: int,
    num_children: int,
    monthly_mileage: int,
    ownership_years: int,
    user_monthly_budget: Optional[int],
    df: pd.DataFrame,
) -> str:
    # 사용자 상황 JSON
    user_context = {
        "monthly_income": monthly_income,
        "monthly_fixed_expense": monthly_fixed_expense,
        "num_children": num_children,
        "monthly_mileage": monthly_mileage,
        "ownership_years": ownership_years,
        # 사용자가 스스로 정한 월 차량 예산(상한). 없으면 null.
        "user_monthly_vehicle_budget": user_monthly_budget,
    }

    # 차량별 핵심 지표 JSON
    cars = []
    for _, row in df.iterrows():
        cars.append(
            {
                "name": row["차량명"],
                "segment": row["차급"],
                "safety_grade": row["재무 안전도"],
                "monthly_cost": int(row["월 차량 지출(원)"]),
                "ratio_str": row["월 소득 대비 비율"],
                "tco_3y": int(row["3년 TCO(원)"]),
                "budget_diff": int(row["예산 대비 차이(원)"]) if "예산 대비 차이(원)" in row else None,
            }
        )

    prompt = f"""
당신은 한국의 중고차 재무·리스크 코치입니다.

이 서비스는 기본적으로
- 30대 중반 직장인 A
- 자녀 2명
- 패밀리카를 고민하는 가장
이라는 페르소나를 대상으로 설계되었습니다.

설명에서는 이 사용자를 항상
"자녀 둘을 둔 30대 직장인 A" 페르소나로 언급해 주세요.
단, 월 소득/지출/비율 등 구체 숫자는 아래 JSON의 실제 값을 사용해야 합니다.

[사용자 상황 JSON]
{json.dumps(user_context, ensure_ascii=False)}

[후보 차량들 JSON]
{json.dumps(cars, ensure_ascii=False)}

위 정보를 바탕으로, 다음과 같은 JSON만 반환하세요. 다른 설명/텍스트/마크다운은 절대 포함하지 마세요.

반환 JSON 스키마:
{{
  "user_summary": "페르소나 A(자녀 둘 30대 직장인)의 재무 상태와 차량 여력을 3~4문장으로 설명한 한국어 문장. 월 소득, 고정 지출, 차량에 쓸 수 있는 적정 비율(예: 10~15%), 자녀 둘이라는 가족 상황을 모두 언급해야 합니다.",
  "grade_overview": "A/B/C 등급이 페르소나 A에게 각각 어떤 의미인지 3~4문장으로 설명한 문장. A 등급은 여유, B는 타협 가능한 선택, C는 위험 신호라는 식으로, 각 등급별 추천 월 지출 비율 범위를 함께 제시하세요.",
  "recommended": [
    {{
      "name": "추천 차량명",
      "reason": "이 차량이 페르소나 A에게 왜 상대적으로 안전한 선택인지 2~3문장으로 설명. 월 지출 비율, 3년 TCO, 재무 안전도 등급 같은 수치를 최소 1개 이상 포함하고, 주말 나들이/아이 통학/비상자금 등 생활 맥락도 함께 언급하세요."
    }}
  ],
  "avoid": {{
    "name": "피하거나 주의할 차량명 (없으면 null)",
    "reason": "왜 피하는 것이 좋은지 2~3문장으로 설명. C 등급, 월 지출 비율이 높음, 3년 TCO가 과도함, 예기치 못한 수리비 등 수치와 스토리를 함께 언급하세요."
  }},
  "advice": "페르소나 A가 패밀리카를 살 때 꼭 기억해야 할 한 문장 조언. 월 소득 대비 몇 %를 차량에 쓰는 것이 건강한지, 그리고 스스로 정한 월 차량 예산을 가드레일로 삼아야 한다는 메시지를 함께 담으세요."
}}

추가 규칙:
- recommended 배열에는 최대 2대까지만 넣으세요.
- 재무 안전도 A/B 등급 중에서 우선 추천하고, C 등급은 추천하지 마세요.
- avoid.name 은 재무 안전도 C 등급 중에서 가장 위험한 차량 1대를 고르거나, 없으면 null 로 설정하세요.
- user_monthly_vehicle_budget 값이 null 이 아니면,
  - user_summary 와 grade_overview, recommended.reason, avoid.reason, advice 에서
    "사용자가 스스로 정한 월 차량 예산"과 후보 차량들의 실제 월 지출의 차이를 반드시 언급하세요.
  - 예: "당초 스스로 정한 예산선보다 매달 8만~10만 원을 더 쓰는 조합"과 같은 표현.
- user_summary, grade_overview, recommended.reason, avoid.reason 에는 반드시
  - (1) 하나 이상의 구체 숫자 (월 지출 %, 3년 TCO, 소득, 예산 차이 등)
  - (2) 페르소나 A의 가족/생활 맥락
  이 두 가지가 모두 등장해야 합니다.
- advice 는 반드시 1문장으로 작성하세요.
- 반드시 위 스키마에 맞는 **유효한 JSON만** 출력하세요. JSON 밖에 다른 텍스트를 넣지 마세요.
"""
    return prompt.strip()


def call_llm(prompt: str) -> dict:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다. (.env 파일을 확인해 주세요)")

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 한국의 중고차 재무 코치입니다. "
                    "페르소나 A(자녀 둘 30대 직장인)의 관점에서, "
                    "수치와 스토리를 함께 사용해 과도한 소비를 막으면서도 현실적인 패밀리카 선택을 돕습니다. "
                    "사용자가 스스로 정한 월 차량 예산이 있다면 그 예산을 가드레일로 삼아, "
                    "후보 차량들이 그 선을 얼마나 넘거나 지키고 있는지 명확하게 설명해야 합니다. "
                    "JSON 형식 요구사항을 엄격히 지키세요."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,
    )

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
        return data
    except Exception:
        return {"_raw": raw}


# ==============================
# 4. UI 헬퍼 (색깔 카드/칩)
# ==============================

def colored_box(title: str, body: str, bg: str, border: str):
    st.markdown(
        f"""
        <div style="
            background-color:{bg};
            border:1px solid {border};
            padding:0.8rem 1rem;
            border-radius:0.6rem;
            font-size:0.9rem;
            margin-bottom:0.7rem;
        ">
            <div style="font-weight:600;margin-bottom:0.25rem;">{title}</div>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def grade_chip(label: str, count: int, items: List[str], bg: str, border: str, text_color: str):
    """등급 요약 칩: 등급/개수 + 해당 차량 이름들까지 표시"""
    if items:
        cars_text = ", ".join(items)
    else:
        cars_text = "해당 차량 없음"

    st.markdown(
        f"""
        <div style="
            background-color:{bg};
            border:1px solid {border};
            color:{text_color};
            padding:0.7rem 1rem;
            border-radius:999px;
            font-size:0.85rem;
            text-align:center;
        ">
            <div style="font-weight:700;">{label}</div>
            <div style="margin-top:0.1rem;">{count}대</div>
            <div style="font-size:0.75rem;margin-top:0.25rem;white-space:normal;">
                <span style="opacity:0.8;">차량: {cars_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def grade_style(grade: str):
    """재무 안전도 등급별 색상 스타일"""
    if grade == "A":
        return "#ecfdf3", "#22c55e", "#166534"  # bg, border, text
    elif grade == "B":
        return "#fffbeb", "#facc15", "#854d0e"
    else:
        return "#fef2f2", "#f97373", "#991b1b"


def format_budget_diff(diff: Optional[int]) -> str:
    if diff is None:
        return ""
    if diff == 0:
        return "내가 정한 예산과 거의 동일한 수준"
    if diff < 0:
        return f"내가 정한 예산보다 **{-diff:,}원 여유**"
    return f"내가 정한 예산보다 **{diff:,}원 초과**"


def car_card(row: pd.Series, user_monthly_budget: Optional[int]):
    """개별 차량을 고객용 카드로 표현 (예산 대비 차이 포함)"""
    bg, border, text = grade_style(row["재무 안전도"])
    diff = None
    if user_monthly_budget is not None:
        diff = int(row["월 차량 지출(원)"]) - user_monthly_budget
    budget_line = format_budget_diff(diff) if user_monthly_budget is not None else ""

    st.markdown(
        f"""
        <div style="
            background-color:{bg};
            border:1px solid {border};
            color:{text};
            padding:0.9rem 1rem;
            border-radius:0.8rem;
            margin-bottom:0.8rem;
            font-size:0.9rem;
        ">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                <div style="font-weight:700;font-size:1rem;">{row["차량명"]}</div>
                <div style="font-weight:700;">재무 안전도 {row["재무 안전도"]}</div>
            </div>
            <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:0.3rem;">
                <div>월 부담: <b>{row["월 차량 지출(원)"]:,}원</b></div>
                <div>소득 대비: <b>{row["월 소득 대비 비율"]}</b></div>
                <div>3년 총비용(TCO): <b>{row["3년 TCO(원)"]:,}원</b></div>
            </div>
            <div style="font-size:0.8rem;opacity:0.85;margin-bottom:0.2rem;">
                차급: {row["차급"]} · 차량가: {row["가격(만원)"]:,}만원 · 3년 감가율: {row["3년 감가율"]}
            </div>
            {f'<div style="font-size:0.8rem;margin-top:0.15rem;">{budget_line}</div>' if budget_line else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================
# 5. Streamlit UI (고객용 화면, 예산 가드레일 포함)
# ==============================

def main():
    st.set_page_config(
        page_title="내 월급과 예산으로, 이 차 괜찮을까? (Prototype)",
        layout="wide",
    )

    st.title("내 월급과 예산으로, 이 차 괜찮을까? ")
    st.caption(
        "엔카에서 이미 골라 둔 후보 차량들을 가져와, "
        "내 월 소득·고정 지출·자녀 수·주행거리·월 차량 예산을 기준으로 "
        "월 부담, 3년 총비용, 재무 안전도 A/B/C를 비교해 주는 프로토타입입니다."
    )

    # ---- 좌측: 내 상황 & 예산 입력 ----
    with st.sidebar:
        st.header("1) 내 상황 입력하기")

        monthly_income = st.number_input(
            "세후 월 소득 (원)",
            min_value=0,
            value=3000000,
            step=100000,
            help="급여에서 실수령액 기준으로 입력해 주세요.",
        )

        monthly_fixed_expense = st.number_input(
            "월 고정 지출 (주거비/통신비/대출 등, 원)",
            min_value=0,
            value=2000000,
            step=100000,
            help="매달 거의 변하지 않는 고정 비용을 합산해서 입력해 주세요.",
        )

        num_children = st.number_input(
            "자녀 수",
            min_value=0,
            max_value=5,
            value=2,
            step=1,
        )

        monthly_mileage = st.number_input(
            "예상 월 주행거리 (km)",
            min_value=0,
            value=800,
            step=100,
            help="출퇴근, 주말 나들이 등을 포함한 월 평균 주행거리를 입력해 주세요.",
        )

        ownership_years = st.selectbox(
            "차를 몇 년 정도 탈 예정인가요?",
            options=[3, 5],
            index=0,
            format_func=lambda y: f"{y}년 정도",
        )

        loan_months = st.slider(
            "할부 기간 (개월)",
            min_value=12,
            max_value=72,
            value=36,
            step=12,
            help="월 납입액 계산에 사용됩니다. 아래 3년 총비용은 비교 기준용입니다.",
        )

        has_paid_parking = st.checkbox(
            "유료 주차비/통행료가 매달 발생해요",
            value=True,
        )

        st.markdown("---")

        user_monthly_budget: Optional[int] = None
        use_budget = st.checkbox("나는 월 차량 예산(상한)을 정해두었어요", value=True)
        if use_budget:
            user_monthly_budget = st.number_input(
                "월 차량 예산 상한 (원)",
                min_value=50000,
                value=400000,
                step=50000,
                help="할부 + 보험 + 기름 + 주차비 등을 모두 포함해서, "
                     "차량에 최대 얼마까지 쓰겠다고 생각하는지 입력해 주세요.",
            )

        if monthly_income > 0:
            free_cash = max(monthly_income - monthly_fixed_expense, 0)
            safe_range_low = int(monthly_income * 0.10)
            safe_range_high = int(monthly_income * 0.15)

            st.markdown("#### 💡 재무 여력 가이드")
            st.write(f"- 매달 남는 돈(대략): **{free_cash:,}원**")
            st.write(
                f"- 일반적으로 차량에 써도 괜찮은 권장 범위: "
                f"**{safe_range_low:,} ~ {safe_range_high:,}원/월** (월 소득의 10~15%)"
            )
            if user_monthly_budget:
                st.write(
                    f"- 내가 정한 월 차량 예산: **{user_monthly_budget:,}원/월** "
                    f"(권장 범위와 얼마나 차이가 나는지도 함께 참고해 보세요.)"
                )
        else:
            st.info("월 소득을 입력하면, 차량에 써도 괜찮은 권장 범위를 보여드립니다.")

        if not OPENAI_API_KEY:
            st.error("⚠️ .env 파일의 OPENAI_API_KEY가 설정되지 않아 AI 설명 기능을 사용할 수 없습니다.")

    # ---- 중앙: 차량별 지표 계산 ----
    st.header("2) 내가 골라 둔 후보 차량들, 숫자로 비교해 보기")

    rows = []
    FUEL_PRICE_PER_L = 1800
    LOAN_RATE = 0.05  # 연 이자율 (5%)

    for car in CAR_LIST:
        price_krw = car.price * 10000

        monthly_payment = calc_monthly_payment(
            price_krw=price_krw,
            months=loan_months,
            annual_rate=LOAN_RATE,
        )
        monthly_fuel = calc_monthly_fuel_cost(
            monthly_mileage_km=monthly_mileage,
            fuel_efficiency_km_per_l=car.fuel_efficiency,
            fuel_price_per_l=FUEL_PRICE_PER_L,
        )
        monthly_insurance = calc_monthly_insurance_cost(car.base_insurance_year)
        monthly_parking_toll = calc_monthly_parking_toll_cost(has_paid_parking)

        monthly_vehicle_cost = monthly_payment + monthly_fuel + monthly_insurance + monthly_parking_toll
        ratio = monthly_vehicle_cost / monthly_income if monthly_income > 0 else 0

        depr_amount_3y, depr_rate_3y = calc_depreciation(
            price_krw=price_krw,
            base_depr_3y=car.base_depr_3y,
        )
        maintenance_cost_3y = maintenance_risk_to_cost(car.maintenance_risk)

        # TCO는 3년 기준 (36개월분 지출)
        tco_3y = monthly_vehicle_cost * 36 + maintenance_cost_3y + depr_amount_3y

        safety_grade = calc_safety_grade(
            monthly_vehicle_cost=monthly_vehicle_cost,
            monthly_income=monthly_income,
            depr_rate_3y=depr_rate_3y,
            maintenance_risk=car.maintenance_risk,
        )

        budget_diff = None
        if user_monthly_budget is not None:
            budget_diff = monthly_vehicle_cost - user_monthly_budget

        rows.append({
            "차량명": car.name,
            "차급": car.segment,
            "가격(만원)": car.price,
            "월 차량 지출(원)": monthly_vehicle_cost,
            "월 소득 대비 비율": f"{ratio * 100:.1f}%",
            "3년 감가액(원)": depr_amount_3y,
            "3년 감가율": f"{depr_rate_3y * 100:.0f}%",
            "3년 정비비(원)": maintenance_cost_3y,
            "3년 TCO(원)": tco_3y,
            "재무 안전도": safety_grade,
            "예산 대비 차이(원)": budget_diff if budget_diff is not None else 0,
        })

    df = pd.DataFrame(rows)

    safety_order = {"A": 0, "B": 1, "C": 2}
    df["_order"] = df["재무 안전도"].map(safety_order)
    df = df.sort_values(by=["_order", "3년 TCO(원)"], ascending=[True, True]).drop(columns=["_order"])

    # ---- 고객용 카드 뷰 ----
    st.markdown("#### ✅ 내 월급·예산 기준으로, 각 후보는 이렇게 보입니다.")

    for _, row in df.iterrows():
        car_card(row, user_monthly_budget)

    # ---- 필요할 때만 숫자 표 열기 ----
    with st.expander("📊 숫자 자세히 보기 (관심 있는 분용)"):
        st.dataframe(df, use_container_width=True)

    # ---- 등급 요약 배지 ----
    st.subheader("3) 등급별로 후보 정리해 보기")

    num_A = (df["재무 안전도"] == "A").sum()
    num_B = (df["재무 안전도"] == "B").sum()
    num_C = (df["재무 안전도"] == "C").sum()

    items_A = df[df["재무 안전도"] == "A"]["차량명"].tolist()
    items_B = df[df["재무 안전도"] == "B"]["차량명"].tolist()
    items_C = df[df["재무 안전도"] == "C"]["차량명"].tolist()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        grade_chip("A (예산 안의 안심 구간)", num_A, items_A, bg="#ecfdf3", border="#22c55e", text_color="#166534")
    with col_b:
        grade_chip("B (조금 타이트하지만 고려 가능)", num_B, items_B, bg="#fffbeb", border="#facc15", text_color="#854d0e")
    with col_c:
        grade_chip("C (예산/소득 대비 무리 가능성 높음)", num_C, items_C, bg="#fef2f2", border="#f97373", text_color="#991b1b")

    if num_C > 0:
        st.warning(
            "C 등급 차량은 지금 재무 상황에서 월 부담과 3년 총비용이 꽤 큰 편입니다. "
            "가능하면 A/B 등급에서 먼저 후보를 골라보고, C 차량은 '감성 소비'인지 한 번 더 생각해 보세요."
        )

    # ---- AI 설명 (수치 + 스토리 요약) ----
    st.subheader("4) AI가 내 예산과 후보들을 함께 정리해 드려요")

    if st.button("AI에게 내 상황 정리받기"):
        with st.spinner("AI가 월 소득·지출과 예산, 후보 차량들을 함께 보고 정리하는 중입니다..."):
            try:
                prompt = build_llm_prompt(
                    monthly_income=monthly_income,
                    monthly_fixed_expense=monthly_fixed_expense,
                    num_children=num_children,
                    monthly_mileage=monthly_mileage,
                    ownership_years=ownership_years,
                    user_monthly_budget=user_monthly_budget,
                    df=df,
                )
                summary = call_llm(prompt=prompt)

                if summary is None:
                    st.error("AI 응답을 해석할 수 없습니다.")
                elif "_raw" in summary:
                    st.error("AI 응답이 JSON 형식이 아니었습니다. 원본 텍스트를 표시합니다.")
                    st.markdown(summary["_raw"])
                else:
                    user_summary = summary.get("user_summary", "")
                    grade_overview = summary.get("grade_overview", "")
                    recommended = summary.get("recommended", [])
                    avoid = summary.get("avoid", {})
                    advice = summary.get("advice", "")

                    # 1) 내 재무 & 예산 상황 요약
                    colored_box(
                        "내 재무 & 예산 상황 요약",
                        user_summary,
                        bg="#eff6ff",
                        border="#3b82f6",
                    )

                    # 2) A/B/C 등급이 의미하는 것
                    colored_box(
                        "A/B/C 등급이 의미하는 것",
                        grade_overview,
                        bg="#fefce8",
                        border="#facc15",
                    )

                    # 3) 내 상황에 더 안전한 선택
                    if recommended:
                        lines = []
                        for r in recommended:
                            name = r.get("name", "")
                            reason = r.get("reason", "")
                            lines.append(f"• **{name}** – {reason}")
                        colored_box(
                            "내 상황에 더 안전한 선택",
                            "<br/>".join(lines),
                            bg="#ecfdf3",
                            border="#22c55e",
                        )
                    else:
                        colored_box(
                            "내 상황에 더 안전한 선택",
                            "추천할 차량이 명확하지 않습니다.",
                            bg="#f1f5f9",
                            border="#cbd5f5",
                        )

                    # 4) 조금 더 조심해서 볼 차량
                    avoid_name = avoid.get("name") if isinstance(avoid, dict) else None
                    avoid_reason = avoid.get("reason", "") if isinstance(avoid, dict) else ""

                    if avoid_name and avoid_name.lower() != "null":
                        body = f"• **{avoid_name}** – {avoid_reason}"
                        colored_box(
                            "조금 더 조심해서 볼 차량",
                            body,
                            bg="#fef2f2",
                            border="#f97373",
                        )
                    else:
                        colored_box(
                            "조금 더 조심해서 볼 차량",
                            "특별히 하나를 집어 피해야 할 차량은 없습니다.",
                            bg="#fdf2ff",
                            border="#e879f9",
                        )

                    # 5) 한 문장 조언
                    colored_box(
                        "한 문장 조언",
                        advice,
                        bg="#f4f4f5",
                        border="#d4d4d8",
                    )

            except Exception as e:
                st.error(f"LLM 호출 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
