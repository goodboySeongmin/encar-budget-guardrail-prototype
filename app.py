import math
import os
from dataclasses import dataclass
from typing import List

import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv  # ✅ dotenv 사용


# ==============================
# 0. 기본 설정 & OpenAI 키 로딩
# ==============================
st.set_page_config(
    page_title="Encar Budget Guardrail – Internal PoC",
    layout="wide",
)

# 로컬(.env) → 클라우드(secrets) 순서로 키 찾기
load_dotenv()  # .env 로드

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error(
        "OPENAI_API_KEY가 설정되어 있지 않습니다.\n"
        ".env 파일 또는 Streamlit Secrets에 키를 추가한 뒤 다시 실행해주세요."
    )
    st.stop()

client = OpenAI(api_key=api_key)

# 커스텀 스타일 (카드 / 등급 뱃지 등)
st.markdown(
    """
    <style>
    .pill {
        display: inline-block;
        padding: 0.35rem 1.2rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
        border: 1px solid rgba(0,0,0,0.08);
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }
    .pill-A { background-color: #e8f8ec; color: #10793f; border-color: #6fd39b; }
    .pill-B { background-color: #fff7e6; color: #b26b00; border-color: #ffd27f; }
    .pill-C { background-color: #ffecec; color: #c2352b; border-color: #ff9b9b; }

    .card {
        border-radius: 12px;
        padding: 1.0rem 1.2rem;
        border: 1px solid rgba(0,0,0,0.06);
        margin-bottom: 0.75rem;
        background-color: #ffffff;
    }
    .card-soft {
        background-color: #f7f9fc;
    }
    .card-danger {
        background-color: #fff5f5;
    }
    .card-safe {
        background-color: #f2fbf5;
    }
    .card-info {
        background-color: #f2f4ff;
    }
    .section-title {
        font-weight: 800;
        font-size: 1.25rem;
        margin-bottom: 0.6rem;
        margin-top: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================
# 1. 데이터 모델 & 유틸 함수
# ==============================
@dataclass
class Car:
    name: str
    year: int
    segment: str
    price: int  # 차량가(원)
    fuel_eff_km_per_l: float
    maintenance_risk: str  # "낮음" / "보통" / "높음"


def get_demo_cars() -> List[Car]:
    """
    데모용 차량 3대.
    실제 엔카 연동 시에는 API 결과를 여기에 매핑하면 됨.
    """
    return [
        Car(
            name="2018 소형 SUV A*",
            year=2018,
            segment="소형 SUV",
            price=18000000,
            fuel_eff_km_per_l=12.0,
            maintenance_risk="낮음",
        ),
        Car(
            name="2019 소형 SUV B**",
            year=2019,
            segment="소형 SUV",
            price=21000000,
            fuel_eff_km_per_l=11.0,
            maintenance_risk="보통",
        ),
        Car(
            name="2017 중형 SUV C**",
            year=2017,
            segment="중형 SUV",
            price=24000000,
            fuel_eff_km_per_l=9.5,
            maintenance_risk="높음",
        ),
    ]


def monthly_loan_payment(price: int, months: int, annual_rate: float = 0.06) -> float:
    """원리금 균등상환 월 납입액 계산 (간단 버전)."""
    if months <= 0:
        return float(price)
    r = annual_rate / 12.0
    if r == 0:
        return price / months
    n = months
    return price * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def estimate_maintenance_cost(risk: str, years: int = 3) -> int:
    """정비 리스크에 따른 3년 정비비 대략치 (단순 가정)."""
    base = 800000  # 3년간 최소 정비비
    if risk == "낮음":
        return base
    elif risk == "보통":
        return int(base * 1.4)
    else:  # 높음
        return int(base * 2.0)


def estimate_depreciation(price: int, years: int = 3) -> int:
    """3년간 감가액 (단순 가정)."""
    rate = 0.35  # 3년간 35% 감가
    return int(price * rate)


def assign_grade(monthly_cost: float, income: float, risk: str) -> str:
    """월 지출/소득 비율과 정비 리스크를 기준으로 A/B/C 등급 부여."""
    ratio = monthly_cost / income if income > 0 else 1.0
    risk_weight = {"낮음": 0, "보통": 1, "높음": 2}.get(risk, 1)

    if ratio <= 0.15 and risk_weight <= 1:
        return "A"
    elif ratio <= 0.2:
        return "B"
    else:
        return "C"


def format_won(x: float) -> str:
    """천 단위 반올림해서 '1,234,000원' 형태로 표시."""
    return f"{int(round(x, -3)):,}원"


# ==============================
# 2. 사이드바 – 사용자 입력
# ==============================
with st.sidebar:
    st.title("1) 내 상황 입력하기")
    st.caption("실제와 비슷하게 입력할수록 결과가 현실적이에요.")

    income = st.number_input("세후 월 소득", min_value=0, step=100000, value=3000000)
    fixed_cost = st.number_input(
        "월 고정 지출(주거비·통신비·기타 대출 등)",
        min_value=0,
        step=100000,
        value=1000000,
    )
    children = st.number_input("자녀 수", min_value=0, max_value=5, step=1, value=2)
    monthly_km = st.number_input("예상 월 주행거리(km)", min_value=0, step=100, value=800)
    fuel_price = st.number_input(
        "리터당 유류비(원)", min_value=1000, max_value=3000, step=50, value=1700
    )
    has_paid_parking = st.checkbox("유료 주차/통행료가 자주 발생한다", value=True)
    loan_months = st.number_input(
        "할부 개월 수", min_value=0, max_value=84, step=6, value=36
    )
    budget = st.number_input(
        "내가 생각하는 월 차량 예산 상한(원)", min_value=0, step=50000, value=600000
    )

    st.markdown("---")
    st.caption("※ 모든 수치는 단순 예시이며 실제 금융/세무 자문이 아닙니다.")

disposable_income = max(income - fixed_cost, 0)


# ==============================
# 3. 메인 – 헤더 및 차량 데이터 계산
# ==============================
st.title("예산 가드레일 코파일럿 ")
st.write(
    "엔카에서 찜해 둔 후보 차량들이 **내 월급·지출·가족 상황** 기준으로 "
    "얼마나 안전한 선택인지 A/B/C 등급과 함께 보여주는 내부 PoC입니다."
)

cars = get_demo_cars()
rows = []

for car in cars:
    loan = monthly_loan_payment(car.price, loan_months) if loan_months > 0 else car.price
    fuel_cost = (
        (monthly_km / car.fuel_eff_km_per_l) * fuel_price
        if car.fuel_eff_km_per_l > 0
        else 0
    )
    parking_cost = 80000 if has_paid_parking else 20000
    monthly_total = loan + fuel_cost + parking_cost
    maint_3y = estimate_maintenance_cost(car.maintenance_risk, years=3)
    dep_3y = estimate_depreciation(car.price, years=3)
    tco_3y = monthly_total * 36 + maint_3y + dep_3y
    ratio = monthly_total / income if income > 0 else 1.0
    grade = assign_grade(monthly_total, income, car.maintenance_risk)
    budget_diff = monthly_total - budget

    rows.append(
        {
            "차량명": car.name,
            "연식": car.year,
            "차급": car.segment,
            "차량가": car.price,
            "월 할부+운행비": monthly_total,
            "월 소득 대비 비율": ratio,
            "3년 TCO": tco_3y,
            "정비 리스크": car.maintenance_risk,
            "재무 안전도 등급": grade,
            "예산 대비 차이": budget_diff,
        }
    )

df = pd.DataFrame(rows)

# 가이드라인 계산 (월 소득의 10~15%)
guideline_low = income * 0.10
guideline_high = income * 0.15
min_monthly = df["월 할부+운행비"].min()
max_monthly = df["월 할부+운행비"].max()


# ==============================
# 4. 내 재무 스냅샷
# ==============================
st.markdown('<div class="section-title">2) 내 재무 스냅샷</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown(
        f"""
        <div class="card card-info">
          <div style="font-weight:700; margin-bottom:0.4rem;">현재 가계 현황</div>
          <div style="font-size:0.9rem;">
            • 세후 월 소득: <b>{format_won(income)}</b><br>
            • 월 고정 지출: <b>{format_won(fixed_cost)}</b><br>
            • 월 가용 소득(여윳돈): <b>{format_won(disposable_income)}</b><br>
            • 내가 정한 월 차량 예산 상한: <b>{format_won(budget)}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown(
        f"""
        <div class="card card-soft">
          <div style="font-weight:700; margin-bottom:0.4rem;">차량 지출 가이드라인 vs 후보</div>
          <div style="font-size:0.9rem; margin-bottom:0.3rem;">
            • 일반적인 권장 차량 지출 가이드라인: <b>월 소득의 10~15%</b><br>
            → 이 사용자 기준: <b>{format_won(guideline_low)} ~ {format_won(guideline_high)}</b>
          </div>
          <div style="font-size:0.9rem;">
            • 현재 후보 차량 월 지출 범위: <b>{format_won(min_monthly)} ~ {format_won(max_monthly)}</b><br>
            • 가장 부담 큰 차량을 선택하면, 월 소득의 약 <b>{df['월 소득 대비 비율'].max()*100:.1f}%</b>를 차량에 쓰게 됩니다.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================
# 5. 등급 요약 섹션
# ==============================
st.markdown('<div class="section-title">3) 재무 안전도 등급 요약</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

for grade, col in zip(["A", "B", "C"], [col_a, col_b, col_c]):
    subset = df[df["재무 안전도 등급"] == grade]
    count = len(subset)
    if grade == "A":
        desc = "재무적인 여유가 있는 선택입니다."
        css = "pill pill-A"
    elif grade == "B":
        desc = "관리 가능한 타협 구간입니다."
        css = "pill pill-B"
    else:
        desc = "장기적으로 재무 부담 리스크가 있는 구간입니다."
        css = "pill pill-C"

    with col:
        st.markdown(
            f"""
            <div class="card card-soft">
              <div class="{css}" style="margin-bottom:0.5rem;">{grade} 등급 · {count}대</div>
              <div style="font-size:0.9rem; margin-bottom:0.4rem;">{desc}</div>
              <div style="font-size:0.85rem; color:#555;">
            """,
            unsafe_allow_html=True,
        )
        if count == 0:
            st.markdown("해당 등급 차량이 없습니다.", unsafe_allow_html=True)
        else:
            for name in subset["차량명"].tolist():
                st.markdown(f"• {name}", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

# C 등급 경고
if (df["재무 안전도 등급"] == "C").any():
    st.markdown(
        """
        <div class="card card-danger">
        ⚠️ <b>C 등급 차량</b>은 월 소득 대비 지출 비율이 높거나 정비 리스크가 커서,
        장기적으로 재무 부담 및 불만(환불·계약 변경) 가능성이 있는 구간입니다.
        실제 상담 시 예산 재조정 또는 대안 제시가 권장됩니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================
# 6. 차량별 상세 카드
# ==============================
st.markdown(
    '<div class="section-title">4) 후보 차량별 재무 관점 상세</div>',
    unsafe_allow_html=True,
)

for _, row in df.iterrows():
    grade = row["재무 안전도 등급"]
    if grade == "A":
        css = "card card-safe"
    elif grade == "B":
        css = "card card-soft"
    else:
        css = "card card-danger"

    budget_txt = "예산 이내" if row["예산 대비 차이"] <= 0 else "예산 초과"
    budget_detail = (
        f"{format_won(abs(row['예산 대비 차이']))} "
        f"{'여유' if row['예산 대비 차이'] < 0 else '초과'}"
        if budget > 0
        else "예산 상한 미입력"
    )

    st.markdown(
        f"""
        <div class="{css}">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
            <div style="font-weight:700;">{row['차량명']}</div>
            <div class="pill pill-{grade}">{grade} 등급</div>
          </div>
          <div style="font-size:0.9rem; margin-bottom:0.3rem;">
            {row['연식']}년식 · {row['차급']} · 차량가 {format_won(row['차량가'])}
          </div>
          <div style="font-size:0.9rem;">
            • 월 차량 지출(할부+유류비+주차): <b>{format_won(row['월 할부+운행비'])}</b><br>
            • 월 소득 대비: <b>{row['월 소득 대비 비율']*100:.1f}%</b><br>
            • 3년 총비용(TCO): <b>{format_won(row['3년 TCO'])}</b><br>
            • 정비 리스크: <b>{row['정비 리스크']}</b><br>
            • 예산 기준: <b>{budget_txt}</b> ({budget_detail})
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================
# 7. AI 분석 요약
# ==============================
st.markdown('<div class="section-title">5) AI 재무 코치 한눈 요약</div>', unsafe_allow_html=True)

if st.button("AI 재무 코치에게 설명 받기", type="primary"):
    with st.spinner("AI가 내 재무 상황과 차량 조합을 분석하고 있습니다..."):
        # LLM에 넘길 요약 데이터 준비
        user_profile = {
            "월_소득": income,
            "월_고정지출": fixed_cost,
            "자녀수": children,
            "월_주행거리_km": monthly_km,
            "월_차량_예산상한": budget,
            "할부개월수": loan_months,
            "가용소득": disposable_income,
        }

        cars_payload = []
        for _, row in df.iterrows():
            cars_payload.append(
                {
                    "차량명": row["차량명"],
                    "연식": int(row["연식"]),
                    "차급": row["차급"],
                    "재무_안전도_등급": row["재무 안전도 등급"],
                    "월_차량지출": int(row["월 할부+운행비"]),
                    "월_소득_대비_비율": float(row["월 소득 대비 비율"]),
                    "삼년_TCO": int(row["3년 TCO"]),
                    "예산_대비_차이": int(row["예산 대비 차이"]),
                    "정비_리스크": row["정비 리스크"],
                }
            )

        system_msg = {
            "role": "system",
            "content": (
                "너는 중고차 구매를 돕는 재무 코치이자 엔카 상담 보조 도구야. "
                "사용자의 월 소득/지출, 자녀 수, 예산 상한을 고려해서 "
                "각 차량 조합이 얼마나 안전한 선택인지 솔직하지만 균형 있게 설명해줘. "
                "단, '절대 사지 마세요'처럼 단정 짓기보다는 "
                "'이 정도면 장기적으로 부담이 될 수 있다' 수준의 톤으로 조언해."
            ),
        }

        user_msg = {
            "role": "user",
            "content": (
                "다음은 한 사용자의 재무/생활 상황과 엔카에서 고른 후보 차량 목록이야.\n\n"
                f"[사용자 프로필]\n{user_profile}\n\n"
                f"[후보 차량 목록]\n{cars_payload}\n\n"
                "아래 형식으로 한국어로 설명해줘.\n"
                "1) 사용자 상황 요약 (월 소득, 고정 지출, 가용 소득, 자녀 수, 예산 상한 중심)\n"
                "2) 등급별 해석: A/B/C 등급이 각각 이 사용자에게 어떤 의미인지 (숫자+생활 맥락)\n"
                "3) 추천 차량: 상대적으로 재무 부담이 적은 차량 1~2대와 그 이유\n"
                "4) 주의해서 봐야 할 차량: C 등급 차량이 있다면 왜 부담일 수 있는지\n"
                "5) 한 문장 조언: '이 사용자는 월 소득의 몇 %를 차량에 쓰는 것을 가드레일로 삼으면 좋다'처럼 한 줄로 정리\n"
                "가능하면 불릿 포인트를 적절히 사용해서 가독성 좋게 써줘."
            ),
        }

        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[system_msg, user_msg],
                temperature=0.35,
            )
            ai_text = completion.choices[0].message.content
            st.markdown(
                f"""
                <div class="card">
                {ai_text}
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"LLM 호출 중 오류가 발생했습니다: {e}")
else:
    st.info(
        "왼쪽에 내 상황을 입력한 뒤, 위 버튼을 눌러 AI 재무 코치에게 "
        "후보 차량 조합을 한눈에 설명받을 수 있습니다."
    )
