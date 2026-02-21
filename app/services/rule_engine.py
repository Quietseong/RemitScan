"""규칙 기반 이상탐지 엔진: 6개 규칙."""
import pandas as pd


RULES = [
    {
        "name": "야간_대량_송금",
        "condition": lambda r: r["is_nighttime"] == 1 and r["TransactionAmt"] > 500_000,
    },
    {
        "name": "VPN_대량_송금",
        "condition": lambda r: r["vpn"] == 1 and r["TransactionAmt"] > 300_000,
    },
    {
        "name": "약인증_대량_송금",
        "condition": lambda r: r["authentication"] in ("A01", "A02") and r["TransactionAmt"] > 500_000,
    },
    {
        "name": "신규기기_대량_송금",
        "condition": lambda r: r["is_new_device"] == 1 and r["TransactionAmt"] > 300_000,
    },
    {
        "name": "루팅_탐지",
        "condition": lambda r: r["rooting"] == 1,
    },
    {
        "name": "신규계정_야간_VPN",
        "condition": lambda r: (
            r["is_new_account_for_user"] == 1
            and r["is_nighttime"] == 1
            and r["vpn"] == 1
        ),
    },
]


def evaluate(df: pd.DataFrame) -> list[list[str]]:
    """각 거래에 대해 트리거된 규칙 이름 목록을 반환."""
    results: list[list[str]] = []
    for _, row in df.iterrows():
        triggered = [r["name"] for r in RULES if r["condition"](row)]
        results.append(triggered)
    return results
