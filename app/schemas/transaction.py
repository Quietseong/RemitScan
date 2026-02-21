from pydantic import BaseModel


class Transaction(BaseModel):
    TransactionDT: str
    user_id: str
    recipient: str
    device_id: str
    receiver_bank: str
    receiver_account: str
    ip_address: str
    app_version: str
    region: str
    TransactionAmt: float
    hour: int
    avg_amount_to_bank: float
    recent_transaction_gap: float
    payment_method: str
    intent: str
    authentication: str
    voice_match: str
    is_new_account_for_user: int
    is_nighttime: int
    is_new_device: int
    vpn: int
    rooting: int
