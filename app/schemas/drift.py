from pydantic import BaseModel
from typing import Optional


class DriftMetrics(BaseModel):
    timestamp: str
    psi: Optional[float] = None
    ks_stat: Optional[float] = None
    ks_pval: Optional[float] = None
    js_div: Optional[float] = None
    wasserstein: Optional[float] = None
