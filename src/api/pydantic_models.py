from pydantic import BaseModel, Field
from typing import Optional, Dict

class ModelInput(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: str
    PricingStrategy: int
    FraudResult: int

class ModelOutput(BaseModel):
    transaction_id: str
    risk_probability: float
    risk_category: str
    confidence: Optional[float] = None

# Generic request/response used by the API
class PredictionRequest(BaseModel):
    """A mapping of feature name to value. Use the trained model's `feature_names` to
    order features when sending requests."""
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    risk_label: int  # 0 (low risk) or 1 (high risk)
