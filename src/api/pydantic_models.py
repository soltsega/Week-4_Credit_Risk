from pydantic import BaseModel
from typing import Optional

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