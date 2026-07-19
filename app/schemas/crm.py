from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class SalesforceOpportunityEvent(BaseModel):
    Id: str                    # Salesforce Opportunity ID
    Name: str                  # Deal name
    OwnerId: str               # Maps to sales_agent lookup
    StageName: str             # Maps to deal_stage
    CloseDate: str             # ISO date
    Amount: Optional[float] = None # Maps to close_value (may be null)
    CreatedDate: str
    ProductCategory: Optional[str] = "Enterprise Suite" # Helper field if provided
    # Custom fields (populated on write-back, null on inbound)
    AI_Win_Probability__c: Optional[float] = None
    AI_Sales_Cycle_Forecast__c: Optional[int] = None
    AI_Risk_Factors__c: Optional[str] = None
    AI_Recommended_Actions__c: Optional[str] = None

class SalesforceWebhookPayload(BaseModel):
    event: Dict[str, Any]      # {"replayId": int, "createdDate": str}
    sobject: SalesforceOpportunityEvent

class HubSpotDealEvent(BaseModel):
    eventId: int
    subscriptionId: int
    portalId: int
    appId: int
    occurredAt: int            # Unix timestamp ms
    subscriptionType: str      # "deal.propertyChange"
    objectId: int              # HubSpot deal ID
    propertyName: str
    propertyValue: str

class HubSpotWebhookPayload(BaseModel):
    events: List[HubSpotDealEvent]
    # deal properties resolved separately via HubSpot API (simulated here)
    deal_properties: Optional[Dict[str, Any]] = None

class CRMSimulatePayload(BaseModel):
    platform: str              # "salesforce" | "hubspot"
    opportunity_id: str
    sales_agent: str
    product: str
    engage_date: str
    deal_stage: str
