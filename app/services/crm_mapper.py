SALESFORCE_FIELD_MAP = {
    "Id": "opportunity_id",
    "OwnerId": "sales_agent",        # In real integration: lookup agent name from OwnerId
    "StageName": "deal_stage",
    "CloseDate": "close_date",
    "Amount": "close_value",
    "CreatedDate": "engage_date",    # Proxy for engage_date in this dataset
}

HUBSPOT_PROPERTY_MAP = {
    "dealname": "opportunity_id",
    "hubspot_owner_id": "sales_agent",
    "dealstage": "deal_stage",
    "closedate": "close_date",
    "amount": "close_value",
    "createdate": "engage_date",
}

SALESFORCE_STAGE_MAP = {
    "Prospecting": "prospecting",
    "Qualification": "qualification",
    "Proposal/Price Quote": "proposal",
    "Negotiation/Review": "negotiation",
    "Closed Won": "closed-won",
    "Closed Lost": "closed-lost",
}

HUBSPOT_STAGE_MAP = {
    "appointmentscheduled": "prospecting",
    "qualifiedtobuy": "qualification",
    "presentationscheduled": "proposal",
    "decisionmakerboughtin": "negotiation",
    "closedwon": "closed-won",
    "closedlost": "closed-lost",
}

def map_salesforce_to_internal(sobject_dict: dict) -> dict:
    stage_raw = sobject_dict.get("StageName", "proposal")
    stage_mapped = SALESFORCE_STAGE_MAP.get(stage_raw, stage_raw.lower())
    
    # Extract date part if ISO string
    created_date = sobject_dict.get("CreatedDate", "2024-01-15")
    if "T" in created_date:
        created_date = created_date.split("T")[0]
        
    return {
        "opportunity_id": sobject_dict.get("Id", "OPP_UNKNOWN"),
        "sales_agent": sobject_dict.get("OwnerId", "Sarah Chen"),
        "product": sobject_dict.get("ProductCategory", "Enterprise Suite"),
        "engage_date": created_date,
        "deal_stage": stage_mapped,
        "close_value": sobject_dict.get("Amount") or 0.0,
        "close_date": sobject_dict.get("CloseDate")
    }

def map_hubspot_to_internal(deal_props: dict) -> dict:
    stage_raw = deal_props.get("dealstage", "presentationscheduled")
    stage_mapped = HUBSPOT_STAGE_MAP.get(stage_raw, stage_raw.lower())
    
    create_date = deal_props.get("createdate", "2024-01-15")
    if "T" in create_date:
        create_date = create_date.split("T")[0]
        
    return {
        "opportunity_id": deal_props.get("dealname", "OPP_UNKNOWN"),
        "sales_agent": deal_props.get("hubspot_owner_id", "Sarah Chen"),
        "product": deal_props.get("product", "Enterprise Suite"),
        "engage_date": create_date,
        "deal_stage": stage_mapped,
        "close_value": float(deal_props.get("amount", 0.0) or 0.0),
        "close_date": deal_props.get("closedate")
    }
