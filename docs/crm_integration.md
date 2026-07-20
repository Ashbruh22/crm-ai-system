# CRM Integration Guide

This document outlines how the CRM AI System connects with Salesforce and HubSpot CRM platforms, both via live webhooks and via the built-in webhook simulator.

---

## 1. Webhook Endpoints & Architecture

The API exposes three CRM integration endpoints under `/api/v1/crm/`:

| Endpoint | Method | Authentication | Purpose |
|---|---|---|---|
| `/api/v1/crm/webhook/salesforce` | `POST` | `X-Salesforce-Signature` (HMAC-SHA256) | Production endpoint for Salesforce Streaming API outbound webhooks |
| `/api/v1/crm/webhook/hubspot` | `POST` | `X-HubSpot-Signature-v3` (HMAC-SHA256) | Production endpoint for HubSpot deal property change subscriptions |
| `/api/v1/crm/webhook/simulate` | `POST` | None (controlled by `CRM_SIMULATE_ENABLED`) | Dev/demo simulation endpoint for testing deal pipelines |

---

## 2. Salesforce Developer Org Setup (Step-by-Step)

To integrate a real Salesforce org with this API:

### Step 1: Create Custom Fields on Opportunity
Navigate to **Setup -> Object Manager -> Opportunity -> Fields & Relationships** and create 4 custom fields:
- `AI_Win_Probability__c` (Percent or Number, 3 decimal places)
- `AI_Sales_Cycle_Forecast__c` (Number, 0 decimal places)
- `AI_Risk_Factors__c` (Text area or Long Text Area)
- `AI_Recommended_Actions__c` (Long Text Area)

### Step 2: Configure Connected App / Remote Site
- Go to **Setup -> Security -> Remote Site Settings** and add your public server URL (or `ngrok` URL during development, e.g. `https://your-ngrok-id.ngrok-free.app`).

### Step 3: Create Apex Trigger & Outbound Callout
Create an Apex Trigger on `Opportunity` (after update / after insert):
```apex
trigger OpportunityAIWebhookTrigger on Opportunity (after insert, after update) {
    for (Opportunity opp : Trigger.new) {
        // Asynchronously call out to /api/v1/crm/webhook/salesforce
        CRMWebhookCallout.sendOpportunityEvent(opp.Id);
    }
}
```

In your callout utility, sign the payload using `HMAC-SHA256` with your `CRM_WEBHOOK_SECRET` and attach header `X-Salesforce-Signature: sha256=<signature>`.

---

## 3. HubSpot Account Setup (Step-by-Step)

To integrate a HubSpot Account:

### Step 1: Create Custom Deal Properties
Navigate to **Settings -> Properties -> Deal Properties** and create 4 custom properties:
- `ai_win_probability` (Single-line text or Number)
- `ai_sales_cycle_forecast` (Single-line text or Number)
- `ai_risk_factors` (Multi-line text)
- `ai_recommended_actions` (Multi-line text)

### Step 2: Create a HubSpot Private App
- Go to **Settings -> Integrations -> Private Apps**.
- Click **Create a private app**, title it `CRM AI System`.
- Under **Scopes**, select `crm.objects.deals.read` and `crm.objects.deals.write`.

### Step 3: Configure Webhook Subscription
- Under **Webhooks**, set the Target URL to `http://<your-host>/api/v1/crm/webhook/hubspot`.
- Create a subscription for event type `deal.propertyChange`.
- Copy your App Client Secret into `.env` as `CRM_WEBHOOK_SECRET`.

---

## 4. Running the Simulator

For demonstrations and offline development, use the `scripts/crm_simulator.py` tool.

```bash
# Run simulator for both platforms (10 synthetic deals)
python scripts/crm_simulator.py --platform both --count 10 --delay 2

# Run simulator specifically for Salesforce with specific deal IDs
python scripts/crm_simulator.py --platform salesforce --count 5 --opp-ids OPP_10001,OPP_10002
```

---

## 5. Security & Environment Configuration

Set the following variables in your `.env` file:

```env
# HMAC Webhook Secret shared with CRM platforms
CRM_WEBHOOK_SECRET=your-hmac-secret-here

# Enable or disable /simulate endpoint (Set false in production)
CRM_SIMULATE_ENABLED=true
```
