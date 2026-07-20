import sys
import os
import time
import argparse
import requests
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="CRM Webhook Simulator")
    parser.add_argument("--platform", type=str, default="both", choices=["salesforce", "hubspot", "both"], help="CRM Platform")
    parser.add_argument("--count", type=int, default=10, help="Number of synthetic deal events to fire")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay in seconds between events")
    parser.add_argument("--opp-ids", type=str, default=None, help="Comma-separated list of opportunity IDs")
    parser.add_argument("--api-url", type=str, default="http://localhost/api/v1", help="Base API URL")
    
    args = parser.parse_args()
    
    # Load dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "sales_pipeline.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Fallback dummy data if CSV missing
        df = pd.DataFrame([
            {"opportunity_id": "OPP_10001", "sales_agent": "Sarah Chen", "product": "Enterprise Suite", "created_date": "2024-01-15", "deal_stage": "proposal"},
            {"opportunity_id": "OPP_10002", "sales_agent": "Mike Torres", "product": "Starter Pack", "created_date": "2024-02-01", "deal_stage": "qualification"}
        ])
        
    if args.opp_ids:
        target_ids = [x.strip() for x in args.opp_ids.split(",")]
        sampled_df = df[df["opportunity_id"].isin(target_ids)]
        if sampled_df.empty:
            sampled_df = df.sample(n=min(args.count, len(df)), replace=True)
    else:
        sampled_df = df.sample(n=min(args.count, len(df)), replace=True if args.count > len(df) else False)
        
    platforms = ["salesforce", "hubspot"] if args.platform == "both" else [args.platform]
    
    processed = 0
    total_prob = 0.0
    cache_hits = 0
    latencies = []
    priorities = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    
    simulate_url = f"{args.api_url}/crm/webhook/simulate"
    
    print(f"Starting CRM Simulation -> Platform: {args.platform} | Count: {args.count} | Delay: {args.delay}s\n")
    
    for idx, (_, row) in enumerate(sampled_df.iterrows()):
        if idx >= args.count:
            break
            
        current_platform = platforms[idx % len(platforms)]
        
        opp_id = str(row.get("opportunity_id", f"OPP_SIM_{idx:03d}"))
        sales_agent = str(row.get("sales_agent", "Sarah Chen"))
        product = str(row.get("product", "Enterprise Suite"))
        engage_date = str(row.get("created_date", row.get("engage_date", "2024-01-15")))
        deal_stage = str(row.get("deal_stage", "proposal"))
        
        payload = {
            "platform": current_platform,
            "opportunity_id": opp_id,
            "sales_agent": sales_agent,
            "product": product,
            "engage_date": engage_date,
            "deal_stage": deal_stage
        }
        
        t0 = time.perf_counter()
        try:
            resp = requests.post(simulate_url, json=payload, timeout=10)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)
            
            if resp.status_code == 200:
                data = resp.json()
                outcome = data.get("outcome", {})
                win_prob = outcome.get("win_probability", 0.0)
                cached = outcome.get("cached", False)
                if cached:
                    cache_hits += 1
                    
                total_prob += win_prob
                cycle_days = data.get("cycle_forecast_days", 14)
                
                # SHAP top risk
                top_factors = outcome.get("shap_explanation", {}).get("top_factors", [])
                neg_factors = [f for f in top_factors if f.get("direction") == "negative"]
                if neg_factors:
                    top_risk = f"{neg_factors[0]['feature']} ({neg_factors[0]['shap']:.2f} SHAP)"
                else:
                    top_risk = "None"
                    
                # Recommendation & Priority
                recs = outcome.get("recommendations", [])
                if recs:
                    rec_action = recs[0].get("action", "No action needed")
                    priority = recs[0].get("priority", "LOW")
                else:
                    rec_action = "No action needed"
                    priority = "LOW"
                    
                priorities[priority] = priorities.get(priority, 0) + 1
                
                risk_level = "HIGH" if win_prob < 0.3 else ("MEDIUM" if win_prob < 0.7 else "LOW")
                cache_str = "HIT" if cached else "MISS"
                
                wb = data.get("write_back", {})
                if current_platform == "salesforce":
                    wb_str = f"AI_Win_Probability__c={wb.get('AI_Win_Probability__c')}, AI_Sales_Cycle_Forecast__c={wb.get('AI_Sales_Cycle_Forecast__c')}"
                else:
                    wb_str = f"ai_win_probability={wb.get('ai_win_probability')}, ai_sales_cycle_forecast={wb.get('ai_sales_cycle_forecast')}"
                    
                print(f"[{current_platform.upper()}] {opp_id} - {sales_agent} / {product}")
                print(f"  -> Win Probability:    {win_prob * 100:.1f}% ({risk_level} risk)")
                print(f"  -> Days to Close:      {cycle_days} days")
                print(f"  -> Top Risk:           {top_risk}")
                print(f"  -> Recommendation:     {rec_action} [{priority}]")
                print(f"  -> Write-back fields:  {wb_str}")
                print(f"  -> Latency:            {int(elapsed_ms)}ms (cache: {cache_str})\n")
                
                processed += 1
            else:
                print(f"[{current_platform.upper()}] {opp_id} Failed with status {resp.status_code}: {resp.text}\n")
        except Exception as e:
            print(f"[{current_platform.upper()}] {opp_id} Error: {e}\n")
            
        time.sleep(args.delay)
        
    avg_prob = (total_prob / processed * 100) if processed > 0 else 0.0
    avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
    
    print("=== CRM SIMULATION COMPLETE ===")
    print(f"Deals processed:     {processed}")
    print(f"Avg win probability: {avg_prob:.1f}%")
    print(f"Cache hits:          {cache_hits} / {processed}")
    print(f"Avg latency:         {int(avg_lat)}ms")
    print(f"CRITICAL priority:   {priorities['CRITICAL']}")
    print(f"HIGH priority:       {priorities['HIGH']}")
    print(f"MEDIUM priority:     {priorities['MEDIUM']}")
    print(f"LOW priority:        {priorities['LOW']}")

if __name__ == "__main__":
    main()
