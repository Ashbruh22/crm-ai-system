import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_records=8800):
    np.random.seed(42)
    random.seed(42)

    # Base parameters
    agents = ['Sarah Chen', 'Mike Torres', 'Priya Nair', 'James Wu', 'Anita Rao', 'Carlos Webb', 'Ryan Osei', 'Lisa Park']
    products = ['Starter Pack', 'Analytics Pro', 'Enterprise Suite']
    deal_stages = ['prospecting', 'qualification', 'proposal', 'negotiation', 'closed-won', 'closed-lost']
    
    # Generate Account names
    accounts = [f"Company_{i}" for i in range(1000)]
    
    # Product base values and win rate mod
    product_meta = {
        'Starter Pack': {'base_val': 15000, 'std': 5000, 'win_mod': 0.1},
        'Analytics Pro': {'base_val': 55000, 'std': 15000, 'win_mod': 0.0},
        'Enterprise Suite': {'base_val': 250000, 'std': 80000, 'win_mod': -0.15}
    }
    
    # Agent win rate modifiers
    agent_mod = {
        'Sarah Chen': 0.15, 'Mike Torres': 0.12, 'Ryan Osei': 0.05,
        'James Wu': 0.0, 'Lisa Park': 0.0, 'Carlos Webb': -0.05,
        'Priya Nair': -0.10, 'Anita Rao': -0.12
    }
    
    data = []
    
    start_date = datetime(2023, 1, 1)
    
    for i in range(num_records):
        opp_id = f"OPP_{10000 + i}"
        agent = random.choice(agents)
        product = random.choice(products)
        account = random.choice(accounts)
        
        # Engage date
        offset_days = np.random.randint(0, 700)
        engage_date = start_date + timedelta(days=offset_days)
        
        # Win probability calculation for realistic data correlation
        base_win_prob = 0.40 # 40% initial baseline
        win_prob = base_win_prob + agent_mod[agent] + product_meta[product]['win_mod']
        win_prob += np.random.normal(0, 0.1) # Add some noise
        win_prob = max(0.05, min(0.95, win_prob))
        
        is_won = np.random.random() < win_prob
        
        # If the deal is newly created (engage date very recent), it might still be open
        days_since_engage = (datetime(2025, 6, 1) - engage_date).days
        
        # Cycle length based on product
        if product == 'Starter Pack':
            cycle_mean, cycle_std = 30, 10
        elif product == 'Analytics Pro':
            cycle_mean, cycle_std = 45, 15
        else:
            cycle_mean, cycle_std = 90, 30
            
        cycle_days = int(max(5, np.random.normal(cycle_mean, cycle_std)))
        
        if days_since_engage < cycle_days:
            # Still open
            stage = random.choice(['prospecting', 'qualification', 'proposal', 'negotiation'])
            close_date = None
            is_closed = False
        else:
            # Closed
            stage = 'closed-won' if is_won else 'closed-lost'
            close_date = engage_date + timedelta(days=cycle_days)
            is_closed = True
            
        # Value calculation
        if stage == 'closed-lost':
            close_value = None # Value might be null if lost
        else:
            val_mean = product_meta[product]['base_val']
            val_std = product_meta[product]['std']
            close_value = round(max(val_mean*0.5, np.random.normal(val_mean, val_std)), 2)
            
        data.append({
            'opportunity_id': opp_id,
            'sales_agent': agent,
            'product': product,
            'account': account,
            'deal_stage': stage,
            'engage_date': engage_date.strftime('%Y-%m-%d'),
            'close_date': close_date.strftime('%Y-%m-%d') if close_date else None,
            'close_value': close_value
        })
        
    df = pd.DataFrame(data)
    
    # Introduce ~5% missing values in value and some categorical columns to test imputation pipeline
    num_missing = int(num_records * 0.05)
    df.loc[np.random.choice(df.index, num_missing), 'close_value'] = np.nan
    df.loc[np.random.choice(df.index, num_missing), 'account'] = np.nan
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sales_pipeline.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records at {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()
