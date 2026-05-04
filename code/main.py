import pandas as pd
import os
from indexer import HybridIndexer
from agent import SupportAgent
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def run_single_issue(issue: str):
    # Initialize Indexer and Agent
    data_dir = "data"
    if not os.path.exists(data_dir):
        data_dir = "../data"
    
    indexer = HybridIndexer()
    indexer.load_data(data_dir)
    indexer.fit()
    
    agent = SupportAgent()
    
    # Retrieval
    context = indexer.hybrid_search(issue, top_k=3)
    
    # Agent Processing
    res = agent.process_ticket(issue, "", "None", context)
    
    # Output 5 JSON fields
    import json
    print("\n" + "="*30)
    print("SINGLE TICKET OUTPUT")
    print("="*30)
    print(json.dumps(res, indent=4))
    print("="*30)

def main(input_file="support_tickets/support_tickets.csv", output_file="support_tickets/output.csv"):
    # 1. Initialize Indexer and Agent
    print("Initializing indexing...")
    # Assume we run from repo root
    data_dir = "data"
    if not os.path.exists(data_dir):
        # Try parent if we are in code/
        data_dir = "../data"
    
    indexer = HybridIndexer()
    indexer.load_data(data_dir)
    indexer.fit()
    
    agent = SupportAgent()
    
    # 2. Load Tickets
    if not os.path.exists(input_file):
        # Try relative to repo root if current is different
        alt_input = os.path.join("..", input_file)
        if os.path.exists(alt_input):
            input_file = alt_input
            output_file = os.path.join("..", output_file)
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} tickets to process from {input_file}.")
    
    # 3. Process Tickets
    results = []
    checkpoint_file = output_file + ".tmp"
    
    # Load existing progress if any
    if os.path.exists(checkpoint_file):
        progress_df = pd.read_csv(checkpoint_file)
        results = progress_df.to_dict('records')
        print(f"Resuming from checkpoint: {len(results)} tickets already processed.")
    
    start_idx = len(results)
    
    for index, row in tqdm(df.iloc[start_idx:].iterrows(), total=len(df)-start_idx):
        issue = row['Issue']
        subject = row.get('Subject', '')
        company = row.get('Company', 'None')
        
        # Retrieval
        context = indexer.hybrid_search(issue, company=company, top_k=3)
        
        # Agent Processing
        res = agent.process_ticket(issue, str(subject), str(company), context)
        
        # Store results
        ticket_result = {
            'Issue': issue,
            'Subject': subject,
            'Company': company,
            'status': res.get('status', 'escalated'),
            'product_area': res.get('product_area', ''),
            'response': res.get('response', ''),
            'justification': res.get('justification', ''),
            'request_type': res.get('request_type', 'product_issue')
        }
        results.append(ticket_result)
        
        # Save checkpoint after each ticket
        pd.DataFrame(results).to_csv(checkpoint_file, index=False)
        
        # Rate limiting: wait for 5 seconds (respecting Groq limits)
        import time
        time.sleep(5)
    
    # 4. Save Final Results
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    
    if len(args) == 0:
        # Default CSV mode
        main()
    elif len(args) == 1:
        arg = args[0]
        if arg.lower().endswith('.csv'):
            main(input_file=arg)
        else:
            run_single_issue(arg)
    elif len(args) == 2:
        main(input_file=args[0], output_file=args[1])
    else:
        print("Usage:")
        print("  Single issue: python code/main.py \"issue description\"")
        print("  CSV processing: python code/main.py input.csv output.csv")
        print("  Default: python code/main.py")
