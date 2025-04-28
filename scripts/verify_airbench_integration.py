#!/usr/bin/env python3
"""
Script to verify the integration of the full AIRBench taxonomy into Risk Atlas Nexus.
This script checks that all 314 AIRBench risks are properly loaded.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.risk_atlas_nexus.library import RiskAtlasNexus

def main():
    # Initialize Risk Atlas Nexus
    nexus = RiskAtlasNexus()
    
    # Get all taxonomies
    taxonomies = nexus.get_all_taxonomies()
    print(f"Found {len(taxonomies)} taxonomies:")
    for taxonomy in taxonomies:
        print(f"  - {taxonomy.name} ({taxonomy.id})")
    
    # Find AIRBench HuggingFace taxonomy
    airbench_taxonomy = None
    for taxonomy in taxonomies:
        if taxonomy.id == "airbench-hf":
            airbench_taxonomy = taxonomy
            break
    
    if not airbench_taxonomy:
        print("AIRBench HuggingFace taxonomy not found!")
        # Let's try with just "AIRBench" in the name as fallback
        for taxonomy in taxonomies:
            if taxonomy.name and "AIRBench" in taxonomy.name:
                airbench_taxonomy = taxonomy
                print(f"Using fallback taxonomy: {taxonomy.name} ({taxonomy.id})")
                break
        
        if not airbench_taxonomy:
            print("AIRBench taxonomy not found!")
            return
    
    print(f"\nFound AIRBench taxonomy: {airbench_taxonomy.name} ({airbench_taxonomy.id})")
    
    # Get all risks from AIRBench taxonomy
    airbench_risks = nexus.get_all_risks(taxonomy=airbench_taxonomy.id)
    print(f"\nFound {len(airbench_risks)} risks in AIRBench taxonomy")
    
    # Display first few risks as example
    print("\nExample risks from AIRBench:")
    for risk in airbench_risks[:5]:
        print(f"  - {risk.name} ({risk.id})")
    
    # Check if we have all 314 risks
    if len(airbench_risks) == 314:
        print("\n✅ Successfully integrated all 314 AIRBench risks!")
    else:
        print(f"\n❌ Expected 314 risks, but found {len(airbench_risks)} risks.")
        
    # Check hierarchical structure by counting unique prefixes in risk IDs
    print("\nAnalyzing hierarchical structure based on risk IDs:")
    
    # Get unique prefixes from risk IDs (which should follow our hierarchy)
    tier1_prefixes = set()
    tier2_prefixes = set() 
    tier3_prefixes = set()
    
    for risk in airbench_risks:
        if risk.id and risk.id.startswith("airbench-hf-"):
            parts = risk.id.split("-")
            if len(parts) >= 3:
                tier1_prefixes.add(parts[2])  # e.g., "system"
            if len(parts) >= 4:
                tier2_prefixes.add("-".join(parts[2:4]))  # e.g., "system-operational" 
            if len(parts) >= 5:
                tier3_prefixes.add("-".join(parts[2:5]))  # e.g., "system-operational-reliability"
    
    print(f"  Estimated Tier 1 categories: {len(tier1_prefixes)}")
    print(f"  Estimated Tier 2 categories: {len(tier2_prefixes)}")
    print(f"  Estimated Tier 3 categories: {len(tier3_prefixes)}")

if __name__ == "__main__":
    main()