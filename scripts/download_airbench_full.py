#!/usr/bin/env python3
"""
Script to download the complete AIRBench taxonomy from Hugging Face
and create a standalone YAML file that doesn't depend on the existing code.
"""

import os
import yaml
import json
import sys
from datasets import load_dataset

# Load the dataset from Hugging Face (requires HF_TOKEN in environment)
print("Loading AIRBench dataset from Hugging Face...")
try:
    dataset = load_dataset("stanford-crfm/air-bench-2024", trust_remote_code=True)
    print(f"Successfully loaded AIRBench dataset!")
    print(f"Dataset info: {dataset}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# New file path
output_path = "/Users/ddahlem/Documents/repos/risk-atlas-nexus/src/risk_atlas_nexus/data/knowledge_graph/airbench_hf_data.yaml"

# Extract taxonomy structure
print("Extracting AIRBench taxonomy...")
try:
    # Get the default split (or any available split)
    split_name = next(iter(dataset.keys()))
    data = dataset[split_name]
    
    # Mapping for level 1 categories based on first digit in cate-idx
    level1_map = {
        '1': 'System & Operational Risks',
        '2': 'Content Safety Risks',
        '3': 'Societal Risks',
        '4': 'Legal & Rights Risks'
    }
    
    # Create taxonomy structure
    taxonomy_data = {
        "documents": [
            {
                "id": "arxiv.org/abs/2407.17436",
                "name": "AIR-Bench: A Regulation-Aligned Safety Benchmark for Responsible AI Development",
                "description": "AIR-Bench is the first AI safety benchmark aligned with emerging government regulations and company policies. It features a four-tiered safety taxonomy with 314 granular risk categories derived from analyzing 8 government regulations and 16 company policies.",
                "url": "https://arxiv.org/abs/2407.17436",
                "dateCreated": "2024-07-29",
                "dateModified": "2024-07-29"
            }
        ],
        "taxonomies": [
            {
                "id": "airbench-hf",
                "name": "AIR-Bench Taxonomy (HuggingFace)",
                "description": "A four-tiered safety taxonomy with 314 granular risk categories derived from analyzing 8 government regulations and 16 company policies.",
                "url": "https://arxiv.org/abs/2407.17436",
                "dateCreated": "2024-07-29",
                "version": "1.0",
                "hasDocumentation": ["arxiv.org/abs/2407.17436"]
            }
        ],
        "riskgroups": [],
        "risks": []
    }
    
    # Process unique category entries
    unique_categories = set()
    for item in data:
        if 'cate-idx' in item and 'l2-name' in item and 'l3-name' in item and 'l4-name' in item:
            category_tuple = (item['cate-idx'], item['l2-name'], item['l3-name'], item['l4-name'])
            unique_categories.add(category_tuple)
    
    print(f"Found {len(unique_categories)} unique categories")
    
    # Create tier 1 risk groups (based on first digit of cate-idx)
    tier1_groups = {}
    for level1_id, level1_name in level1_map.items():
        tier1_id = f"airbench-hf-{level1_id}"
        tier1_groups[level1_id] = tier1_id
        taxonomy_data["riskgroups"].append({
            "id": tier1_id,
            "name": level1_name,
            "description": f"{level1_name} - A top level category in the AIR-Bench taxonomy.",
            "isDefinedByTaxonomy": "airbench-hf"
        })
    
    # Process tier 2, 3, and 4 categories
    tier2_groups = {}  # map from l2_name to group id
    tier3_groups = {}  # map from l3_name to group id
    
    for cate_idx, l2_name, l3_name, l4_name in sorted(unique_categories):
        # Parse tier 1 from cate_idx
        l1_idx = cate_idx.split('.')[0]
        tier1_id = tier1_groups.get(l1_idx)
        
        # Create tier 2 group if not exists
        if l2_name not in tier2_groups:
            tier2_id = f"airbench-hf-{l2_name.lower().replace(' & ', '-').replace(' ', '-')}"
            tier2_groups[l2_name] = tier2_id
            taxonomy_data["riskgroups"].append({
                "id": tier2_id,
                "name": l2_name,
                "description": f"{l2_name} - A subcategory of {level1_map.get(l1_idx)} in the AIR-Bench taxonomy.",
                "isDefinedByTaxonomy": "airbench-hf",
                "broadMatch": [tier1_id] if tier1_id else []
            })
        
        # Create tier 3 group if not exists
        tier2_id = tier2_groups[l2_name]
        if l3_name not in tier3_groups:
            tier3_id = f"airbench-hf-{l3_name.lower().replace(' & ', '-').replace(' ', '-')}"
            tier3_groups[l3_name] = tier3_id
            taxonomy_data["riskgroups"].append({
                "id": tier3_id,
                "name": l3_name,
                "description": f"{l3_name} - A subcategory of {l2_name} in the AIR-Bench taxonomy.",
                "isDefinedByTaxonomy": "airbench-hf",
                "broadMatch": [tier2_id] if tier2_id else []
            })
        
        # Create tier 4 risk
        tier3_id = tier3_groups[l3_name]
        tier4_id = f"airbench-hf-{l4_name.lower().replace(' & ', '-').replace(' ', '-')}"
        
        # Check if we already added this risk
        if not any(r['id'] == tier4_id for r in taxonomy_data["risks"]):
            taxonomy_data["risks"].append({
                "id": tier4_id,
                "name": l4_name,
                "description": f"Risk related to {l4_name} within the {l3_name} category.",
                "isDefinedByTaxonomy": "airbench-hf",
                "isPartOf": tier3_id
            })
    
    # Update narrowMatch fields for tier 1 groups
    for tier1_idx, tier1_id in tier1_groups.items():
        narrowMatch = []
        for l2_name, tier2_id in tier2_groups.items():
            # Check if this tier2 belongs to this tier1
            for group in taxonomy_data["riskgroups"]:
                if group["id"] == tier2_id and "broadMatch" in group and tier1_id in group["broadMatch"]:
                    narrowMatch.append(tier2_id)
        
        # Add narrowMatch to tier1 group
        for group in taxonomy_data["riskgroups"]:
            if group["id"] == tier1_id:
                group["narrowMatch"] = narrowMatch if narrowMatch else []
    
    # Update narrowMatch fields for tier 2 groups
    for l2_name, tier2_id in tier2_groups.items():
        narrowMatch = []
        for l3_name, tier3_id in tier3_groups.items():
            # Check if this tier3 belongs to this tier2
            for group in taxonomy_data["riskgroups"]:
                if group["id"] == tier3_id and "broadMatch" in group and tier2_id in group["broadMatch"]:
                    narrowMatch.append(tier3_id)
        
        # Add narrowMatch to tier2 group
        for group in taxonomy_data["riskgroups"]:
            if group["id"] == tier2_id:
                group["narrowMatch"] = narrowMatch if narrowMatch else []
    
    # Update narrowMatch fields for tier 3 groups
    for l3_name, tier3_id in tier3_groups.items():
        narrowMatch = []
        for risk in taxonomy_data["risks"]:
            if risk["isPartOf"] == tier3_id:
                narrowMatch.append(risk["id"])
        
        # Add narrowMatch to tier3 group
        for group in taxonomy_data["riskgroups"]:
            if group["id"] == tier3_id:
                group["narrowMatch"] = narrowMatch if narrowMatch else []
    
    # Helper function to fix string duplication issues and ensure match fields are lists
    def fix_data_for_yaml(data):
        if isinstance(data, list):
            return [fix_data_for_yaml(item) for item in data]
        elif isinstance(data, dict):
            result = {}
            for k, v in data.items():
                # Ensure match fields are always lists
                if k in ["broadMatch", "narrowMatch", "closeMatch", "exactMatch", "relatedMatch"]:
                    if v is None:
                        result[k] = None
                    elif isinstance(v, list):
                        result[k] = [fix_data_for_yaml(item) for item in v]
                    else:
                        result[k] = [fix_data_for_yaml(v)]
                else:
                    result[k] = fix_data_for_yaml(v)
            return result
        elif isinstance(data, str):
            # Ensure the string is not duplicated when saved to YAML
            return data
        else:
            return data
    
    # Fix any data format issues
    taxonomy_data = fix_data_for_yaml(taxonomy_data)
    
    # Write the output YAML
    with open(output_path, "w") as f:
        yaml.dump(taxonomy_data, f, sort_keys=False, default_flow_style=False)
    
    print(f"Successfully wrote AIRBench taxonomy to {output_path}")
    print(f"Total risks: {len(taxonomy_data['risks'])}")
    print(f"Total risk groups: {len(taxonomy_data['riskgroups'])}")
    print(f"Tier 1 groups: {len(tier1_groups)}")
    print(f"Tier 2 groups: {len(tier2_groups)}")
    print(f"Tier 3 groups: {len(tier3_groups)}")
    
except Exception as e:
    print(f"Error processing taxonomy data: {e}")
    import traceback
    traceback.print_exc()
    exit(1)