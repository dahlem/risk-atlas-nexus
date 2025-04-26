# Standard Library
import re
import os
import csv
import datetime
import json
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict

# Third Party
import pandas as pd
import numpy as np
from tqdm import tqdm
from sssom_schema import Mapping, EntityReference, MappingSet
from transformers import Pipeline
from sentence_transformers import SentenceTransformer, util

# Local
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import *
from risk_atlas_nexus.metadata_base import MappingMethod
from risk_atlas_nexus.blocks.risk_mapping.risk_mapper import RiskMapper 
from risk_atlas_nexus.blocks.inference import create_inference_engine
from risk_atlas_nexus.toolkit.logging import configure_logger

logger = configure_logger(__name__)

MAP_DIR = "src/risk_atlas_nexus/data/mappings/"
DATA_DIR = "src/risk_atlas_nexus/data/knowledge_graph/"
RESOURCES_DIR = "resources/airbench"

# Ensure directories exist
os.makedirs(MAP_DIR, exist_ok=True)
os.makedirs(RESOURCES_DIR, exist_ok=True)

# Define taxonomies to map with AIR-Bench
TAXONOMIES = {
    "ibm-ai-risk-atlas": {
        "file": "risk_atlas_data.yaml",
        "prefix": "ibmairisk",
        "name": "IBM AI Risk Atlas",
        "url": "https://www.ibm.com/risk-atlas"
    },
    "owasp-llm-2.0": {
        "file": "owasp_llm_2.0_data.yaml",
        "prefix": "owaspai",
        "name": "OWASP GenAI Top 10 2023",
        "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/"
    },
    "nist-ai-rmf": {
        "file": "nist_ai_rmf_data.yaml",
        "prefix": "nistai",
        "name": "NIST AI Risk Management Framework",
        "url": "https://www.nist.gov/itl/ai-risk-management-framework"
    },
    "mit-ai-risk-repository": {
        "file": "mit_ai_risk_repository_data.yaml",
        "prefix": "mitairisk",
        "name": "MIT AI Risk Repository",
        "url": "https://www.nist.gov/itl/ai-risk-management-framework"
    }
}

def load_risks_from_yaml(file_path: str, taxonomy_id: str) -> List[Risk]:
    """
    Load risks from a YAML file
    
    Args:
        file_path: Path to the YAML file
        taxonomy_id: ID of the taxonomy
        
    Returns:
        List[Risk]: List of Risk objects
    """
    import yaml
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        risks = []
        for risk_data in data.get("risks", []):
            risk = Risk(
                id=risk_data.get("id"),
                name=risk_data.get("name"),
                description=risk_data.get("description", ""),
                isDefinedByTaxonomy=taxonomy_id
            )
            risks.append(risk)
        
        return risks
    except Exception as e:
        logger.error(f"Error loading risks from {file_path}: {str(e)}")
        return []

def load_airbench_risks() -> List[Risk]:
    """
    Load the AIR-Bench risks from the knowledge graph
    
    Returns:
        List[Risk]: List of AIR-Bench risks
    """
    file_path = os.path.join(DATA_DIR, "airbench_data.yaml")
    return load_risks_from_yaml(file_path, "airbench")

def load_taxonomy_risks(taxonomy_id: str) -> List[Risk]:
    """
    Load risks from a specific taxonomy
    
    Args:
        taxonomy_id: ID of the taxonomy
        
    Returns:
        List[Risk]: List of risks from the taxonomy
    """
    if taxonomy_id not in TAXONOMIES:
        logger.error(f"Unknown taxonomy: {taxonomy_id}")
        return []
    
    file_path = os.path.join(DATA_DIR, TAXONOMIES[taxonomy_id]["file"])
    return load_risks_from_yaml(file_path, taxonomy_id)

def generate_mappings(taxonomy_id: str, method: MappingMethod) -> MappingSet:
    """
    Generate mappings between AIR-Bench and another taxonomy using the specified method
    
    Args:
        taxonomy_id: ID of the taxonomy to map with AIR-Bench
        method: Method to use for generating mappings
        
    Returns:
        MappingSet: Set of mappings
    """
    # Load risks
    airbench_risks = load_airbench_risks()
    target_risks = load_taxonomy_risks(taxonomy_id)
    
    if not airbench_risks or not target_risks:
        logger.error("Failed to load risks")
        return MappingSet(mappings=[], mapping_set_id="empty")
    
    logger.info(f"Loaded {len(airbench_risks)} AIR-Bench risks and {len(target_risks)} {taxonomy_id} risks")
    
    # Create inference engine for LLM-based mapping if needed
    inference_engine = None
    if method == MappingMethod.INFERENCE:
        try:
            inference_engine = create_inference_engine()
        except Exception as e:
            logger.error(f"Error creating inference engine: {str(e)}")
            logger.info("Falling back to semantic mapping")
            method = MappingMethod.SEMANTIC
    
    # Create risk mapper
    risk_mapper = RiskMapper()
    
    # Generate mappings
    logger.info(f"Generating mappings using {method.name} method")
    mappings = risk_mapper.generate(
        new_risks=airbench_risks,
        existing_risks=target_risks,
        inference_engine=inference_engine,
        new_prefix="airbench",
        mapping_method=method
    )
    
    logger.info(f"Generated {len(mappings)} mappings")
    
    # Create mapping set with metadata
    mapping_set = MappingSet(
        mappings=mappings,
        mapping_set_id=f"airbench-{taxonomy_id}-{method.name.lower()}",
        license="CC-BY-4.0",
        mapping_date=datetime.date.today(),
        mapping_provider="Risk Atlas Nexus",
        curie_map={
            "airbench": "https://arxiv.org/abs/2407.17436",
            TAXONOMIES[taxonomy_id]["prefix"]: TAXONOMIES[taxonomy_id]["url"]
        }
    )
    
    return mapping_set

def save_mappings_to_tsv(mapping_set: MappingSet):
    """
    Save mappings to TSV file in SSSOM format
    
    Args:
        mapping_set: Set of mappings to save
    """
    output_file = os.path.join(MAP_DIR, f"{mapping_set.mapping_set_id}.sssom.tsv")
    
    # Extract header fields from the first mapping
    header = []
    if mapping_set.mappings:
        header = list(mapping_set.mappings[0].keys())
    
    # Write mappings to TSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for mapping in mapping_set.mappings:
            writer.writerow(mapping)
    
    logger.info(f"Saved {len(mapping_set.mappings)} mappings to {output_file}")

def generate_mappings_for_tier_categories() -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Generate mappings between AIR-Bench tier categories and other taxonomies
    
    Returns:
        Dict[str, List[Tuple[str, str, str]]]: Dictionary mapping taxonomy IDs to lists of mappings
    """
    import yaml
    
    # Load AIR-Bench data
    airbench_file = os.path.join(DATA_DIR, "airbench_data.yaml")
    with open(airbench_file, "r", encoding="utf-8") as f:
        airbench_data = yaml.safe_load(f)
    
    # Extract tier groups
    tier_groups = {}
    for group in airbench_data.get("riskgroups", []):
        tier_groups[group.get("id")] = {
            "name": group.get("name"),
            "description": group.get("description", ""),
            "taxonomy": group.get("isDefinedByTaxonomy")
        }
    
    # Define an embedding model for semantic similarity
    try:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    except Exception as e:
        logger.error(f"Error loading sentence transformer: {str(e)}")
        return {}
    
    # Generate embeddings for AIR-Bench tier groups
    airbench_embeddings = {}
    for group_id, group_info in tier_groups.items():
        text = f"{group_info['name']} - {group_info['description']}"
        embedding = model.encode(text, convert_to_tensor=True)
        airbench_embeddings[group_id] = {
            "embedding": embedding,
            "info": group_info
        }
    
    # Store mappings for each taxonomy
    all_mappings = {}
    
    # For each taxonomy, find the most similar groups
    for taxonomy_id, taxonomy_info in TAXONOMIES.items():
        logger.info(f"Generating tier category mappings for {taxonomy_id}")
        
        # Load taxonomy data
        taxonomy_file = os.path.join(DATA_DIR, taxonomy_info["file"])
        try:
            with open(taxonomy_file, "r", encoding="utf-8") as f:
                taxonomy_data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading {taxonomy_id} data: {str(e)}")
            continue
        
        # Extract groups
        taxonomy_groups = {}
        for group in taxonomy_data.get("riskgroups", []):
            group_id = group.get("id")
            group_name = group.get("name", "")
            group_desc = group.get("description", "")
            
            # Skip if no name or not from this taxonomy
            if not group_name or group.get("isDefinedByTaxonomy") != taxonomy_id:
                continue
                
            taxonomy_groups[group_id] = {
                "name": group_name,
                "description": group_desc,
                "embedding": model.encode(f"{group_name} - {group_desc}", convert_to_tensor=True)
            }
        
        # Find mappings for AIR-Bench tier groups
        taxonomy_mappings = []
        
        for airbench_id, airbench_data in airbench_embeddings.items():
            airbench_name = airbench_data["info"]["name"]
            airbench_embedding = airbench_data["embedding"]
            
            best_match = None
            best_score = -1
            best_match_name = ""
            
            # Find the best match in this taxonomy
            for taxonomy_group_id, taxonomy_group in taxonomy_groups.items():
                taxonomy_embedding = taxonomy_group["embedding"]
                similarity = util.pytorch_cos_sim(airbench_embedding, taxonomy_embedding).item()
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = taxonomy_group_id
                    best_match_name = taxonomy_group["name"]
            
            # Determine the relationship type based on similarity score
            relationship = "skos:relatedMatch"
            if best_score > 0.9:
                relationship = "skos:exactMatch"
            elif best_score > 0.8:
                relationship = "skos:closeMatch"
            
            # Add the mapping if score is high enough
            if best_score > 0.6 and best_match is not None:
                taxonomy_mappings.append((airbench_id, relationship, best_match, best_score))
                logger.info(f"  Mapped '{airbench_name}' to '{best_match_name}' with score {best_score:.2f}")
        
        all_mappings[taxonomy_id] = taxonomy_mappings
    
    # Save the tier category mappings
    save_tier_category_mappings(all_mappings)
    
    return all_mappings

def save_tier_category_mappings(mappings: Dict[str, List[Tuple[str, str, str, float]]]):
    """
    Save tier category mappings to a JSON file
    
    Args:
        mappings: Dictionary mapping taxonomy IDs to lists of mappings
    """
    output_file = os.path.join(RESOURCES_DIR, "airbench_tier_category_mappings.json")
    
    formatted_mappings = {}
    for taxonomy_id, taxonomy_mappings in mappings.items():
        formatted_mappings[taxonomy_id] = [
            {
                "airbench_id": mapping[0],
                "relationship": mapping[1],
                "target_id": mapping[2],
                "similarity_score": mapping[3]
            }
            for mapping in taxonomy_mappings
        ]
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_mappings, f, indent=2)
    
    logger.info(f"Saved tier category mappings to {output_file}")

def generate_mapping_statistics(taxonomy_id: str, mapping_set: MappingSet):
    """
    Generate statistics about the mappings
    
    Args:
        taxonomy_id: ID of the taxonomy
        mapping_set: Set of mappings
    """
    # Count mappings by relationship type
    relationship_counts = defaultdict(int)
    for mapping in mapping_set.mappings:
        relationship_counts[mapping.predicate_id] += 1
    
    # Count AIR-Bench risks with mappings
    airbench_ids = set()
    for mapping in mapping_set.mappings:
        if str(mapping.subject_id).startswith("airbench:"):
            airbench_ids.add(str(mapping.subject_id).split(":", 1)[1])
    
    # Count target risks with mappings
    target_ids = set()
    for mapping in mapping_set.mappings:
        if str(mapping.object_id).startswith(f"{TAXONOMIES[taxonomy_id]['prefix']}:"):
            target_ids.add(str(mapping.object_id).split(":", 1)[1])
    
    # Calculate average similarity score
    similarity_scores = [float(m.similarity_score) for m in mapping_set.mappings if hasattr(m, "similarity_score")]
    avg_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    
    # Create statistics dictionary
    stats = {
        "taxonomy_id": taxonomy_id,
        "taxonomy_name": TAXONOMIES[taxonomy_id]["name"],
        "mapping_set_id": mapping_set.mapping_set_id,
        "total_mappings": len(mapping_set.mappings),
        "relationships": {k: v for k, v in relationship_counts.items()},
        "airbench_risks_mapped": len(airbench_ids),
        "target_risks_mapped": len(target_ids),
        "average_similarity_score": avg_score,
        "date": str(datetime.date.today())
    }
    
    # Save statistics to file
    output_file = os.path.join(RESOURCES_DIR, f"{mapping_set.mapping_set_id}_stats.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved mapping statistics to {output_file}")
    
    return stats

def create_mapping_report(all_stats: List[Dict[str, Any]]):
    """
    Create a comprehensive mapping report in Markdown format
    
    Args:
        all_stats: List of statistics dictionaries for all mapping sets
    """
    if not all_stats:
        logger.error("No statistics available for creating report")
        return
    
    report_file = os.path.join(RESOURCES_DIR, "airbench_mapping_report.md")
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# AIR-Bench Taxonomy Mapping Report\n\n")
        f.write(f"Generated on: {datetime.date.today()}\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report summarizes the mappings between the AIR-Bench taxonomy and other AI risk taxonomies.\n\n")
        
        f.write("| Taxonomy | Total Mappings | AIR-Bench Risks Mapped | Target Risks Mapped | Avg. Similarity Score |\n")
        f.write("|----------|---------------|----------------------|-------------------|----------------------|\n")
        
        for stats in all_stats:
            f.write(f"| {stats['taxonomy_name']} | {stats['total_mappings']} | {stats['airbench_risks_mapped']} | {stats['target_risks_mapped']} | {stats['average_similarity_score']:.2f} |\n")
        
        f.write("\n## Detailed Statistics\n\n")
        
        for stats in all_stats:
            f.write(f"### {stats['taxonomy_name']} ({stats['taxonomy_id']})\n\n")
            f.write(f"- **Total mappings**: {stats['total_mappings']}\n")
            f.write(f"- **AIR-Bench risks mapped**: {stats['airbench_risks_mapped']}\n")
            f.write(f"- **{stats['taxonomy_name']} risks mapped**: {stats['target_risks_mapped']}\n")
            f.write(f"- **Average similarity score**: {stats['average_similarity_score']:.2f}\n\n")
            
            f.write("#### Relationship Types\n\n")
            for rel, count in stats['relationships'].items():
                rel_name = rel.replace("skos:", "").replace("Match", " Match")
                f.write(f"- **{rel_name}**: {count}\n")
            
            f.write("\n")
    
    logger.info(f"Created mapping report at {report_file}")

def run_all_mappings():
    """Run all mapping operations for AIR-Bench with all supported taxonomies"""
    # Generate tier category mappings
    logger.info("Generating tier category mappings")
    generate_mappings_for_tier_categories()
    
    # Store all statistics for the report
    all_stats = []
    
    # For each taxonomy, generate and save mappings
    for taxonomy_id in TAXONOMIES.keys():
        try:
            logger.info(f"Generating mappings for {taxonomy_id}")
            
            # Generate semantic mappings
            semantic_mapping_set = generate_mappings(taxonomy_id, MappingMethod.SEMANTIC)
            save_mappings_to_tsv(semantic_mapping_set)
            
            # Generate statistics
            stats = generate_mapping_statistics(taxonomy_id, semantic_mapping_set)
            all_stats.append(stats)
            
        except Exception as e:
            logger.error(f"Error processing {taxonomy_id}: {str(e)}")
    
    # Create the mapping report
    create_mapping_report(all_stats)
    
    logger.info("All mapping operations completed successfully")

if __name__ == "__main__":
    try:
        run_all_mappings()
    except Exception as e:
        logger.error(f"Error running AIR-Bench mappings: {str(e)}")