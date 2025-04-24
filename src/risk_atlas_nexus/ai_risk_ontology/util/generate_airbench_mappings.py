# Standard Library
import re
import os
import csv
import datetime
from typing import List, Dict, Any

# Third Party
import pandas as pd
from sssom_schema import Mapping, EntityReference, MappingSet

# Local
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import *
from risk_atlas_nexus.metadata_base import MappingMethod
from risk_atlas_nexus.blocks.risk_mapping.risk_mapper import RiskMapper 
from risk_atlas_nexus.blocks.inference import create_inference_engine
from risk_atlas_nexus.toolkit.logging import configure_logger

logger = configure_logger(__name__)

MAP_DIR = "src/risk_atlas_nexus/data/mappings/"
DATA_DIR = "src/risk_atlas_nexus/data/knowledge_graph/"

def load_airbench_risks() -> List[Risk]:
    """
    Load the AIR-Bench risks from the knowledge graph
    """
    import yaml
    from linkml_runtime.loaders import YAMLLoader
    
    file_path = os.path.join(DATA_DIR, "airbench_data.yaml")
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    risks = []
    for risk_data in data.get("risks", []):
        risk = Risk(
            id=risk_data.get("id"),
            name=risk_data.get("name"),
            description=risk_data.get("description", ""),
            isDefinedByTaxonomy="airbench"
        )
        risks.append(risk)
    
    return risks

def load_ibm_risks() -> List[Risk]:
    """
    Load the IBM Risk Atlas risks from the knowledge graph
    """
    import yaml
    from linkml_runtime.loaders import YAMLLoader
    
    file_path = os.path.join(DATA_DIR, "risk_atlas_data.yaml")
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    risks = []
    for risk_data in data.get("risks", []):
        risk = Risk(
            id=risk_data.get("id"),
            name=risk_data.get("name"),
            description=risk_data.get("description", ""),
            isDefinedByTaxonomy="ibm-ai-risk-atlas"
        )
        risks.append(risk)
    
    return risks

def generate_mappings(method: MappingMethod) -> MappingSet:
    """
    Generate mappings between AIR-Bench and IBM Risk Atlas using the specified method
    """
    # Load risks
    airbench_risks = load_airbench_risks()
    ibm_risks = load_ibm_risks()
    
    logger.info(f"Loaded {len(airbench_risks)} AIR-Bench risks and {len(ibm_risks)} IBM risks")
    
    # Create inference engine for LLM-based mapping if needed
    inference_engine = None
    if method == MappingMethod.INFERENCE:
        inference_engine = create_inference_engine()
    
    # Create risk mapper
    risk_mapper = RiskMapper()
    
    # Generate mappings
    logger.info(f"Generating mappings using {method.name} method")
    mappings = risk_mapper.generate(
        new_risks=airbench_risks,
        existing_risks=ibm_risks,
        inference_engine=inference_engine,
        new_prefix="airbench",
        mapping_method=method
    )
    
    logger.info(f"Generated {len(mappings)} mappings")
    
    # Create mapping set with metadata
    mapping_set = MappingSet(
        mappings=mappings,
        mapping_set_id=f"airbench-ibm-{method.name.lower()}",
        license="CC-BY-4.0",
        mapping_date=datetime.date.today(),
        mapping_provider="Risk Atlas Nexus",
        curie_map={
            "airbench": "https://arxiv.org/abs/2407.17436",
            "ibmairisk": "https://www.ibm.com/risk-atlas"
        }
    )
    
    return mapping_set

def save_mappings_to_tsv(mapping_set: MappingSet, method: MappingMethod):
    """
    Save mappings to TSV file in SSSOM format
    """
    output_file = os.path.join(MAP_DIR, f"airbench2ibm-{method.name.lower()}.sssom.tsv")
    
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

if __name__ == "__main__":
    try:
        # Generate semantic mappings
        semantic_mappings = generate_mappings(MappingMethod.SEMANTIC)
        save_mappings_to_tsv(semantic_mappings, MappingMethod.SEMANTIC)
        
        # Generate inference-based mappings if needed
        # This part can be commented out if not using LLM-based mapping
        # inference_mappings = generate_mappings(MappingMethod.INFERENCE)
        # save_mappings_to_tsv(inference_mappings, MappingMethod.INFERENCE)
    except Exception as e:
        logger.error(f"Error generating AIR-Bench mappings: {str(e)}")