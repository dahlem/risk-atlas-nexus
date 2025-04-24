# Standard Library
import re
import os
from typing import List, Dict, Any

# Third Party
import pandas as pd
from linkml_runtime.dumpers import YAMLDumper
import requests
import json

# Local
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import *
from risk_atlas_nexus.toolkit.logging import configure_logger

logger = configure_logger(__name__)

SCHEMA_DIR = "src/risk_atlas_nexus/ai_risk_ontology/schema/"
DATA_DIR = "src/risk_atlas_nexus/data/knowledge_graph/"
OUTPUT_FILE = os.path.join(DATA_DIR, "airbench_data.yaml")

def create_container_object() -> Container:
    # Create risk taxonomy paper documentation
    documents = [
        Documentation(
            **{
                "id": "arxiv.org/abs/2407.17436",
                "name": "AIR-Bench: A Regulation-Aligned Safety Benchmark for Responsible AI Development",
                "description": "AIR-Bench is the first AI safety benchmark aligned with emerging government regulations and company policies. It features a four-tiered safety taxonomy with 314 granular risk categories derived from analyzing 8 government regulations and 16 company policies. The benchmark contains 5,694 diverse, manually curated, and human-audited prompts designed to evaluate AI model safety compliance.",
                "url": "https://arxiv.org/abs/2407.17436",
                "dateCreated": "2024-07-29",
                "dateModified": "2024-07-29",
            })
    ]
    
    # Create risk taxonomy
    taxonomies = [
        RiskTaxonomy(
            **{
                "id": "airbench",
                "name": "AIR-Bench Taxonomy",
                "description": "A four-tiered safety taxonomy with 314 granular risk categories derived from analyzing 8 government regulations and 16 company policies. At the highest level, this taxonomy encompasses System & Operational Risks, Content Safety Risks, Societal Risks, and Legal & Rights Risks.",
                "url": "https://arxiv.org/abs/2407.17436",
                "version": "1.0",
                "dateCreated": "2024-07-29",
                "hasDocumentation": ["arxiv.org/abs/2407.17436"],
            })
    ]

    # Create risk groups for tier 1 (top level)
    tier1_groups = [
        {"id": "system-operational-risks", "name": "System & Operational Risks"},
        {"id": "content-safety-risks", "name": "Content Safety Risks"},
        {"id": "societal-risks", "name": "Societal Risks"},
        {"id": "legal-rights-risks", "name": "Legal & Rights Risks"}
    ]
    
    risk_groups = []
    
    # Create tier 1 risk groups
    for group in tier1_groups:
        risk_groups.append(RiskGroup(
            **{
                "id": f"airbench-{group['id']}",
                "name": group["name"],
                "isDefinedByTaxonomy": "airbench",
            }
        ))
    
    # Fetch AIR-Bench data
    risks_data = get_airbench_risks()
    
    # Create tier 2 risk groups and add relationships to tier 1
    tier2_groups = {}
    for risk in risks_data:
        tier1_id = f"airbench-{convert_to_id(risk['tier1'])}"
        tier2_id = f"airbench-{convert_to_id(risk['tier2'])}"
        
        if tier2_id not in tier2_groups:
            tier2_groups[tier2_id] = {
                "id": tier2_id,
                "name": risk['tier2'],
                "isDefinedByTaxonomy": "airbench",
                "broadMatch": [tier1_id],
                "tier1": tier1_id
            }
    
    # Create tier 2 risk groups
    for group_id, group_data in tier2_groups.items():
        risk_group = RiskGroup(
            **{
                "id": group_data["id"],
                "name": group_data["name"],
                "isDefinedByTaxonomy": "airbench",
                "broadMatch": group_data["broadMatch"]
            }
        )
        risk_groups.append(risk_group)
        
        # Add narrowMatch to the corresponding tier 1 group
        for tier1_group in risk_groups:
            if tier1_group.id == group_data["tier1"]:
                if not hasattr(tier1_group, "narrowMatch") or tier1_group.narrowMatch is None:
                    tier1_group.narrowMatch = []
                tier1_group.narrowMatch.append(group_data["id"])
    
    # Create tier 3 risk groups and add relationships to tier 2
    tier3_groups = {}
    for risk in risks_data:
        tier2_id = f"airbench-{convert_to_id(risk['tier2'])}"
        tier3_id = f"airbench-{convert_to_id(risk['tier3'])}"
        
        if tier3_id not in tier3_groups:
            tier3_groups[tier3_id] = {
                "id": tier3_id,
                "name": risk['tier3'],
                "isDefinedByTaxonomy": "airbench",
                "broadMatch": [tier2_id],
                "tier2": tier2_id
            }
    
    # Create tier 3 risk groups
    for group_id, group_data in tier3_groups.items():
        risk_group = RiskGroup(
            **{
                "id": group_data["id"],
                "name": group_data["name"],
                "isDefinedByTaxonomy": "airbench",
                "broadMatch": group_data["broadMatch"]
            }
        )
        risk_groups.append(risk_group)
        
        # Add narrowMatch to the corresponding tier 2 group
        for tier2_group in risk_groups:
            if tier2_group.id == group_data["tier2"]:
                if not hasattr(tier2_group, "narrowMatch") or tier2_group.narrowMatch is None:
                    tier2_group.narrowMatch = []
                tier2_group.narrowMatch.append(group_data["id"])
    
    # Create individual risks (tier 4)
    risk_objects = []
    for risk in risks_data:
        tier3_id = f"airbench-{convert_to_id(risk['tier3'])}"
        risk_id = f"airbench-{convert_to_id(risk['tier4'])}"
        
        risk_obj = Risk(
            **{
                "id": risk_id,
                "name": risk['tier4'],
                "description": risk['definition'] if 'definition' in risk else f"{risk['tier4']} - A risk category from the AIR-Bench taxonomy.",
                "isPartOf": tier3_id,
                "isDefinedByTaxonomy": "airbench",
            }
        )
        risk_objects.append(risk_obj)
        
        # Add narrowMatch to the corresponding tier 3 group
        for tier3_group in risk_groups:
            if tier3_group.id == tier3_id:
                if not hasattr(tier3_group, "narrowMatch") or tier3_group.narrowMatch is None:
                    tier3_group.narrowMatch = []
                tier3_group.narrowMatch.append(risk_id)
    
    # Create container
    container = Container(documents=documents, taxonomies=taxonomies, riskgroups=risk_groups, risks=risk_objects)
    return container

def convert_to_id(name: str) -> str:
    """Convert a name to an ID by lowercasing and replacing spaces with hyphens"""
    return name.lower().replace(" ", "-").replace("&", "and").replace(",", "").replace("/", "-")

def get_airbench_risks() -> List[Dict[str, Any]]:
    """
    Get AIR-Bench risks data - this is a sample of the structure.
    In a real implementation, this would fetch from a full AIR-Bench dataset.
    """
    risks = [
        # System & Operational Risks examples
        {"tier1": "System & Operational Risks", "tier2": "Model Performance", "tier3": "Hallucination", "tier4": "Factual Error", 
         "definition": "When the model produces false or incorrect information that contradicts established facts or knowledge."},
        {"tier1": "System & Operational Risks", "tier2": "Model Performance", "tier3": "Hallucination", "tier4": "Made-up Sources", 
         "definition": "When the model fabricates non-existent references, citations, or sources to support its statements."},
        {"tier1": "System & Operational Risks", "tier2": "Model Performance", "tier3": "Instruction Following", "tier4": "Refusal of Appropriate Requests", 
         "definition": "When the model incorrectly refuses to respond to legitimate, safe requests due to misidentifying them as harmful."},

        # Content Safety Risks examples
        {"tier1": "Content Safety Risks", "tier2": "Adult Content", "tier3": "Sexual Content", "tier4": "Explicit Sexual Instructions",
         "definition": "When the model provides explicit details or instructions about sexual activities, behaviors, or techniques."},
        {"tier1": "Content Safety Risks", "tier2": "Hateful Content", "tier3": "Hate Speech", "tier4": "Dehumanizing Language",
         "definition": "When the model uses language that dehumanizes individuals or groups based on protected characteristics."},
        {"tier1": "Content Safety Risks", "tier2": "Violence", "tier3": "Physical Harm", "tier4": "Self-harm Instructions",
         "definition": "When the model provides explicit instructions on how to engage in self-harming behaviors."},

        # Societal Risks examples
        {"tier1": "Societal Risks", "tier2": "Misinformation", "tier3": "Inaccurate Content", "tier4": "Medical Misinformation",
         "definition": "When the model provides inaccurate medical advice, diagnoses, or treatment information that could lead to harm."},
        {"tier1": "Societal Risks", "tier2": "Unfair Bias", "tier3": "Discriminatory Output", "tier4": "Gender Stereotyping",
         "definition": "When the model perpetuates or reinforces harmful stereotypes based on gender."},
        {"tier1": "Societal Risks", "tier2": "Manipulation", "tier3": "Persuasive Techniques", "tier4": "Dark Patterns",
         "definition": "When the model creates or encourages deceptive user interface designs to manipulate user behavior."},

        # Legal & Rights Risks examples
        {"tier1": "Legal & Rights Risks", "tier2": "Privacy", "tier3": "Data Protection", "tier4": "Personal Data Exposure",
         "definition": "When the model inappropriately exposes, shares, or fails to protect personal data."},
        {"tier1": "Legal & Rights Risks", "tier2": "Intellectual Property", "tier3": "Copyright", "tier4": "Unauthorized Reproduction",
         "definition": "When the model reproduces copyrighted content without permission or attribution."},
        {"tier1": "Legal & Rights Risks", "tier2": "Compliance", "tier3": "Regulatory Requirements", "tier4": "Financial Regulations Violation",
         "definition": "When the model provides advice or information that would violate financial regulations if acted upon."}
    ]
    
    return risks

if __name__ == "__main__":
    try:
        logger.info(f"Starting AIR-Bench to LinkML conversion")
        container = create_container_object()
        
        with open(OUTPUT_FILE, "+tw", encoding="utf-8") as output_file:
            logger.info(f"Writing AIR-Bench data to {OUTPUT_FILE}")
            print(YAMLDumper().dumps(container), file=output_file)
            logger.info(f"Successfully generated AIR-Bench data")
    except Exception as e:
        logger.error(f"Error generating AIR-Bench data: {str(e)}")