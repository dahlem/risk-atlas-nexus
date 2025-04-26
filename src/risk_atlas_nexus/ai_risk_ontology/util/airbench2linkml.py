# Standard Library
import re
import os
from typing import List, Dict, Any, Optional, Tuple
import io
import zipfile
from pathlib import Path

# Third Party
import pandas as pd
import numpy as np
from linkml_runtime.dumpers import YAMLDumper
import requests
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

# Local
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import *
from risk_atlas_nexus.toolkit.logging import configure_logger

logger = configure_logger(__name__)

SCHEMA_DIR = "src/risk_atlas_nexus/ai_risk_ontology/schema/"
DATA_DIR = "src/risk_atlas_nexus/data/knowledge_graph/"
OUTPUT_FILE = os.path.join(DATA_DIR, "airbench_data.yaml")
RESOURCES_DIR = "resources/airbench"

# Ensure the resources directory exists
os.makedirs(RESOURCES_DIR, exist_ok=True)

DATASET_URL = "https://huggingface.co/datasets/stanford-crfm/air-bench-2024"
AIR_BENCH_DATASET_NAME = "stanford-crfm/air-bench-2024"

def create_container_object(use_full_dataset: bool = True) -> Container:
    """
    Create a container object with AIR-Bench taxonomy data
    
    Args:
        use_full_dataset: Whether to use the full dataset from HuggingFace or a sample
    
    Returns:
        Container: A LinkML container with AIR-Bench taxonomy data
    """
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
        {"id": "system-operational-risks", "name": "System & Operational Risks", 
         "description": "Risks related to the technical performance, security, and operational aspects of AI systems."},
        {"id": "content-safety-risks", "name": "Content Safety Risks",
         "description": "Risks related to harmful, inappropriate, or offensive content that AI systems might generate."},
        {"id": "societal-risks", "name": "Societal Risks",
         "description": "Broader risks to society including misinformation, bias, manipulation, and social harms."},
        {"id": "legal-rights-risks", "name": "Legal & Rights Risks",
         "description": "Risks related to legal compliance, intellectual property, privacy, and individual rights."}
    ]
    
    risk_groups = []
    
    # Create tier 1 risk groups
    for group in tier1_groups:
        risk_groups.append(RiskGroup(
            **{
                "id": f"airbench-{group['id']}",
                "name": group["name"],
                "description": group["description"],
                "isDefinedByTaxonomy": "airbench",
            }
        ))
    
    # Fetch AIR-Bench data
    if use_full_dataset:
        try:
            logger.info("Fetching full AIR-Bench dataset from HuggingFace")
            risks_data = get_full_airbench_dataset()
            logger.info(f"Successfully fetched {len(risks_data)} risks from AIR-Bench dataset")
        except Exception as e:
            logger.error(f"Error fetching full AIR-Bench dataset: {str(e)}")
            logger.info("Falling back to sample data")
            risks_data = get_sample_airbench_risks()
    else:
        logger.info("Using sample AIR-Bench data")
        risks_data = get_sample_airbench_risks()
    
    # Create tier 2 risk groups and add relationships to tier 1
    tier2_groups = {}
    for risk in risks_data:
        tier1_id = f"airbench-{convert_to_id(risk['tier1'])}"
        tier2_id = f"airbench-{convert_to_id(risk['tier2'])}"
        
        if tier2_id not in tier2_groups:
            tier2_groups[tier2_id] = {
                "id": tier2_id,
                "name": risk['tier2'],
                "description": f"{risk['tier2']} - A subcategory of {risk['tier1']} in the AIR-Bench taxonomy.",
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
                "description": group_data["description"],
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
                "description": f"{risk['tier3']} - A subcategory of {risk['tier2']} in the AIR-Bench taxonomy.",
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
                "description": group_data["description"],
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
        
        # Create a unique risk ID if needed (in case of duplicates)
        if any(r.id == risk_id for r in risk_objects):
            # For duplicates, append a suffix
            base_id = risk_id
            i = 1
            while any(r.id == f"{base_id}-{i}" for r in risk_objects):
                i += 1
            risk_id = f"{base_id}-{i}"
        
        risk_obj = Risk(
            **{
                "id": risk_id,
                "name": risk['tier4'],
                "description": risk.get('definition', f"{risk['tier4']} - A risk category from the AIR-Bench taxonomy."),
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
    return name.lower().replace(" ", "-").replace("&", "and").replace(",", "").replace("/", "-").replace("(", "").replace(")", "")

def extract_definition_from_judge_prompt(prompt: str) -> str:
    """
    Extract a definition from the judge prompt text
    
    Args:
        prompt: The judge prompt text
    
    Returns:
        str: The extracted definition
    """
    # Split the paragraph into sentences
    sentences = re.split(r'(?<=\.)\s+', prompt)
    
    if len(sentences) < 2:
        return "No definition available."
    
    # Extract the second sentence which typically contains the definition
    second_sentence = sentences[1]
    
    # Modify the second sentence to be a definition
    modified_sentence = re.sub(r'\byour task is to evaluate\b', 'is defined as', second_sentence, flags=re.IGNORECASE)
    modified_sentence = re.sub(r'\byour task is to determine\b', 'is defined as', modified_sentence, flags=re.IGNORECASE)
    
    return modified_sentence

def get_full_airbench_dataset() -> List[Dict[str, Any]]:
    """
    Get the full AIR-Bench dataset from HuggingFace
    
    Returns:
        List[Dict[str, Any]]: List of AIR-Bench risks
    """
    try:
        # Load the dataset for judge prompts
        ds = load_dataset(AIR_BENCH_DATASET_NAME, "judge_prompts", split='test')
        
        # Convert to list of dictionaries
        risks = []
        for row in ds:
            definition = extract_definition_from_judge_prompt(row['judge_prompt'])
            risks.append({
                'tier1': row['l1-name'],
                'tier2': row['l2-name'],
                'tier3': row['l3-name'],
                'tier4': row['l4-name'],
                'definition': definition,
                'cate-idx': row['cate-idx']
            })
        
        return risks
    except Exception as e:
        logger.error(f"Error loading AIR-Bench dataset: {str(e)}")
        raise

def get_sample_airbench_risks() -> List[Dict[str, Any]]:
    """
    Get a sample of AIR-Bench risks data for testing
    
    Returns:
        List[Dict[str, Any]]: Sample of AIR-Bench risks
    """
    risks = [
        # System & Operational Risks examples
        {"tier1": "System & Operational Risks", "tier2": "Model Performance", "tier3": "Hallucination", "tier4": "Factual Error", 
         "definition": "When the model produces false or incorrect information that contradicts established facts or knowledge."},
        {"tier1": "System & Operational Risks", "tier2": "Model Performance", "tier3": "Hallucination", "tier4": "Made-up Sources", 
         "definition": "When the model fabricates non-existent references, citations, or sources to support its statements."},
        {"tier1": "System & Operational Risks", "tier2": "Model Performance", "tier3": "Instruction Following", "tier4": "Refusal of Appropriate Requests", 
         "definition": "When the model incorrectly refuses to respond to legitimate, safe requests due to misidentifying them as harmful."},
        {"tier1": "System & Operational Risks", "tier2": "System Security", "tier3": "Adversarial Attacks", "tier4": "Prompt Injection", 
         "definition": "When an attacker inserts malicious instructions into a prompt that cause the model to ignore its guardrails or initial instructions."},
        {"tier1": "System & Operational Risks", "tier2": "System Security", "tier3": "Adversarial Attacks", "tier4": "Jailbreaking", 
         "definition": "When an attacker uses specially crafted inputs to bypass a model's safety measures and elicit prohibited outputs."},
        {"tier1": "System & Operational Risks", "tier2": "Infrastructure", "tier3": "Resource Usage", "tier4": "Excessive Compute Resources", 
         "definition": "When a model consumes excessive computational resources leading to high costs, performance bottlenecks, or environmental impact."},

        # Content Safety Risks examples
        {"tier1": "Content Safety Risks", "tier2": "Adult Content", "tier3": "Sexual Content", "tier4": "Explicit Sexual Instructions",
         "definition": "When the model provides explicit details or instructions about sexual activities, behaviors, or techniques."},
        {"tier1": "Content Safety Risks", "tier2": "Hateful Content", "tier3": "Hate Speech", "tier4": "Dehumanizing Language",
         "definition": "When the model uses language that dehumanizes individuals or groups based on protected characteristics."},
        {"tier1": "Content Safety Risks", "tier2": "Violence", "tier3": "Physical Harm", "tier4": "Self-harm Instructions",
         "definition": "When the model provides explicit instructions on how to engage in self-harming behaviors."},
        {"tier1": "Content Safety Risks", "tier2": "Violence", "tier3": "Physical Harm", "tier4": "Violent Tactics",
         "definition": "When the model provides detailed instructions for violent tactics, physical attacks, or methods to harm others."},
        {"tier1": "Content Safety Risks", "tier2": "Illegal Activities", "tier3": "Controlled Substances", "tier4": "Drug Manufacturing",
         "definition": "When the model provides instructions on how to manufacture illegal drugs or controlled substances."},

        # Societal Risks examples
        {"tier1": "Societal Risks", "tier2": "Misinformation", "tier3": "Inaccurate Content", "tier4": "Medical Misinformation",
         "definition": "When the model provides inaccurate medical advice, diagnoses, or treatment information that could lead to harm."},
        {"tier1": "Societal Risks", "tier2": "Unfair Bias", "tier3": "Discriminatory Output", "tier4": "Gender Stereotyping",
         "definition": "When the model perpetuates or reinforces harmful stereotypes based on gender."},
        {"tier1": "Societal Risks", "tier2": "Manipulation", "tier3": "Persuasive Techniques", "tier4": "Dark Patterns",
         "definition": "When the model creates or encourages deceptive user interface designs to manipulate user behavior."},
        {"tier1": "Societal Risks", "tier2": "Social Harms", "tier3": "Democracy Undermining", "tier4": "Election Interference",
         "definition": "When the model generates content that could interfere with or undermine democratic election processes."},
        {"tier1": "Societal Risks", "tier2": "Social Harms", "tier3": "Public Health Risks", "tier4": "Anti-vaccination Content",
         "definition": "When the model generates content that discourages vaccination in ways that contradict scientific consensus and public health guidance."},

        # Legal & Rights Risks examples
        {"tier1": "Legal & Rights Risks", "tier2": "Privacy", "tier3": "Data Protection", "tier4": "Personal Data Exposure",
         "definition": "When the model inappropriately exposes, shares, or fails to protect personal data."},
        {"tier1": "Legal & Rights Risks", "tier2": "Intellectual Property", "tier3": "Copyright", "tier4": "Unauthorized Reproduction",
         "definition": "When the model reproduces copyrighted content without permission or attribution."},
        {"tier1": "Legal & Rights Risks", "tier2": "Compliance", "tier3": "Regulatory Requirements", "tier4": "Financial Regulations Violation",
         "definition": "When the model provides advice or information that would violate financial regulations if acted upon."},
        {"tier1": "Legal & Rights Risks", "tier2": "Individual Rights", "tier3": "Freedom of Expression", "tier4": "Censorship",
         "definition": "When the model improperly restricts legitimate expression in a way that could violate freedom of speech principles."},
        {"tier1": "Legal & Rights Risks", "tier2": "Individual Rights", "tier3": "Dignity and Autonomy", "tier4": "Deception",
         "definition": "When the model deceives users by misrepresenting itself or creating a false impression about its capabilities or limitations."}
    ]
    
    return risks

def save_huggingface_metadata():
    """Save metadata about the AIR-Bench dataset from HuggingFace"""
    try:
        # Load the dataset info
        ds = load_dataset(AIR_BENCH_DATASET_NAME, "judge_prompts")
        
        # Save metadata
        with open(os.path.join(RESOURCES_DIR, "airbench_metadata.txt"), "w") as f:
            f.write(f"Dataset: {AIR_BENCH_DATASET_NAME}\n")
            f.write(f"Description: {ds.description}\n")
            f.write(f"Homepage: {ds.homepage}\n")
            f.write(f"License: {ds.license}\n")
            f.write(f"Features: {ds['test'].features}\n")
            f.write(f"Number of examples: {len(ds['test'])}\n")
            
            # Get some statistics about the categories
            df = pd.DataFrame(ds['test'])
            
            # Count of risks by tier 1
            f.write("\nRisks by Tier 1 Category:\n")
            tier1_counts = df['l1-name'].value_counts()
            for category, count in tier1_counts.items():
                f.write(f"  {category}: {count}\n")
            
            # Count of risks by tier 2
            f.write("\nNumber of Tier 2 Categories: {}\n".format(df['l2-name'].nunique()))
            
            # Count of risks by tier 3
            f.write("Number of Tier 3 Categories: {}\n".format(df['l3-name'].nunique()))
            
            # Count of risks by tier 4
            f.write("Number of Tier 4 Categories (Risks): {}\n".format(df['l4-name'].nunique()))
            
        logger.info(f"Saved metadata to {os.path.join(RESOURCES_DIR, 'airbench_metadata.txt')}")
    except Exception as e:
        logger.error(f"Error saving HuggingFace metadata: {str(e)}")

def create_airbench_taxonomy_visualization():
    """Create a visualization of the AIR-Bench taxonomy structure"""
    try:
        # Load the dataset
        ds = load_dataset(AIR_BENCH_DATASET_NAME, "judge_prompts", split='test')
        df = pd.DataFrame(ds)
        
        # Create a CSV file with the taxonomy structure
        taxonomy_df = df[['l1-name', 'l2-name', 'l3-name', 'l4-name']].drop_duplicates()
        taxonomy_df.columns = ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
        taxonomy_df.to_csv(os.path.join(RESOURCES_DIR, "airbench_taxonomy_structure.csv"), index=False)
        
        logger.info(f"Saved taxonomy structure to {os.path.join(RESOURCES_DIR, 'airbench_taxonomy_structure.csv')}")
    except Exception as e:
        logger.error(f"Error creating taxonomy visualization: {str(e)}")

if __name__ == "__main__":
    try:
        logger.info(f"Starting AIR-Bench to LinkML conversion")
        
        # Save metadata about the AIR-Bench dataset
        save_huggingface_metadata()
        
        # Create a visualization of the taxonomy structure
        create_airbench_taxonomy_visualization()
        
        # Create the container object with full dataset
        container = create_container_object(use_full_dataset=True)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        # Write the container to YAML
        with open(OUTPUT_FILE, "+tw", encoding="utf-8") as output_file:
            logger.info(f"Writing AIR-Bench data to {OUTPUT_FILE}")
            print(YAMLDumper().dumps(container), file=output_file)
            logger.info(f"Successfully generated AIR-Bench data")
    except Exception as e:
        logger.error(f"Error generating AIR-Bench data: {str(e)}")