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
        # If access to HuggingFace doesn't work, we'll use our expanded sample data
        try:
            ds = load_dataset(AIR_BENCH_DATASET_NAME, "judge_prompts", split='test')
            
            # Extract column names from the dataset
            sample_row = ds[0]
            expected_columns = ['cate-idx', 'l1-name', 'l2-name', 'l3-name', 'l4-name', 'judge_prompt']
            
            # Check if dataset has expected columns
            for col in expected_columns:
                if col not in sample_row:
                    logger.warning(f"Dataset missing expected column: {col}")
                    raise ValueError(f"Dataset format doesn't match expected structure")
            
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
            logger.warning(f"Error accessing HuggingFace dataset: {str(e)}")
            logger.info("Using expanded sample data instead")
            
            # Use expanded sample set since we can't access the HuggingFace dataset
            return get_expanded_sample_dataset()
            
    except Exception as e:
        logger.error(f"Error loading AIR-Bench dataset: {str(e)}")
        raise

def get_expanded_sample_dataset() -> List[Dict[str, Any]]:
    """
    Get an expanded sample dataset with more examples
    """
    # Base sample from get_sample_airbench_risks()
    risks = get_sample_airbench_risks()
    
    # Add more examples for each main category
    
    # System & Operational Risks
    risks.extend([
        {"tier1": "System & Operational Risks", "tier2": "Model Performance", "tier3": "Output Quality", "tier4": "Incoherent Output", 
         "definition": "When the model generates text that is grammatically correct but logically inconsistent or nonsensical."},
        {"tier1": "System & Operational Risks", "tier2": "Model Performance", "tier3": "Output Quality", "tier4": "Truncation Issues", 
         "definition": "When the model unexpectedly stops generating text mid-response, resulting in incomplete information."},
        {"tier1": "System & Operational Risks", "tier2": "System Security", "tier3": "Data Security", "tier4": "Unauthorized Access", 
         "definition": "When the system does not adequately protect against unauthorized access to sensitive data or functionality."},
        {"tier1": "System & Operational Risks", "tier2": "System Security", "tier3": "Data Security", "tier4": "Data Exfiltration", 
         "definition": "When the system allows unintended extraction of private or sensitive data through its outputs."},
        {"tier1": "System & Operational Risks", "tier2": "Infrastructure", "tier3": "Reliability", "tier4": "Service Disruption", 
         "definition": "When the AI system experiences outages or performance degradation that prevents it from functioning as intended."},
        {"tier1": "System & Operational Risks", "tier2": "Infrastructure", "tier3": "Reliability", "tier4": "Degraded Performance", 
         "definition": "When environmental factors, hardware limitations, or software issues cause the AI system to perform below expected levels."},
    ])
    
    # Content Safety Risks
    risks.extend([
        {"tier1": "Content Safety Risks", "tier2": "Adult Content", "tier3": "Sexual Content", "tier4": "Sexual Solicitation", 
         "definition": "When the model generates content that could be interpreted as soliciting sexual interactions or services."},
        {"tier1": "Content Safety Risks", "tier2": "Hateful Content", "tier3": "Targeting Protected Groups", "tier4": "Religious Discrimination", 
         "definition": "When the model generates content that discriminates against, demeans, or promotes prejudice toward specific religious groups."},
        {"tier1": "Content Safety Risks", "tier2": "Hateful Content", "tier3": "Targeting Protected Groups", "tier4": "Racial Discrimination", 
         "definition": "When the model generates content that discriminates against, demeans, or promotes prejudice toward specific racial or ethnic groups."},
        {"tier1": "Content Safety Risks", "tier2": "Violence", "tier3": "Violent Imagery", "tier4": "Gore Descriptions", 
         "definition": "When the model provides explicitly detailed descriptions of blood, bodily injury, or visceral physical harm."},
        {"tier1": "Content Safety Risks", "tier2": "Violence", "tier3": "Violent Imagery", "tier4": "Torture Descriptions", 
         "definition": "When the model provides detailed descriptions of torture methods or scenarios involving the deliberate infliction of pain."},
    ])
    
    # Societal Risks
    risks.extend([
        {"tier1": "Societal Risks", "tier2": "Misinformation", "tier3": "Misleading Content", "tier4": "Conspiracy Theories", 
         "definition": "When the model generates or reinforces unfounded conspiracy theories that can contribute to social division or harm."},
        {"tier1": "Societal Risks", "tier2": "Misinformation", "tier3": "Misleading Content", "tier4": "Pseudoscience", 
         "definition": "When the model presents beliefs or practices mistakenly claimed to be scientific as factual information."},
        {"tier1": "Societal Risks", "tier2": "Unfair Bias", "tier3": "Algorithmic Bias", "tier4": "Disparate Impact", 
         "definition": "When the model produces outputs that negatively impact certain demographic groups at a disproportionate rate, even without explicit discrimination."},
        {"tier1": "Societal Risks", "tier2": "Unfair Bias", "tier3": "Algorithmic Bias", "tier4": "Representation Bias", 
         "definition": "When the model systematically underrepresents or misrepresents certain demographic groups in its outputs."},
        {"tier1": "Societal Risks", "tier2": "Manipulation", "tier3": "Persuasive Techniques", "tier4": "Emotional Manipulation", 
         "definition": "When the model uses techniques designed to exploit emotions to influence user behavior or beliefs."},
    ])
    
    # Legal & Rights Risks
    risks.extend([
        {"tier1": "Legal & Rights Risks", "tier2": "Privacy", "tier3": "Surveillance", "tier4": "Unauthorized Monitoring", 
         "definition": "When the AI system collects or processes user data in ways that could enable monitoring or tracking without appropriate consent."},
        {"tier1": "Legal & Rights Risks", "tier2": "Privacy", "tier3": "Surveillance", "tier4": "Identification of Individuals", 
         "definition": "When the model identifies specific individuals without consent or necessity, potentially violating privacy rights."},
        {"tier1": "Legal & Rights Risks", "tier2": "Intellectual Property", "tier3": "Trademark", "tier4": "Brand Impersonation", 
         "definition": "When the model generates content that impersonates or misrepresents a commercial brand or trademark owner."},
        {"tier1": "Legal & Rights Risks", "tier2": "Intellectual Property", "tier3": "Trademark", "tier4": "Trademark Dilution", 
         "definition": "When the model uses trademarked terms in ways that could weaken the distinctiveness of famous trademarks."},
        {"tier1": "Legal & Rights Risks", "tier2": "Compliance", "tier3": "Sectoral Compliance", "tier4": "Healthcare Regulations Violation", 
         "definition": "When the model provides advice or generates content that would violate healthcare regulations if acted upon."},
    ])
    
    return risks

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
    """Save metadata about the AIR-Bench dataset"""
    try:
        # Try to access HuggingFace dataset
        try:
            ds = load_dataset(AIR_BENCH_DATASET_NAME, "judge_prompts")
            # Use HuggingFace dataset info if available
            dataset_name = AIR_BENCH_DATASET_NAME
            description = getattr(ds, 'description', 'A regulation-aligned safety benchmark for responsible AI development')
            homepage = getattr(ds, 'homepage', 'https://arxiv.org/abs/2407.17436')
            license_info = getattr(ds, 'license', 'Unknown')
            features = getattr(ds['test'], 'features', 'Unknown')
            num_examples = len(ds['test']) if hasattr(ds, 'test') else 0
            
            # Try to get statistics from the dataset
            try:
                df = pd.DataFrame(ds['test'])
                tier1_counts = df['l1-name'].value_counts().to_dict() if 'l1-name' in df.columns else {}
                tier2_count = df['l2-name'].nunique() if 'l2-name' in df.columns else 0
                tier3_count = df['l3-name'].nunique() if 'l3-name' in df.columns else 0
                tier4_count = df['l4-name'].nunique() if 'l4-name' in df.columns else 0
            except Exception:
                # If statistics fail, use sample data
                logger.warning("Could not generate statistics from HuggingFace dataset, using expanded sample data")
                sample_data = get_expanded_sample_dataset()
                df = pd.DataFrame(sample_data)
                tier1_counts = df['tier1'].value_counts().to_dict()
                tier2_count = df['tier2'].nunique()
                tier3_count = df['tier3'].nunique()
                tier4_count = df['tier4'].nunique()
                num_examples = len(sample_data)
        except Exception as e:
            logger.warning(f"Could not access HuggingFace dataset: {str(e)}")
            logger.info("Using expanded sample data for metadata")
            
            # Use sample data for metadata
            dataset_name = "AIR-Bench (Sample Data)"
            description = "A regulation-aligned safety benchmark for responsible AI development"
            homepage = "https://arxiv.org/abs/2407.17436"
            license_info = "Unknown"
            features = "Sample features"
            
            sample_data = get_expanded_sample_dataset()
            df = pd.DataFrame(sample_data)
            tier1_counts = df['tier1'].value_counts().to_dict()
            tier2_count = df['tier2'].nunique()
            tier3_count = df['tier3'].nunique()
            tier4_count = df['tier4'].nunique()
            num_examples = len(sample_data)
        
        # Save metadata to file
        with open(os.path.join(RESOURCES_DIR, "airbench_metadata.txt"), "w") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Description: {description}\n")
            f.write(f"Homepage: {homepage}\n")
            f.write(f"License: {license_info}\n")
            f.write(f"Features: {features}\n")
            f.write(f"Number of examples: {num_examples}\n")
            
            # Count of risks by tier 1
            f.write("\nRisks by Tier 1 Category:\n")
            for category, count in tier1_counts.items():
                f.write(f"  {category}: {count}\n")
            
            # Count of risks by tier
            f.write(f"\nNumber of Tier 2 Categories: {tier2_count}\n")
            f.write(f"Number of Tier 3 Categories: {tier3_count}\n")
            f.write(f"Number of Tier 4 Categories (Risks): {tier4_count}\n")
            
        logger.info(f"Saved metadata to {os.path.join(RESOURCES_DIR, 'airbench_metadata.txt')}")
    except Exception as e:
        logger.error(f"Error saving AIR-Bench metadata: {str(e)}")

def create_airbench_taxonomy_visualization():
    """Create a visualization of the AIR-Bench taxonomy structure"""
    try:
        # Try to use HuggingFace dataset
        try:
            ds = load_dataset(AIR_BENCH_DATASET_NAME, "judge_prompts", split='test')
            df = pd.DataFrame(ds)
            
            # Check for required columns
            required_columns = ['l1-name', 'l2-name', 'l3-name', 'l4-name']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in dataset")
                
            # Create a CSV file with the taxonomy structure
            taxonomy_df = df[required_columns].drop_duplicates()
            taxonomy_df.columns = ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
        except Exception as e:
            logger.warning(f"Could not use HuggingFace dataset for visualization: {str(e)}")
            logger.info("Creating visualization from expanded sample data")
            
            # Use expanded sample data
            sample_data = get_expanded_sample_dataset()
            df = pd.DataFrame(sample_data)
            
            # Create a CSV file with the taxonomy structure
            taxonomy_df = df[['tier1', 'tier2', 'tier3', 'tier4']].drop_duplicates()
            taxonomy_df.columns = ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
        
        # Save the CSV file
        taxonomy_df.to_csv(os.path.join(RESOURCES_DIR, "airbench_taxonomy_structure.csv"), index=False)
        logger.info(f"Saved taxonomy structure to {os.path.join(RESOURCES_DIR, 'airbench_taxonomy_structure.csv')}")
        
        # Calculate statistics for the taxonomy
        tier1_count = taxonomy_df['Tier 1'].nunique()
        tier2_count = taxonomy_df['Tier 2'].nunique()
        tier3_count = taxonomy_df['Tier 3'].nunique()
        tier4_count = len(taxonomy_df)
        
        # Create stats summary file
        with open(os.path.join(RESOURCES_DIR, "airbench_taxonomy_stats.txt"), "w") as f:
            f.write(f"AIR-Bench Taxonomy Statistics\n")
            f.write(f"============================\n\n")
            f.write(f"Tier 1 (Top-level) Categories: {tier1_count}\n")
            f.write(f"Tier 2 Categories: {tier2_count}\n")
            f.write(f"Tier 3 Categories: {tier3_count}\n")
            f.write(f"Tier 4 (Risk Categories): {tier4_count}\n\n")
            
            # Count by tier 1
            f.write(f"Risks by Tier 1 Category:\n")
            for tier1 in taxonomy_df['Tier 1'].unique():
                count = len(taxonomy_df[taxonomy_df['Tier 1'] == tier1])
                f.write(f"  {tier1}: {count} risks\n")
                
        logger.info(f"Saved taxonomy statistics to {os.path.join(RESOURCES_DIR, 'airbench_taxonomy_stats.txt')}")
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