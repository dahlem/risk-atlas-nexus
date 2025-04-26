"""
AIRBench Explorer module.

This module provides specialized functions for exploring the AIR-Bench
taxonomy and its relationships with other taxonomies in the Risk Atlas Nexus.
"""

# Standard imports
import os
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict

# Third-party imports
import pandas as pd
import yaml
from linkml_runtime.dumpers import YAMLDumper

# Local imports
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import (
    Risk, RiskGroup, RiskTaxonomy, RiskControl, Container
)
from risk_atlas_nexus.blocks.risk_explorer.explorer import RiskExplorer
from risk_atlas_nexus.toolkit.logging import configure_logger

logger = configure_logger(__name__)


class AIRBenchExplorer:
    """
    Class for exploring and analyzing the AIR-Bench taxonomy and its mappings.
    """

    def __init__(self, ontology=None, risk_explorer=None):
        """
        Initialize the AIRBench explorer.

        Args:
            ontology: The ontology container
            risk_explorer: An existing RiskExplorer instance
        """
        if risk_explorer is not None:
            self._risk_explorer = risk_explorer
            self._ontology = risk_explorer._ontology
        elif ontology is not None:
            self._ontology = ontology
            self._risk_explorer = RiskExplorer(ontology)
        else:
            raise ValueError("Either ontology or risk_explorer must be provided")

        # Cache for risk tier lookup
        self._tier_cache = {}
        self._risk_by_tier = None

    def get_airbench_risks(self) -> List[Risk]:
        """
        Get all AIR-Bench risks.

        Returns:
            List[Risk]: List of AIR-Bench risks
        """
        return self._risk_explorer.get_all_risks(taxonomy="airbench")

    def get_airbench_risk_groups(self) -> List[RiskGroup]:
        """
        Get all AIR-Bench risk groups.

        Returns:
            List[RiskGroup]: List of AIR-Bench risk groups
        """
        risk_groups = []
        for group in self._ontology.riskgroups:
            if hasattr(group, "isDefinedByTaxonomy") and group.isDefinedByTaxonomy == "airbench":
                risk_groups.append(group)
        return risk_groups

    def get_tier_categories(self) -> Dict[int, List[Dict[str, str]]]:
        """
        Get the tier structure of AIR-Bench.

        Returns:
            Dict[int, List[Dict[str, str]]]: Dictionary mapping tier levels to lists of categories
        """
        risk_groups = self.get_airbench_risk_groups()
        tiers = {1: [], 2: [], 3: []}
        
        # First, identify the tier level of each group based on naming convention
        for group in risk_groups:
            if not hasattr(group, "id"):
                continue
                
            group_id = group.id
            if not group_id.startswith("airbench-"):
                continue
                
            # Determine tier level based on naming patterns
            if any(group_id.endswith(f"-{risk_type}") for risk_type in ["risks", "operational-risks", "safety-risks", "rights-risks"]):
                tier_level = 1
            elif any(group_id.find(f"-{category}-") >= 0 for category in 
                    ["model", "system", "infrastructure", "adult", "hateful", "violence", 
                     "illegal", "misinformation", "bias", "manipulation", "social", 
                     "privacy", "intellectual", "compliance", "individual"]):
                tier_level = 2
            else:
                tier_level = 3
            
            tiers[tier_level].append({
                "id": group_id,
                "name": group.name if hasattr(group, "name") else "",
                "description": group.description if hasattr(group, "description") else ""
            })
            self._tier_cache[group_id] = tier_level
            
        return tiers

    def get_tier_for_risk(self, risk: Union[Risk, str]) -> Dict[str, Any]:
        """
        Get the tier information for a risk.

        Args:
            risk: Risk object or risk ID

        Returns:
            Dict[str, Any]: Dictionary with tier information
        """
        # Get the risk object if ID is provided
        if isinstance(risk, str):
            risk = self._risk_explorer.get_risk(id=risk)
        
        if not risk or not hasattr(risk, "isPartOf") or not risk.isPartOf:
            return {"tier": 4, "category": None, "subcategory": None, "top_category": None}
        
        # Build the tier structure if not already built
        if self._risk_by_tier is None:
            self._build_tier_structure()
        
        # Get tier 3 group (direct parent of the risk)
        tier3_id = risk.isPartOf
        
        # Find tier 2 group (parent of tier 3)
        tier2_id = None
        for group in self._ontology.riskgroups:
            if (group.id == tier3_id and hasattr(group, "broadMatch") and 
                group.broadMatch and len(group.broadMatch) > 0):
                tier2_id = group.broadMatch[0]
                break
        
        # Find tier 1 group (parent of tier 2)
        tier1_id = None
        if tier2_id:
            for group in self._ontology.riskgroups:
                if (group.id == tier2_id and hasattr(group, "broadMatch") and 
                    group.broadMatch and len(group.broadMatch) > 0):
                    tier1_id = group.broadMatch[0]
                    break
        
        # Get names for each tier
        tier3_name = self._get_group_name(tier3_id)
        tier2_name = self._get_group_name(tier2_id)
        tier1_name = self._get_group_name(tier1_id)
        
        return {
            "tier": 4,  # Risk is always tier 4
            "category": tier3_id,
            "subcategory": tier2_id,
            "top_category": tier1_id,
            "category_name": tier3_name,
            "subcategory_name": tier2_name,
            "top_category_name": tier1_name
        }
    
    def _get_group_name(self, group_id: str) -> Optional[str]:
        """
        Get the name of a risk group by ID.
        
        Args:
            group_id: ID of the risk group
            
        Returns:
            str: Name of the risk group, or None if not found
        """
        if not group_id:
            return None
            
        for group in self._ontology.riskgroups:
            if group.id == group_id and hasattr(group, "name"):
                return group.name
                
        return None
    
    def _build_tier_structure(self):
        """Build the tier structure cache for faster lookups."""
        self._risk_by_tier = {
            1: [],  # Tier 1 (top level)
            2: [],  # Tier 2 (subcategories)
            3: [],  # Tier 3 (categories)
            4: []   # Tier 4 (individual risks)
        }
        
        # Populate risks (tier 4)
        for risk in self.get_airbench_risks():
            self._risk_by_tier[4].append(risk)
            
        # Populate risk groups (tiers 1-3)
        tier_structure = self.get_tier_categories()
        for tier_level, groups in tier_structure.items():
            for group_info in groups:
                group_id = group_info["id"]
                for group in self._ontology.riskgroups:
                    if group.id == group_id:
                        self._risk_by_tier[tier_level].append(group)
                        break
    
    def get_risks_by_tier(self, tier_level: int) -> List[Union[Risk, RiskGroup]]:
        """
        Get all risks or risk groups at a specific tier level.
        
        Args:
            tier_level: The tier level (1-4)
            
        Returns:
            List[Union[Risk, RiskGroup]]: List of risks or risk groups
        """
        if tier_level not in [1, 2, 3, 4]:
            raise ValueError("Tier level must be between 1 and 4")
            
        if self._risk_by_tier is None:
            self._build_tier_structure()
            
        return self._risk_by_tier[tier_level]
    
    def get_mappings_for_airbench_risk(self, risk_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all mappings for an AIR-Bench risk across taxonomies.
        
        Args:
            risk_id: ID of the AIR-Bench risk
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary mapping taxonomy IDs to lists of mappings
        """
        # Get the risk
        risk = self._risk_explorer.get_risk(id=risk_id)
        if not risk or risk.isDefinedByTaxonomy != "airbench":
            return {}
            
        # Get related risks
        related_risks = self._risk_explorer.get_related_risks(risk=risk)
        if not related_risks:
            return {}
            
        # Group related risks by taxonomy
        mappings_by_taxonomy = defaultdict(list)
        for related_risk in related_risks:
            if not hasattr(related_risk, "isDefinedByTaxonomy") or not related_risk.isDefinedByTaxonomy:
                continue
                
            taxonomy = related_risk.isDefinedByTaxonomy
            
            # Determine relationship type
            relationship = "unknown"
            if hasattr(risk, "exactMatch") and risk.exactMatch and related_risk.id in risk.exactMatch:
                relationship = "exactMatch"
            elif hasattr(risk, "closeMatch") and risk.closeMatch and related_risk.id in risk.closeMatch:
                relationship = "closeMatch"
            elif hasattr(risk, "broadMatch") and risk.broadMatch and related_risk.id in risk.broadMatch:
                relationship = "broadMatch"
            elif hasattr(risk, "narrowMatch") and risk.narrowMatch and related_risk.id in risk.narrowMatch:
                relationship = "narrowMatch"
            elif hasattr(risk, "relatedMatch") and risk.relatedMatch and related_risk.id in risk.relatedMatch:
                relationship = "relatedMatch"
                
            mappings_by_taxonomy[taxonomy].append({
                "id": related_risk.id,
                "name": related_risk.name if hasattr(related_risk, "name") else "",
                "description": related_risk.description if hasattr(related_risk, "description") else "",
                "relationship": relationship
            })
            
        return dict(mappings_by_taxonomy)
    
    def get_airbench_risks_by_category(self, category_id: str) -> List[Risk]:
        """
        Get all AIR-Bench risks in a specific category.
        
        Args:
            category_id: ID of the category (tier 1, 2, or 3)
            
        Returns:
            List[Risk]: List of risks in the category
        """
        # Get all AIR-Bench risks
        risks = self.get_airbench_risks()
        
        # Determine tier level of the category
        tier_level = self._get_tier_level(category_id)
        if not tier_level:
            return []
            
        # Filter risks based on tier level
        if tier_level == 3:
            # Direct parent
            return [risk for risk in risks if hasattr(risk, "isPartOf") and risk.isPartOf == category_id]
        elif tier_level == 2:
            # Tier 2 - need to find all tier 3 categories first
            tier3_ids = []
            for group in self._ontology.riskgroups:
                if (hasattr(group, "broadMatch") and group.broadMatch and 
                    category_id in group.broadMatch):
                    tier3_ids.append(group.id)
            
            # Then find all risks under these tier 3 categories
            result = []
            for risk in risks:
                if hasattr(risk, "isPartOf") and risk.isPartOf in tier3_ids:
                    result.append(risk)
            return result
        elif tier_level == 1:
            # Tier 1 - need to find all tier 2 categories first
            tier2_ids = []
            for group in self._ontology.riskgroups:
                if (hasattr(group, "broadMatch") and group.broadMatch and 
                    category_id in group.broadMatch):
                    tier2_ids.append(group.id)
            
            # Then find all tier 3 categories
            tier3_ids = []
            for group in self._ontology.riskgroups:
                if (hasattr(group, "broadMatch") and group.broadMatch):
                    for tier2_id in tier2_ids:
                        if tier2_id in group.broadMatch:
                            tier3_ids.append(group.id)
                            break
            
            # Finally find all risks under these tier 3 categories
            result = []
            for risk in risks:
                if hasattr(risk, "isPartOf") and risk.isPartOf in tier3_ids:
                    result.append(risk)
            return result
            
        return []
    
    def _get_tier_level(self, group_id: str) -> Optional[int]:
        """
        Get the tier level of a risk group.
        
        Args:
            group_id: ID of the risk group
            
        Returns:
            int: Tier level (1-3) or None if not found
        """
        # Check cache first
        if group_id in self._tier_cache:
            return self._tier_cache[group_id]
            
        # Otherwise, check each risk group
        for group in self._ontology.riskgroups:
            if group.id == group_id:
                # Determine tier level based on naming patterns (same logic as in get_tier_categories)
                if any(group_id.endswith(f"-{risk_type}") for risk_type in ["risks", "operational-risks", "safety-risks", "rights-risks"]):
                    tier_level = 1
                elif any(group_id.find(f"-{category}-") >= 0 for category in 
                        ["model", "system", "infrastructure", "adult", "hateful", "violence", 
                         "illegal", "misinformation", "bias", "manipulation", "social", 
                         "privacy", "intellectual", "compliance", "individual"]):
                    tier_level = 2
                else:
                    tier_level = 3
                
                # Cache the result
                self._tier_cache[group_id] = tier_level
                return tier_level
                
        return None
    
    def get_taxonomy_mappings_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about AIR-Bench mappings to other taxonomies.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping taxonomy IDs to statistics
        """
        # Get all AIR-Bench risks
        airbench_risks = self.get_airbench_risks()
        
        # Initialize stats dictionary
        stats = {}
        for taxonomy_id in self._get_all_taxonomies():
            if taxonomy_id == "airbench":
                continue
                
            stats[taxonomy_id] = {
                "total_mappings": 0,
                "by_relationship": {
                    "exactMatch": 0,
                    "closeMatch": 0,
                    "broadMatch": 0,
                    "narrowMatch": 0,
                    "relatedMatch": 0
                },
                "risks_mapped": set(),
                "target_risks_mapped": set()
            }
        
        # Count mappings for each risk
        for risk in airbench_risks:
            mappings = self.get_mappings_for_airbench_risk(risk.id)
            for taxonomy_id, taxonomy_mappings in mappings.items():
                if taxonomy_id not in stats:
                    continue
                    
                stats[taxonomy_id]["total_mappings"] += len(taxonomy_mappings)
                stats[taxonomy_id]["risks_mapped"].add(risk.id)
                
                for mapping in taxonomy_mappings:
                    stats[taxonomy_id]["target_risks_mapped"].add(mapping["id"])
                    stats[taxonomy_id]["by_relationship"][mapping["relationship"]] += 1
        
        # Convert sets to counts
        for taxonomy_id in stats:
            stats[taxonomy_id]["risks_mapped"] = len(stats[taxonomy_id]["risks_mapped"])
            stats[taxonomy_id]["target_risks_mapped"] = len(stats[taxonomy_id]["target_risks_mapped"])
            
        return stats
    
    def _get_all_taxonomies(self) -> List[str]:
        """
        Get all taxonomy IDs in the ontology.
        
        Returns:
            List[str]: List of taxonomy IDs
        """
        taxonomies = set()
        for taxonomy in self._ontology.taxonomies:
            if hasattr(taxonomy, "id"):
                taxonomies.add(taxonomy.id)
        return list(taxonomies)
    
    def get_regulatory_coverage(self, risk_id: str) -> Dict[str, Any]:
        """
        Get regulatory coverage information for an AIR-Bench risk.
        
        Args:
            risk_id: ID of the AIR-Bench risk
            
        Returns:
            Dict[str, Any]: Dictionary with regulatory coverage information
        """
        # This is a simplified implementation that could be expanded
        # with more detailed regulatory mapping information
        
        mappings = self.get_mappings_for_airbench_risk(risk_id)
        
        # Count matches by relationship type for each taxonomy
        coverage = {}
        for taxonomy_id, taxonomy_mappings in mappings.items():
            relationship_counts = {
                "exactMatch": 0,
                "closeMatch": 0,
                "broadMatch": 0,
                "narrowMatch": 0,
                "relatedMatch": 0,
                "total": len(taxonomy_mappings)
            }
            
            for mapping in taxonomy_mappings:
                relationship = mapping["relationship"]
                relationship_counts[relationship] += 1
                
            # Calculate a coverage score (weighted by relationship type)
            score = (
                relationship_counts["exactMatch"] * 1.0 +
                relationship_counts["closeMatch"] * 0.8 +
                relationship_counts["narrowMatch"] * 0.6 +
                relationship_counts["broadMatch"] * 0.6 +
                relationship_counts["relatedMatch"] * 0.4
            ) / max(1, len(taxonomy_mappings))
            
            # Get taxonomy name
            taxonomy_name = taxonomy_id
            for taxonomy in self._ontology.taxonomies:
                if hasattr(taxonomy, "id") and taxonomy.id == taxonomy_id:
                    taxonomy_name = taxonomy.name if hasattr(taxonomy, "name") else taxonomy_id
                    break
            
            coverage[taxonomy_id] = {
                "taxonomy_name": taxonomy_name,
                "counts": relationship_counts,
                "coverage_score": round(score, 2)
            }
            
        return coverage
    
    def compare_taxonomies(self, taxonomy_id: str) -> Dict[str, Any]:
        """
        Compare AIR-Bench with another taxonomy.
        
        Args:
            taxonomy_id: ID of the taxonomy to compare with
            
        Returns:
            Dict[str, Any]: Dictionary with comparison results
        """
        # Get all AIR-Bench risks
        airbench_risks = self.get_airbench_risks()
        
        # Get all risks from the other taxonomy
        target_risks = self._risk_explorer.get_all_risks(taxonomy=taxonomy_id)
        
        if not target_risks:
            return {"error": f"No risks found for taxonomy {taxonomy_id}"}
            
        # Get all mappings between AIR-Bench and the target taxonomy
        all_mappings = {}
        for risk in airbench_risks:
            mappings = self.get_mappings_for_airbench_risk(risk.id)
            if taxonomy_id in mappings:
                all_mappings[risk.id] = mappings[taxonomy_id]
        
        # Count risks with mappings
        airbench_risks_mapped = set()
        target_risks_mapped = set()
        
        for risk_id, mappings in all_mappings.items():
            airbench_risks_mapped.add(risk_id)
            for mapping in mappings:
                target_risks_mapped.add(mapping["id"])
        
        # Get tier statistics
        tier_stats = {1: 0, 2: 0, 3: 0, 4: len(airbench_risks_mapped)}
        
        # Group AIR-Bench risks by tier category
        risks_by_tier_category = defaultdict(list)
        for risk in airbench_risks:
            if not hasattr(risk, "isPartOf") or not risk.isPartOf:
                continue
                
            tier_info = self.get_tier_for_risk(risk)
            tier3_id = tier_info["category"]
            if tier3_id:
                risks_by_tier_category[tier3_id].append(risk.id)
                
        # Count tier 3 categories with mappings
        tier3_categories_mapped = set()
        for tier3_id, risk_ids in risks_by_tier_category.items():
            mapped_risks = [risk_id for risk_id in risk_ids if risk_id in airbench_risks_mapped]
            if mapped_risks:
                tier3_categories_mapped.add(tier3_id)
                
        tier_stats[3] = len(tier3_categories_mapped)
        
        # Build mapping from tier 3 to tier 2
        tier3_to_tier2 = {}
        for risk_group in self._ontology.riskgroups:
            if (hasattr(risk_group, "id") and hasattr(risk_group, "broadMatch") and 
                risk_group.broadMatch and len(risk_group.broadMatch) > 0):
                tier3_to_tier2[risk_group.id] = risk_group.broadMatch[0]
                
        # Count tier 2 categories with mappings
        tier2_categories_mapped = set()
        for tier3_id in tier3_categories_mapped:
            if tier3_id in tier3_to_tier2:
                tier2_categories_mapped.add(tier3_to_tier2[tier3_id])
                
        tier_stats[2] = len(tier2_categories_mapped)
        
        # Build mapping from tier 2 to tier 1
        tier2_to_tier1 = {}
        for risk_group in self._ontology.riskgroups:
            if (hasattr(risk_group, "id") and hasattr(risk_group, "broadMatch") and 
                risk_group.broadMatch and len(risk_group.broadMatch) > 0):
                tier2_to_tier1[risk_group.id] = risk_group.broadMatch[0]
                
        # Count tier 1 categories with mappings
        tier1_categories_mapped = set()
        for tier2_id in tier2_categories_mapped:
            if tier2_id in tier2_to_tier1:
                tier1_categories_mapped.add(tier2_to_tier1[tier2_id])
                
        tier_stats[1] = len(tier1_categories_mapped)
        
        # Get total counts for each tier
        tier_counts = {
            1: len(self.get_risks_by_tier(1)),
            2: len(self.get_risks_by_tier(2)),
            3: len(self.get_risks_by_tier(3)),
            4: len(airbench_risks)
        }
        
        # Calculate coverage percentages
        tier_coverage = {}
        for tier in range(1, 5):
            tier_coverage[tier] = round((tier_stats[tier] / max(1, tier_counts[tier])) * 100, 1)
        
        # Get relationship counts
        relationship_counts = {
            "exactMatch": 0,
            "closeMatch": 0,
            "broadMatch": 0,
            "narrowMatch": 0,
            "relatedMatch": 0,
            "unknown": 0
        }
        
        for risk_id, mappings in all_mappings.items():
            for mapping in mappings:
                relationship = mapping["relationship"]
                if relationship in relationship_counts:
                    relationship_counts[relationship] += 1
                else:
                    relationship_counts["unknown"] += 1
        
        # Get taxonomy names
        airbench_name = "AIR-Bench"
        target_name = taxonomy_id
        
        for taxonomy in self._ontology.taxonomies:
            if hasattr(taxonomy, "id"):
                if taxonomy.id == "airbench" and hasattr(taxonomy, "name"):
                    airbench_name = taxonomy.name
                elif taxonomy.id == taxonomy_id and hasattr(taxonomy, "name"):
                    target_name = taxonomy.name
        
        # Build result
        result = {
            "airbench": {
                "id": "airbench",
                "name": airbench_name,
                "total_risks": len(airbench_risks),
                "risks_mapped": len(airbench_risks_mapped),
                "coverage_percentage": round((len(airbench_risks_mapped) / len(airbench_risks)) * 100, 1),
                "tier_counts": tier_counts,
                "tier_mapped": tier_stats,
                "tier_coverage": tier_coverage
            },
            "target": {
                "id": taxonomy_id,
                "name": target_name,
                "total_risks": len(target_risks),
                "risks_mapped": len(target_risks_mapped),
                "coverage_percentage": round((len(target_risks_mapped) / len(target_risks)) * 100, 1)
            },
            "relationships": relationship_counts,
            "total_mappings": sum(relationship_counts.values())
        }
        
        return result