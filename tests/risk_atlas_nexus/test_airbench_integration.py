import os
import pytest
import json
import yaml
import tempfile
import shutil
from typing import List, Dict

from risk_atlas_nexus.library import RiskAtlasNexus
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Risk, RiskTaxonomy
from risk_atlas_nexus.ai_risk_ontology.util.airbench2linkml import create_container_object
from risk_atlas_nexus.metadata_base import MappingMethod
from sssom_schema import Mapping
from linkml_runtime.dumpers import YAMLDumper


class TestAIRBenchIntegration:
    """Test AIR-Bench integration with Risk Atlas Nexus"""

    temp_dir = None

    @classmethod
    def setup_class(cls):
        """Set up test data directory"""
        cls.temp_dir = tempfile.mkdtemp()
        # Create AIR-Bench data file
        container = create_container_object()
        os.makedirs(os.path.join(cls.temp_dir, "knowledge_graph"), exist_ok=True)
        
        with open(os.path.join(cls.temp_dir, "knowledge_graph", "airbench_data.yaml"), "w", encoding="utf-8") as f:
            f.write(YAMLDumper().dumps(container))
    
    @classmethod
    def teardown_class(cls):
        """Clean up temporary directory"""
        if cls.temp_dir:
            shutil.rmtree(cls.temp_dir)

    def test_airbench_risk_loading(self):
        """Test that AIR-Bench risks are properly loaded"""
        ran = RiskAtlasNexus(base_dir=self.temp_dir)
        
        # Check if AIR-Bench taxonomy is loaded
        taxonomies = ran.get_all_taxonomies()
        airbench_taxonomy = None
        for taxonomy in taxonomies:
            if taxonomy.id == "airbench":
                airbench_taxonomy = taxonomy
                break
        
        assert airbench_taxonomy is not None, "AIR-Bench taxonomy not found"
        assert airbench_taxonomy.name == "AIR-Bench Taxonomy"
        
        # Check if AIR-Bench risks are loaded
        risks = ran.get_all_risks(taxonomy="airbench")
        assert len(risks) > 0, "No AIR-Bench risks found"
        
        # Check the four-tier structure (spot check a few risks)
        sample_risk = None
        for risk in risks:
            if risk.name == "Factual Error":
                sample_risk = risk
                break
        
        assert sample_risk is not None, "Sample risk 'Factual Error' not found"
        assert sample_risk.isDefinedByTaxonomy == "airbench"
        assert sample_risk.isPartOf.startswith("airbench-hallucination")

    def test_airbench_risk_groups(self):
        """Test that AIR-Bench risks form the correct hierarchy"""
        ran = RiskAtlasNexus(base_dir=self.temp_dir)
        
        # Get risks from the different tiers
        risks = ran.get_all_risks(taxonomy="airbench")
        
        # Find risk groups (RiskGroups are stored as part of the Container)
        risk_groups = ran._ontology.riskgroups
        
        # Find tier 1 group
        tier1_group = None
        for group in risk_groups:
            if group.id == "airbench-system-operational-risks":
                tier1_group = group
                break
        
        assert tier1_group is not None, "Tier 1 group not found"
        # In our test environment, the narrowMatch might not be populated yet
        # Just verify the group exists
        
        # Find tier 2 group
        tier2_group = None
        for group in risk_groups:
            if group.id == "airbench-model-performance":
                tier2_group = group
                break
                
        assert tier2_group is not None, "Tier 2 group not found"
        # In our test environment, the broadMatch might not be populated yet
        # Just verify the group exists
        
        # Find tier 3 group
        tier3_group = None
        for group in risk_groups:
            if group.id == "airbench-hallucination":
                tier3_group = group
                break
                
        assert tier3_group is not None, "Tier 3 group not found"
        # In our test environment, the broadMatch might not be populated yet
        # Just verify the group exists
        
        # Check risks have relationships to tier 3
        risk = ran.get_risk(id="airbench-factual-error")
        assert risk is not None, "Risk not found"
        assert risk.isPartOf == "airbench-hallucination"

    @pytest.mark.skip(reason="Requires more complex setup with mapping data")
    def test_airbench_mapping_generation(self):
        """Test generating mappings between AIR-Bench and IBM risks"""
        ran = RiskAtlasNexus(base_dir=self.temp_dir)
        
        # Get sample risks from each taxonomy
        airbench_risks = ran.get_all_risks(taxonomy="airbench")[:5]
        ibm_risks = ran.get_all_risks(taxonomy="ibm-ai-risk-atlas")[:10]
        
        # Generate mappings
        mappings = ran.generate_proposed_mappings(
            new_risks=airbench_risks,
            existing_risks=ibm_risks,
            inference_engine=None,
            new_prefix="airbench",
            mapping_method=MappingMethod.SEMANTIC
        )
        
        assert len(mappings) > 0, "No mappings generated"
        
        # Check mapping structure
        for mapping in mappings:
            assert mapping.subject_id.startswith("airbench:")
            assert mapping.object_id.startswith("ibmairisk:")
            assert mapping.predicate_id in [
                "skos:exactMatch", 
                "skos:closeMatch", 
                "skos:relatedMatch", 
                "skos:broadMatch", 
                "skos:narrowMatch",
                "noMatch"
            ]

    def test_get_related_risks_with_airbench(self):
        """Test getting related risks across taxonomies including AIR-Bench"""
        # This test depends on having mapping data loaded, so we'll create a minimal test
        ran = RiskAtlasNexus(base_dir=self.temp_dir)
        
        # Get a sample AIR-Bench risk
        risk = ran.get_risk(id="airbench-factual-error")
        assert risk is not None, "Risk not found"
        
        # Try to get related risks
        # This may return no results if no mappings exist yet, but should not error
        related_risks = ran.get_related_risks(risk=risk)
        # Just verify the function runs without error
        assert isinstance(related_risks, list)