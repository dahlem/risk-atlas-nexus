# AIR-Bench Integration

## Overview

AIR-Bench (AI Risk Benchmark) is a regulation-aligned safety benchmark for responsible AI development. It features a four-tiered safety taxonomy with 314 risk categories derived from analyzing 8 government regulations and 16 company policies worldwide.

Risk Atlas Nexus integrates AIR-Bench as an additional risk taxonomy, providing mappings to other taxonomies like IBM AI Risk Atlas, allowing users to navigate between different regulatory and industry standards.

## AIR-Bench Taxonomy Structure

AIR-Bench uses a four-tiered taxonomy structure:

1. **Tier 1 (Top Level)**: Four main risk categories
   - System & Operational Risks
   - Content Safety Risks
   - Societal Risks
   - Legal & Rights Risks

2. **Tier 2**: Sub-categories of the main risk areas
   - For example, under "System & Operational Risks":
     - Model Performance
     - System Security
     - Infrastructure

3. **Tier 3**: More specific risk domains
   - For example, under "Model Performance":
     - Hallucination
     - Instruction Following
     - Output Quality

4. **Tier 4 (Individual Risks)**: 314 specific risk categories
   - For example, under "Hallucination":
     - Factual Error
     - Made-up Sources

## Using AIR-Bench in Risk Atlas Nexus

### Retrieving AIR-Bench Risks

You can access AIR-Bench risks through the Risk Atlas Nexus API:

```python
from risk_atlas_nexus.library import RiskAtlasNexus

# Initialize the library
ran = RiskAtlasNexus()

# Get all AIR-Bench risks
airbench_risks = ran.get_all_risks(taxonomy="airbench")

# Get a specific AIR-Bench risk
risk = ran.get_risk(id="airbench-factual-error")
```

### Exploring Related Risks Across Taxonomies

AIR-Bench risks are mapped to other taxonomies, allowing cross-taxonomy exploration:

```python
# Get a risk from AIR-Bench
risk = ran.get_risk(id="airbench-factual-error")

# Find related risks in other taxonomies
related_risks = ran.get_related_risks(risk=risk)

# Print related risks
for related_risk in related_risks:
    print(f"Related risk: {related_risk.name} ({related_risk.isDefinedByTaxonomy})")
```

### Generating New Mappings

You can generate new mappings between AIR-Bench and other taxonomies:

```python
from risk_atlas_nexus.metadata_base import MappingMethod
from risk_atlas_nexus.blocks.inference import create_inference_engine

# Get risks from two taxonomies
airbench_risks = ran.get_all_risks(taxonomy="airbench")
other_taxonomy_risks = ran.get_all_risks(taxonomy="other-taxonomy")

# Create an inference engine (for LLM-based mapping)
inference_engine = create_inference_engine()

# Generate mappings using semantic similarity
mappings = ran.generate_proposed_mappings(
    new_risks=airbench_risks,
    existing_risks=other_taxonomy_risks,
    inference_engine=inference_engine,
    new_prefix="airbench",
    mapping_method=MappingMethod.SEMANTIC
)
```

## AIR-Bench Benefits

Integration with AIR-Bench provides several benefits:

1. **Regulatory Alignment**: AIR-Bench is derived from actual government regulations and company policies, providing a practical framework for compliance.

2. **Comprehensive Coverage**: With 314 risk categories across four tiers, AIR-Bench offers granular coverage of potential AI risks.

3. **Cross-Taxonomy Navigation**: Mappings between AIR-Bench and other taxonomies enable users to navigate between different frameworks.

4. **Policy Insights**: AIR-Bench helps identify gaps between regulatory requirements and industry practices.

## References

- [AIR-Bench: A Regulation-Aligned Safety Benchmark for Responsible AI Development](https://arxiv.org/abs/2407.17436)