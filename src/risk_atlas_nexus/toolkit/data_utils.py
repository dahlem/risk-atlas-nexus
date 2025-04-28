import glob
import os
import yaml
from linkml_runtime.loaders import yaml_loader
from risk_atlas_nexus.data import get_data_path
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Container
from risk_atlas_nexus.toolkit.logging import configure_logger

logger = configure_logger(__name__)

def fix_string_values(value):
    """Fix string values that might be lists of characters, duplicate strings, or match fields."""
    match_fields = ["closeMatch", "exactMatch", "broadMatch", "narrowMatch", "relatedMatch"]
    
    if isinstance(value, list):
        # Check if it's a list of single characters, which indicates it should be a string
        if all(isinstance(item, str) and len(item) == 1 for item in value):
            return ''.join(value)
        # Check if it's a list of identical strings, which should be a single string
        elif len(value) > 1 and all(isinstance(item, str) for item in value) and len(set(value)) == 1:
            return value[0]
        # If it's a list of other items, recursively fix each item
        return [fix_string_values(item) for item in value]
    elif isinstance(value, dict):
        # Handle match fields in this dictionary if present
        for match_field in match_fields:
            if match_field in value and value[match_field] is not None and not isinstance(value[match_field], list):
                value[match_field] = [value[match_field]]
        
        # Special handling for risks and risk groups collections
        if "risks" in value:
            for risk in value["risks"]:
                for match_field in match_fields:
                    if match_field in risk and risk[match_field] is not None and not isinstance(risk[match_field], list):
                        risk[match_field] = [risk[match_field]]
        
        if "riskgroups" in value:
            for group in value["riskgroups"]:
                for match_field in match_fields:
                    if match_field in group and group[match_field] is not None and not isinstance(group[match_field], list):
                        group[match_field] = [group[match_field]]
                        
        # Recursively fix all values
        return {k: fix_string_values(v) for k, v in value.items()}
    else:
        # Return the value as is
        return value

def load_yamls_to_container(base_dir):
    """Function to load the RiskAtlasNexus with data

    Args:
        base_dir: str
            (Optional) user defined base directory path

    Returns:
        YAMLRoot instance of the Container class
    """

    # Get system yaml data path
    system_data_path = get_data_path()

    master_yaml_files = []
    for yaml_dir in [system_data_path, base_dir]:
        # Include YAML files from the user defined `base_dir` if exist.
        if yaml_dir is not None:
            master_yaml_files.extend(
                glob.glob(
                    os.path.join(yaml_dir, "**", "*.yaml"), recursive=True
                )
            )

    yml_items_result = {}
    for yaml_file in master_yaml_files:
        try:
            # Load directly with PyYAML to avoid string list issues
            with open(yaml_file, 'r') as f:
                yml_items = yaml.safe_load(f)
                
            # Fix any string values that might be lists of characters
            yml_items = fix_string_values(yml_items)
            
            for ontology_class, instances in yml_items.items():
                yml_items_result.setdefault(ontology_class, []).extend(
                    instances
                )
        except Exception as e:
            logger.info(f"YAML ignored: {yaml_file}. Failed to load. {e}")
    
    # Ensure match fields are always lists 
    match_fields = ["closeMatch", "exactMatch", "broadMatch", "narrowMatch", "relatedMatch"]
    
    # Ensure match fields in risks are always lists
    if "risks" in yml_items_result:
        for risk in yml_items_result["risks"]:
            for field in match_fields:
                if field in risk and risk[field] is not None and not isinstance(risk[field], list):
                    risk[field] = [risk[field]]
    
    # Ensure match fields in riskgroups are always lists
    if "riskgroups" in yml_items_result:
        for group in yml_items_result["riskgroups"]:
            for field in match_fields:
                if field in group and group[field] is not None and not isinstance(group[field], list):
                    group[field] = [group[field]]
    
    # combine any risk entries which share the same id, for example a risk, and a secondary entry for a mapping
    combine_risks = {}

    for risk in yml_items_result.get("risks", []):
        risk_id = risk["id"]

        if risk_id not in combine_risks:
            combine_risks[risk_id] = {"id": risk_id}

        for key, value in risk.items():
            if key != "id":
                if key not in combine_risks[risk_id]:
                    # If this is a match field, ensure it's a list
                    if key in match_fields and value is not None and not isinstance(value, list):
                        value = [value]
                    combine_risks[risk_id][key] = value
                else:
                    if combine_risks[risk_id][key] is not None:
                        # Make sure these are lists, not strings
                        if not isinstance(combine_risks[risk_id][key], list):
                            combine_risks[risk_id][key] = [combine_risks[risk_id][key]]
                        if not isinstance(value, list):
                            value = [value]
                        combine_risks[risk_id][key] = [
                            *combine_risks[risk_id][key],
                            *value,
                        ]
                    else:
                        # If this is a match field, ensure it's a list
                        if key in match_fields and value is not None and not isinstance(value, list):
                            value = [value]
                        combine_risks[risk_id][key] = value

    if "risks" in yml_items_result:
        yml_items_result["risks"] = list(combine_risks.values())

    # Fix any string values that might be lists of characters one more time
    yml_items_result = fix_string_values(yml_items_result)
    
    # Final check for match fields after all processing
    if "risks" in yml_items_result:
        for risk in yml_items_result["risks"]:
            for field in match_fields:
                if field in risk and risk[field] is not None and not isinstance(risk[field], list):
                    risk[field] = [risk[field]]
    
    if "riskgroups" in yml_items_result:
        for group in yml_items_result["riskgroups"]:
            for field in match_fields:
                if field in group and group[field] is not None and not isinstance(group[field], list):
                    group[field] = [group[field]]

    ontology = yaml_loader.load_any(
        source=yml_items_result,
        target_class=Container,
    )

    return ontology
