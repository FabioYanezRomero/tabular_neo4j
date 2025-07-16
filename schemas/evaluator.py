import os
import yaml
import json
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CompletenessResults:
    """Data class to store completeness evaluation results"""
    node_completeness: float
    property_completeness: float
    relation_completeness: float
    detailed_analysis: Dict[str, Any]

class DatasetSynonymConfig:
    """Configuration class for dataset-specific synonyms"""
    def __init__(self, node_synonyms: Dict[str, List[str]], relation_synonyms: Dict[str, List[str]]):
        self.node_synonyms = node_synonyms
        self.relation_synonyms = relation_synonyms

class GraphSchemaCompletenessEvaluator:
    """Evaluates completeness of generated graph schema against golden reference with synonym support"""
    
    def __init__(self, synonyms: DatasetSynonymConfig):
        self.synonyms = synonyms
        self.results = None
    
    def normalize_name(self, name: str) -> str:
        """Normalize names for comparison"""
        return name.lower().replace('_', '').replace('-', '')
    
    def map_name_with_synonyms(self, name: str, element_type: str) -> str:
        """Map name to canonical form using synonyms"""
        name_norm = self.normalize_name(name)
        synonyms = self.synonyms.node_synonyms if element_type == 'node' else self.synonyms.relation_synonyms
        
        for canonical, syn_list in synonyms.items():
            normalized_synonyms = [self.normalize_name(syn) for syn in syn_list]
            if name_norm == canonical or name_norm in normalized_synonyms:
                return canonical
        
        return name_norm
    
    def extract_node_info(self, schema: Dict) -> Dict[str, Set[str]]:
        """Extract node types and their attributes from schema"""
        node_info = {}
        for node in schema.get('nodes', []):
            node_type = self.map_name_with_synonyms(node['type'], 'node')
            attributes = {self.normalize_name(attr) for attr in node.get('attributes', [])}
            
            if node_type not in node_info:
                node_info[node_type] = set()
            node_info[node_type].update(attributes)
        
        return node_info
    
    def extract_edge_info(self, schema: Dict) -> Set[Tuple[str, str, str]]:
        """Extract edge information as (type, source, target) tuples"""
        edge_info = set()
        for edge in schema.get('edges', []):
            edge_type = self.map_name_with_synonyms(edge['type'], 'relation')
            source = self.map_name_with_synonyms(edge['source'], 'node')
            target = self.map_name_with_synonyms(edge['target'], 'node')
            edge_info.add((edge_type, source, target))
        
        return edge_info
    
    def calculate_node_completeness(self, gen_nodes: Dict[str, Set[str]], 
                                  gold_nodes: Dict[str, Set[str]]) -> Tuple[float, Dict]:
        """Calculate node type completeness"""
        gen_node_types = set(gen_nodes.keys())
        gold_node_types = set(gold_nodes.keys())
        
        found_nodes = gen_node_types.intersection(gold_node_types)
        missing_nodes = gold_node_types - gen_node_types
        extra_nodes = gen_node_types - gold_node_types
        
        completeness = len(found_nodes) / len(gold_node_types) if gold_node_types else 1.0
        
        details = {
            'found_nodes': sorted(list(found_nodes)),
            'missing_nodes': sorted(list(missing_nodes)),
            'extra_nodes': sorted(list(extra_nodes)),
            'total_golden_nodes': len(gold_node_types),
            'total_generated_nodes': len(gen_node_types),
            'completeness_score': completeness
        }
        
        return completeness, details
    
    def calculate_property_completeness(self, gen_nodes: Dict[str, Set[str]], 
                                      gold_nodes: Dict[str, Set[str]]) -> Tuple[float, Dict]:
        """Calculate property completeness per node and overall"""
        node_property_analysis = {}
        total_completeness = 0
        evaluated_nodes = 0
        
        for node_type, gold_attrs in gold_nodes.items():
            gen_attrs = gen_nodes.get(node_type, set())
            
            found_attrs = gen_attrs.intersection(gold_attrs)
            missing_attrs = gold_attrs - gen_attrs
            extra_attrs = gen_attrs - gold_attrs
            
            if gold_attrs:
                node_completeness = len(found_attrs) / len(gold_attrs)
            else:
                node_completeness = 1.0
            
            node_property_analysis[node_type] = {
                'completeness': node_completeness,
                'found_attributes': sorted(list(found_attrs)),
                'missing_attributes': sorted(list(missing_attrs)),
                'extra_attributes': sorted(list(extra_attrs)),
                'total_golden_attributes': len(gold_attrs),
                'total_generated_attributes': len(gen_attrs)
            }
            
            total_completeness += node_completeness
            evaluated_nodes += 1
        
        overall_completeness = total_completeness / evaluated_nodes if evaluated_nodes > 0 else 1.0
        
        return overall_completeness, node_property_analysis
    
    def calculate_relation_completeness(self, gen_edges: Set[Tuple[str, str, str]], 
                                      gold_edges: Set[Tuple[str, str, str]]) -> Tuple[float, Dict]:
        """Calculate relation completeness"""
        found_edges = gen_edges.intersection(gold_edges)
        missing_edges = gold_edges - gen_edges
        extra_edges = gen_edges - gold_edges
        
        completeness = len(found_edges) / len(gold_edges) if gold_edges else 1.0
        
        details = {
            'found_relations': sorted([f"{edge[0]}: {edge[1]} -> {edge[2]}" for edge in found_edges]),
            'missing_relations': sorted([f"{edge[0]}: {edge[1]} -> {edge[2]}" for edge in missing_edges]),
            'extra_relations': sorted([f"{edge[0]}: {edge[1]} -> {edge[2]}" for edge in extra_edges]),
            'total_golden_relations': len(gold_edges),
            'total_generated_relations': len(gen_edges),
            'completeness_score': completeness
        }
        
        return completeness, details
    
    def evaluate_completeness(self, generated_schema: Dict, golden_schema: Dict) -> CompletenessResults:
        """Main evaluation function"""
        # Extract schema components
        gen_nodes = self.extract_node_info(generated_schema)
        gold_nodes = self.extract_node_info(golden_schema)
        gen_edges = self.extract_edge_info(generated_schema)
        gold_edges = self.extract_edge_info(golden_schema)
        
        # Calculate completeness metrics
        node_completeness, node_details = self.calculate_node_completeness(gen_nodes, gold_nodes)
        property_completeness, property_details = self.calculate_property_completeness(gen_nodes, gold_nodes)
        relation_completeness, relation_details = self.calculate_relation_completeness(gen_edges, gold_edges)
        
        detailed_analysis = {
            'nodes': node_details,
            'properties': property_details,
            'relations': relation_details,
            'evaluation_timestamp': datetime.now().isoformat(),
            'synonym_mappings_applied': {
                'node_synonyms': self.synonyms.node_synonyms,
                'relation_synonyms': self.synonyms.relation_synonyms
            }
        }
        
        self.results = CompletenessResults(
            node_completeness=node_completeness,
            property_completeness=property_completeness,
            relation_completeness=relation_completeness,
            detailed_analysis=detailed_analysis
        )
        
        return self.results
    
    def save_results(self, output_folder: str, output_name: str) -> str:
        """Save evaluation results to JSON file"""
        if not self.results:
            raise ValueError("No results to save. Run evaluate_completeness() first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Create output file path
        output_path = os.path.join(output_folder, f"{output_name}_completeness_report.json")
        
        # Prepare report data
        report_data = {
            'summary': {
                'node_completeness': self.results.node_completeness,
                'property_completeness': self.results.property_completeness,
                'relation_completeness': self.results.relation_completeness,
                'overall_score': (self.results.node_completeness + 
                                self.results.property_completeness + 
                                self.results.relation_completeness) / 3
            },
            'detailed_analysis': self.results.detailed_analysis
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return output_path
    
    def print_summary(self):
        """Print a formatted summary of the evaluation"""
        if not self.results:
            print("No evaluation results available.")
            return
        
        print("=" * 60)
        print("GRAPH SCHEMA COMPLETENESS EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Node Completeness:     {self.results.node_completeness:.2%}")
        print(f"Property Completeness: {self.results.property_completeness:.2%}")
        print(f"Relation Completeness: {self.results.relation_completeness:.2%}")
        
        overall_score = (self.results.node_completeness + 
                        self.results.property_completeness + 
                        self.results.relation_completeness) / 3
        print(f"Overall Score:         {overall_score:.2%}")
        print("=" * 60)

def load_schema(file_path: str) -> Dict:
    """Load schema from YAML or JSON file"""
    with open(file_path, 'r') as file:
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            return yaml.safe_load(file)
        else:
            return json.load(file)

def load_synonyms_from_yaml(synonyms_path: str) -> DatasetSynonymConfig:
    """Load synonyms configuration from YAML file"""
    with open(synonyms_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return DatasetSynonymConfig(
        node_synonyms=config.get('node_synonyms', {}),
        relation_synonyms=config.get('relation_synonyms', {})
    )

def evaluate_and_save(golden_schema_path: str, 
                      generated_schema_path: str, 
                      node_synonyms: Dict[str, List[str]], 
                      relation_synonyms: Dict[str, List[str]],
                      output_folder: str, 
                      output_name: str,
                      print_summary: bool = True) -> str:
    """
    Main function to evaluate schemas and save results
    
    Args:
        golden_schema_path: Path to golden schema file
        generated_schema_path: Path to generated schema file
        node_synonyms: Dictionary of node synonyms
        relation_synonyms: Dictionary of relation synonyms
        output_folder: Output directory for results
        output_name: Base name for output files
        print_summary: Whether to print summary to console
    
    Returns:
        Path to the saved report file
    """
    # Load schemas
    golden_schema = load_schema(golden_schema_path)
    generated_schema = load_schema(generated_schema_path)
    
    # Create synonym configuration
    synonyms = DatasetSynonymConfig(node_synonyms, relation_synonyms)
    
    # Evaluate
    evaluator = GraphSchemaCompletenessEvaluator(synonyms)
    evaluator.evaluate_completeness(generated_schema, golden_schema)
    
    # Print summary if requested
    if print_summary:
        evaluator.print_summary()
    
    # Save results
    output_path = evaluator.save_results(output_folder, output_name)
    
    return output_path

def evaluate_with_synonym_file(golden_schema_path: str,
                               generated_schema_path: str,
                               synonyms_config_path: str,
                               output_folder: str,
                               output_name: str,
                               print_summary: bool = True) -> str:
    """
    Evaluate schemas using synonym configuration from file
    
    Args:
        golden_schema_path: Path to golden schema file
        generated_schema_path: Path to generated schema file
        synonyms_config_path: Path to synonyms YAML configuration file
        output_folder: Output directory for results
        output_name: Base name for output files
        print_summary: Whether to print summary to console
    
    Returns:
        Path to the saved report file
    """
    # Load schemas and synonyms
    golden_schema = load_schema(golden_schema_path)
    generated_schema = load_schema(generated_schema_path)
    synonyms = load_synonyms_from_yaml(synonyms_config_path)
    
    # Evaluate
    evaluator = GraphSchemaCompletenessEvaluator(synonyms)
    evaluator.evaluate_completeness(generated_schema, golden_schema)
    
    # Print summary if requested
    if print_summary:
        evaluator.print_summary()
    
    # Save results
    output_path = evaluator.save_results(output_folder, output_name)
    
    return output_path

if __name__ == "__main__":
    # Example usage
    print("Graph Schema Evaluation Script")
    print("Use the functions evaluate_and_save() or evaluate_with_synonym_file() to run evaluations")
