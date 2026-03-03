"""Knowledge Graph construction pipeline."""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from config import ExtractionConfig, VerificationConfig
from extractor import KGExtractor
from verifier import TripletVerifier
from schemas import DynamicKnowledgeGraphs, Graph


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log', encoding='utf-8')
    ]
)


@dataclass
class PipelineConfig:
    """Configuration for the KG construction pipeline."""
    max_time: int = 3
    extract_llm: str = "mistralai/Mistral-7B-Instruct-v0.2"
    verification_llm1: str = "mistralai/Mistral-7B-Instruct-v0.2"
    verification_llm2: str = "mistralai/Mistral-7B-Instruct-v0.2"
    output_dir: str = "output"
    skip_verification: bool = True
    num_triplets: Optional[int] = None
    extract_temperature: float = 0.0
    verify_temperature: float = 0.0


class KGCPipeline:
    """Main pipeline for verified knowledge graph construction."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline."""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._extractor = None
        self._verifier = None

    
    @property
    def extractor(self) -> KGExtractor:
        """Lazy-load the extractor."""
        if self._extractor is None:
            self._extractor = KGExtractor(
                self.config.extract_llm,
                num_triplets=self.config.num_triplets,
                temperature=self.config.extract_temperature
            )
        return self._extractor
    
    @property
    def verifier(self) -> TripletVerifier:
        """Lazy-load the verifier."""
        if self._verifier is None:
            self._verifier = TripletVerifier(
                self.config.verification_llm1,
                self.config.verification_llm2,
                temperature=self.config.verify_temperature
            )
        return self._verifier
    
    def read_csv(self, input_path: str) -> List[Dict[str, Any]]:
        """Read input CSV file."""
        rows = []
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows
    
    def get_paragraphs_from_row(self, row: Dict[str, Any]) -> List[str]:
        """Extract paragraph columns from a row."""
        paragraphs = []
        i = 1
        while True:
            key = f"paragraph_{i}"
            if key in row and row[key]:
                paragraphs.append(row[key].strip())
                i += 1
            else:
                break
        return paragraphs
    
    def process_single_row(
        self, 
        row_id: str, 
        paragraphs: List[str]
    ) -> Tuple[DynamicKnowledgeGraphs, bool, int]:
        """Process single row: extract and verify graphs."""
        if self.config.skip_verification:
            logging.info(f"Row {row_id}: skipping verification")
            kgs = self.extractor.extract(paragraphs, None, None)
            return kgs, True, 0
        
        best_kgs = None
        best_rejection_count = float('inf')
        failed_triplets = None
        approved_triplets = None
        
        for iteration in range(1, self.config.max_time + 1):
            kgs = self.extractor.extract(paragraphs, failed_triplets, approved_triplets)
            verified_kgs, results = self.verifier.verify_all_graphs(paragraphs, kgs)
            
            rejection_summary = self.verifier.get_rejection_summary(results)
            total_rejections = sum(rejection_summary.values())
            
            if total_rejections < best_rejection_count:
                best_rejection_count = total_rejections
                best_kgs = verified_kgs
            
            if total_rejections == 0:
                break
            
            failed_triplets = self.verifier.get_failed_triplets(results)
            approved_triplets = self.verifier.get_verified_triplets(results)
        
        if best_rejection_count > 0:
            logging.warning(
                f"Row {row_id}: {best_rejection_count} unverified triplets after {self.config.max_time} iterations"
            )
        
        all_verified = best_rejection_count == 0
        return best_kgs, all_verified, best_rejection_count
    
    def format_kg_for_output(self, graph: Graph) -> str:
        """Format a graph to JSON string for CSV output."""
        triplets = [[t.subject, t.relation, t.object] for t in graph.triples]
        return json.dumps(triplets, ensure_ascii=False)
    
    def get_output_path(self, input_path: str) -> Path:
        """Generate output file path from input file."""
        input_name = Path(input_path).stem
        return self.output_dir / f"{input_name}_KGs.csv"
    
    def get_fieldnames(self, row: Dict[str, Any]) -> List[str]:
        """Extract output CSV fieldnames from row paragraphs."""
        para_columns = []
        i = 1
        while f"paragraph_{i}" in row:
            para_columns.append(f"paragraph_{i}")
            i += 1
        
        fieldnames = ['id']
        for i in range(1, len(para_columns) + 1):
            fieldnames.append(f"paragraph_{i}")
            fieldnames.append(f"kg_{i}")
        return fieldnames
    
    def load_processed_ids(self, output_path: Path) -> set:
        """Load IDs of rows that have already been processed."""
        processed_ids = set()
        if not output_path.exists():
            return processed_ids
        
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check if any kg column has data
                has_kg = False
                for key in row:
                    if key.startswith('kg_') and row[key] and row[key] != '[]':
                        has_kg = True
                        break
                if has_kg:
                    processed_ids.add(row.get('id', ''))
        
        return processed_ids
    
    def initialize_output_file(self, output_path: Path, fieldnames: List[str]):
        """Create output CSV file with headers if it doesn't exist."""
        if not output_path.exists():
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
    def append_row_to_output(
        self,
        output_path: Path,
        row: Dict[str, Any],
        kgs: DynamicKnowledgeGraphs,
        fieldnames: List[str]
    ):
        """Write processed row with knowledge graphs to output CSV."""
        output_row = {'id': row.get('id', '')}
        
        # Copy paragraphs and add KGs
        i = 1
        while f"paragraph_{i}" in row:
            output_row[f"paragraph_{i}"] = row.get(f"paragraph_{i}", '')
            kg_col = f"kg_{i}"
            graph = kgs.get_graph(i)
            output_row[kg_col] = self.format_kg_for_output(graph)
            i += 1
        
        # Append to file
        with open(output_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(output_row)
    def run(self, input_path: str) -> str:
        """Process input CSV and generate knowledge graphs."""
        rows = self.read_csv(input_path)
        
        if not rows:
            logging.warning("No rows found in input file")
            return None
        
        output_path = self.get_output_path(input_path)
        fieldnames = self.get_fieldnames(rows[0])
        processed_ids = self.load_processed_ids(output_path)
        self.initialize_output_file(output_path, fieldnames)
        
        processed_count = 0
        skipped_count = 0
        for row in rows:
            row_id = row.get('id', 'unknown')
            
            if row_id in processed_ids:
                skipped_count += 1
                continue
            
            paragraphs = self.get_paragraphs_from_row(row)
            
            if not paragraphs:
                empty_kgs = DynamicKnowledgeGraphs(graphs={})
                self.append_row_to_output(output_path, row, empty_kgs, fieldnames)
                continue
            
            kgs, all_verified, rejection_count = self.process_single_row(row_id, paragraphs)
            self.append_row_to_output(output_path, row, kgs, fieldnames)
            processed_count += 1
        
        self.cleanup()
        return str(output_path)
    
    def cleanup(self):
        """Release resources."""
        if self._extractor is not None:
            self._extractor.clear()
        if self._verifier is not None:
            self._verifier.clear()



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Knowledge Graph construction pipeline"
    )
    
    parser.add_argument(
        "input_csv",
        nargs='?',
        default=None,
        help="Path to input CSV file. If not provided, processes all CSVs in input/ folder."
    )
    
    parser.add_argument(
        "--max_time",
        type=int,
        default=3,
        help="Maximum iterations for extraction/verification loop (default: 3)"
    )
    
    parser.add_argument(
        "--extract_llm",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="LLM for extraction (default: mistralai/Mistral-7B-Instruct-v0.2)"
    )
    
    parser.add_argument(
        "--verification_llm1",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="First LLM for verification (default: mistralai/Mistral-7B-Instruct-v0.2)"
    )
    
    parser.add_argument(
        "--verification_llm2",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Second LLM for verification (default: mistralai/Mistral-7B-Instruct-v0.2)"
    )
    
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--skip_verification",
        action="store_false",
        default=True,
        help="Skip verification step"
    )
    
    parser.add_argument(
        "--num_triplets",
        type=int,
        default=None,
        help="Target number of triplets per graph (if not provided, LLM decides freely)"
    )
    
    parser.add_argument(
        "--extract_temperature",
        type=float,
        default=0.0,
        help="Temperature for extraction LLM (0.0-1.0, default: 0.0)"
    )
    
    parser.add_argument(
        "--verify_temperature",
        type=float,
        default=0.0,
        help="Temperature for verification LLM (0.0-1.0, default: 0.0)"
    )
    
    return parser.parse_args()


def get_all_csv_files(input_dir: str = "input") -> List[str]:
    """Retrieve all CSV files from input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        logging.error(f"Input directory not found: {input_dir}")
        return []
    
    csv_files = list(input_path.glob("*.csv"))
    return [str(f) for f in csv_files]


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.input_csv:
        if not os.path.exists(args.input_csv):
            logging.error(f"Input file not found: {args.input_csv}")
            sys.exit(1)
        input_files = [args.input_csv]
    else:
        input_files = get_all_csv_files("input")
        if not input_files:
            logging.error("No CSV files found in input/ folder")
            sys.exit(1)
    
    config = PipelineConfig(
        max_time=args.max_time,
        extract_llm=args.extract_llm,
        verification_llm1=args.verification_llm1,
        verification_llm2=args.verification_llm2,
        output_dir=args.output_dir,
        skip_verification=args.skip_verification,
        num_triplets=args.num_triplets,
        extract_temperature=args.extract_temperature,
        verify_temperature=args.verify_temperature
    )
    
    pipeline = KGCPipeline(config)
    
    try:
        output_paths = []
        for input_file in input_files:
            output_path = pipeline.run(input_file)
            if output_path:
                output_paths.append(output_path)
    except Exception as e:
        logging.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()