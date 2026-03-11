"""
Multi Knowledge Graph Construction Pipeline.

This pipeline:
1. Reads paragraphs from a CSV file
2. Extracts knowledge graphs for each paragraph
3. Verifies each triplet using two LLMs sequentially
4. Re-generates graphs if verification fails (up to max_time iterations)
5. Saves final verified KGs to output CSV

Usage:
    python main.py input.csv --max_time 3 --extract_llm gemini-2.5-flash 
                  --verification_llm1 mistralai/Mistral-7B-Instruct-v0.2 
                  --verification_llm2 mistralai/Mistral-7B-Instruct-v0.2
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ..utils.env_utils import load_environment

from .config import ExtractionConfig, VerificationConfig
from .extractor import KGExtractor
from .verifier import TripletVerifier
from .schemas import DynamicKnowledgeGraphs, Graph


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log', encoding='utf-8')
    ]
)

load_environment()


@dataclass
class PipelineConfig:
    """Configuration for the KG construction pipeline."""
    max_time: int = 3
    extract_llm: str = "mistralai/Mistral-7B-Instruct-v0.2"
    verification_llm1: str = "mistralai/Mistral-7B-Instruct-v0.2"
    verification_llm2: str = "mistralai/Mistral-7B-Instruct-v0.2"
    output_dir: str = "output"
    skip_verification: bool = True


class KGCPipeline:
    """Main pipeline for verified knowledge graph construction."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components (lazy loading)
        self._extractor = None
        self._verifier = None
        
        logging.info(f"[Pipeline] Initialized with config: {config}")
    
    @property
    def extractor(self) -> KGExtractor:
        """Lazy-load the extractor."""
        if self._extractor is None:
            self._extractor = KGExtractor(self.config.extract_llm)
        return self._extractor
    
    @property
    def verifier(self) -> TripletVerifier:
        """Lazy-load the verifier."""
        if self._verifier is None:
            self._verifier = TripletVerifier(
                self.config.verification_llm1,
                self.config.verification_llm2
            )
        return self._verifier
    
    def read_csv(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Read input CSV file.
        
        Expected format: id, paragraph_1, paragraph_2, paragraph_3, ...
        
        Args:
            input_path: Path to input CSV
            
        Returns:
            List of row dictionaries
        """
        rows = []
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        
        logging.info(f"[Pipeline] Read {len(rows)} rows from {input_path}")
        return rows
    
    def get_paragraphs_from_row(self, row: Dict[str, Any]) -> List[str]:
        """
        Extract paragraph columns from a row.
        
        Args:
            row: CSV row dictionary
            
        Returns:
            List of paragraphs in order
        """
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
        """
        Process a single row: extract and verify KGs with retry logic.
        
        Args:
            row_id: Identifier for the row
            paragraphs: List of paragraphs from the row
            
        Returns:
            Tuple of (final_kgs, all_verified, rejection_count)
            - final_kgs: Best knowledge graphs from all iterations
            - all_verified: True if all triplets verified, False otherwise
            - rejection_count: Number of unverified triplets in best result
        """
        logging.info(f"[Pipeline] Processing row {row_id} with {len(paragraphs)} paragraphs")
        
        # Skip verification if configured
        if self.config.skip_verification:
            logging.info(f"[Pipeline] Row {row_id} - Skipping verification (extraction only)")
            kgs = self.extractor.extract(paragraphs, None, None)
            return kgs, True, 0
        
        best_kgs = None
        best_rejection_count = float('inf')
        failed_triplets = None  # Track failed triplets for feedback to extractor
        approved_triplets = None  # Track approved triplets to keep in regeneration
        
        for iteration in range(1, self.config.max_time + 1):
            logging.info(f"[Pipeline] Row {row_id} - Iteration {iteration}/{self.config.max_time}")
            
            # Step 1: Extract knowledge graphs (pass failed and approved triplets from previous iteration)
            kgs = self.extractor.extract(paragraphs, failed_triplets, approved_triplets)
            
            # Step 2: Verify all graphs
            verified_kgs, results = self.verifier.verify_all_graphs(paragraphs, kgs)
            
            # Count rejections
            rejection_summary = self.verifier.get_rejection_summary(results)
            total_rejections = sum(rejection_summary.values())
            
            logging.info(f"[Pipeline] Row {row_id} - Iteration {iteration}: {total_rejections} triplets rejected")
            
            # Track best result
            if total_rejections < best_rejection_count:
                best_rejection_count = total_rejections
                best_kgs = verified_kgs
            
            # If all verified, we're done
            if total_rejections == 0:
                logging.info(f"[Pipeline] Row {row_id} - All triplets verified on iteration {iteration}")
                break
            
            # Get failed and approved triplets for next extraction attempt
            failed_triplets = self.verifier.get_failed_triplets(results)
            approved_triplets = self.verifier.get_verified_triplets(results)
            
            # If not last iteration, log that we're retrying
            if iteration < self.config.max_time:
                logging.info(f"[Pipeline] Row {row_id} - Retrying extraction due to rejected triplets")
        
        if best_rejection_count > 0:
            logging.warning(
                f"[Pipeline] Row {row_id} - Completed with {best_rejection_count} unverified triplets "
                f"after {self.config.max_time} iterations"
            )
        
        all_verified = best_rejection_count == 0
        return best_kgs, all_verified, best_rejection_count
    
    def format_kg_for_output(self, graph: Graph) -> str:
        """
        Format a graph for CSV output.
        
        Args:
            graph: Graph to format
            
        Returns:
            JSON string of triplets
        """
        triplets = [[t.subject, t.relation, t.object] for t in graph.triples]
        return json.dumps(triplets, ensure_ascii=False)
    
    def get_output_path(self, input_path: str, explicit_output_path: Optional[str] = None) -> Path:
        """Get the output file path for a given input file."""
        if explicit_output_path:
            return Path(explicit_output_path)

        input_name = Path(input_path).stem
        return self.output_dir / f"{input_name}_KGs.csv"
    
    def get_fieldnames(self, row: Dict[str, Any]) -> List[str]:
        """Get fieldnames for the output CSV based on paragraph columns."""
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
        """
        Load IDs of rows that have already been processed.
        
        Args:
            output_path: Path to existing output file
            
        Returns:
            Set of row IDs that have graphs generated
        """
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
        """
        Initialize the output CSV file with headers if it doesn't exist.
        
        Args:
            output_path: Path to output file
            fieldnames: Column names for the CSV
        """
        if not output_path.exists():
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            logging.info(f"[Pipeline] Created output file: {output_path}")
    
    def append_row_to_output(
        self,
        output_path: Path,
        row: Dict[str, Any],
        kgs: DynamicKnowledgeGraphs,
        fieldnames: List[str]
    ):
        """
        Append a single processed row to the output CSV.
        
        Args:
            output_path: Path to output file
            row: Original row data
            kgs: Knowledge graphs for the row
            fieldnames: Column names for the CSV
        """
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
        
        logging.info(f"[Pipeline] Saved row {row.get('id', 'unknown')} to {output_path}")
    
    def run(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Run the full pipeline on an input CSV.
        Saves output row by row and resumes from where it left off.
        
        Args:
            input_path: Path to input CSV file
            output_path: Optional explicit output CSV path
            
        Returns:
            Path to output CSV file
        """
        logging.info(f"[Pipeline] Starting pipeline for {input_path}")
        
        # Read input
        rows = self.read_csv(input_path)
        
        if not rows:
            logging.warning("[Pipeline] No rows found in input file")
            return None
        
        # Get output path and fieldnames
        output_path = self.get_output_path(input_path, output_path)
        fieldnames = self.get_fieldnames(rows[0])
        
        # Load already processed row IDs
        processed_ids = self.load_processed_ids(output_path)
        if processed_ids:
            logging.info(f"[Pipeline] Found {len(processed_ids)} already processed rows, resuming...")
        
        # Initialize output file if needed
        self.initialize_output_file(output_path, fieldnames)
        
        # Process each row
        processed_count = 0
        skipped_count = 0
        for row in rows:
            row_id = row.get('id', 'unknown')
            
            # Skip already processed rows
            if row_id in processed_ids:
                logging.info(f"[Pipeline] Row {row_id} already processed, skipping")
                skipped_count += 1
                continue
            
            paragraphs = self.get_paragraphs_from_row(row)
            
            if not paragraphs:
                logging.warning(f"[Pipeline] Row {row_id} has no paragraphs, skipping")
                # Save empty KGs for this row
                empty_kgs = DynamicKnowledgeGraphs(graphs={})
                self.append_row_to_output(output_path, row, empty_kgs, fieldnames)
                continue
            
            kgs, all_verified, rejection_count = self.process_single_row(row_id, paragraphs)
            
            # Save this row immediately
            self.append_row_to_output(output_path, row, kgs, fieldnames)
            processed_count += 1
        
        # Cleanup
        self.cleanup()
        
        logging.info(f"[Pipeline] Pipeline completed. Processed: {processed_count}, Skipped: {skipped_count}")
        return str(output_path)
    
    def cleanup(self):
        """Clean up resources."""
        if self._extractor is not None:
            self._extractor.clear()
        if self._verifier is not None:
            self._verifier.clear()
        logging.info("[Pipeline] Resources cleaned up")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi Knowledge Graph Construction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py data.csv
    python main.py data.csv --max_time 5
    python main.py data.csv --extract_llm gemini-2.5-flash --verification_llm1 gpt-4 --verification_llm2 gpt-4
    python main.py  # Process all CSVs in input/ folder
        """
    )
    
    parser.add_argument(
        "input_csv",
        nargs='?',
        default=None,
        help="Path to input CSV file (format: id, paragraph_1, paragraph_2, ...). If not provided, processes all CSVs in input/ folder."
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
        "--run_verification",
        action="store_true",
        default=False,
        help="Run verification step (verification is skipped by default)"
    )
    
    # Alternative flag name for backwards compatibility
    parser.add_argument(
        "--skip_verification",
        action="store_false",
        dest="run_verification",
        help="Legacy flag: skip verification (use --run_verification to enable verification)"
    )
    
    return parser.parse_args()


def get_all_csv_files(input_dir: str = "input") -> List[str]:
    """Get all CSV files from the input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        logging.error(f"Input directory not found: {input_dir}")
        return []
    
    csv_files = list(input_path.glob("*.csv"))
    return [str(f) for f in csv_files]



def construct_kgs(
    input_csv: str,
    output_csv: str,
    max_time: int = 1,
    extract_llm: str = "llama-3.3-70b-versatile",
    verification_llm1: Optional[str] = None,
    verification_llm2: Optional[str] = None,
    skip_verification: bool = True,
    verbose: bool = False,
) -> str:
    """
    Run the KG pipeline programmatically without argparse.

    Expected input format: `id`, `paragraph_1`, `paragraph_2`, ...

    Args:
        input_csv: Input CSV file path.
        output_csv: Output CSV file path.
        max_time: Maximum extraction/verification iterations.
        extract_llm: Extraction model name.
        verification_llm1: First verification model; defaults to `extract_llm`.
        verification_llm2: Second verification model; defaults to `extract_llm`.
        skip_verification: Skip verification by default.
        verbose: Enable debug logging.

    Returns:
        Output CSV path.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    output_parent = Path(output_csv).parent
    if str(output_parent) and str(output_parent) != ".":
        output_parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = PipelineConfig(
        max_time=max_time,
        extract_llm=extract_llm,
        verification_llm1=verification_llm1 or extract_llm,
        verification_llm2=verification_llm2 or extract_llm,
        output_dir=str(output_parent) if str(output_parent) else ".",
        skip_verification=skip_verification,
    )

    pipeline = KGCPipeline(config)
    return pipeline.run(input_csv, output_csv)


def construct_graph(paragraph: str, llm: str = "llama-3.3-70b-versatile") -> Graph:
    """
    Construct a knowledge graph for a single paragraph.
    
    Args:
        paragraph: The paragraph to extract knowledge graph from.
        llm: LLM model to use for extraction (default: llama-3.3-70b-versatile).
    
    Returns:
        Knowledge graph for the paragraph.
    """
    extractor = KGExtractor(llm)
    kgs = extractor.extract([paragraph], None, None)
    return kgs.get_graph(1)


def construct_graphs(paragraphs: List[str], llm: str = "llama-3.3-70b-versatile") -> DynamicKnowledgeGraphs:
    """
    Construct knowledge graphs for multiple paragraphs.
    
    Args:
        paragraphs: List of paragraphs to extract knowledge graphs from.
        llm: LLM model to use for extraction (default: llama-3.3-70b-versatile).
    
    Returns:
        DynamicKnowledgeGraphs containing graphs for each paragraph.
    """
    if not paragraphs:
        return DynamicKnowledgeGraphs(graphs={})
    
    extractor = KGExtractor(llm)
    kgs = extractor.extract(paragraphs, None, None)
    return kgs


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine input files to process
    if args.input_csv:
        # Single file specified
        if not os.path.exists(args.input_csv):
            logging.error(f"Input file not found: {args.input_csv}")
            sys.exit(1)
        input_files = [args.input_csv]
    else:
        # No file specified - process all CSVs in input folder
        input_files = get_all_csv_files("input")
        if not input_files:
            logging.error("No CSV files found in input/ folder")
            sys.exit(1)
        logging.info(f"Found {len(input_files)} CSV files in input/ folder")
    
    # Create config
    config = PipelineConfig(
        max_time=args.max_time,
        extract_llm=args.extract_llm,
        verification_llm1=args.verification_llm1,
        verification_llm2=args.verification_llm2,
        output_dir=args.output_dir,
        skip_verification=not args.run_verification  # Invert: skip when run_verification is False
    )
    
    # Run pipeline
    pipeline = KGCPipeline(config)
    
    try:
        output_paths = []
        for input_file in input_files:
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing: {input_file}")
            logging.info(f"{'='*50}")
            output_path = pipeline.run(input_file)
            if output_path:
                output_paths.append(output_path)
        
        print(f"\nPipeline completed! Processed {len(output_paths)} file(s).")
        for path in output_paths:
            print(f"  - {path}")
    except Exception as e:
        logging.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # construct_kgs(
    #     input_csv="input/test.csv",
    #     output_csv="output/sample_output.csv",
    #     verbose=True
    # )
    
    print(construct_graphs(
        paragraphs=[
            "Barack Obama was born in Hawaii. He was elected president in 2008.",
            "The Eiffel Tower is located in Paris. It was completed in 1889."
        ],
        llm="llama-3.3-70b-versatile"
    ))