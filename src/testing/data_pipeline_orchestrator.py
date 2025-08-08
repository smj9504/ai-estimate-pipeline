# src/testing/data_pipeline_orchestrator.py
"""
Data Pipeline Orchestrator for AI Estimation Testing
Implements modern data engineering patterns for test data lifecycle management
"""
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class ProcessingMode(Enum):
    """Data processing execution modes"""
    STREAMING = "streaming"      # Real-time single record processing
    MICRO_BATCH = "micro_batch"  # Small batch processing (5-10 records)
    BATCH = "batch"              # Large batch processing (50+ records)


class DataTier(Enum):
    """Data quality and access tiers"""
    RAW = "raw"                  # Unprocessed input data
    PROCESSED = "processed"      # Phase output data with metadata
    ANALYTICS = "analytics"      # Aggregated metrics and reports
    ARCHIVE = "archive"          # Historical data for compliance


@dataclass
class DataLineage:
    """Tracks data provenance and dependencies"""
    source_hash: str                    # Input data content hash
    phase_dependency: Optional[str]     # Previous phase result hash
    processing_timestamp: datetime
    model_versions: List[str]
    validation_status: str
    confidence_tier: str
    test_batch_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestDataPartition:
    """Defines data partitioning strategy"""
    year: int
    month: int
    day: int
    phase: str
    model_combination: str
    confidence_tier: str
    validation_status: str
    test_batch_id: str
    
    def to_path(self) -> Path:
        """Generate partition path"""
        return Path(
            f"test_outputs/partitioned/"
            f"year={self.year}/month={self.month:02d}/day={self.day:02d}/"
            f"phase={self.phase}/models={self.model_combination}/"
            f"confidence={self.confidence_tier}/status={self.validation_status}/"
            f"batch={self.test_batch_id}"
        )


class DataPipelineOrchestrator:
    """
    Orchestrates test data pipeline with modern data engineering patterns
    - Event-driven architecture for real-time processing
    - Delta Lake pattern for versioned data storage
    - Lineage tracking for reproducibility
    - Quality gates at each processing stage
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = ConfigLoader(config or {})
        self.base_path = Path("test_outputs")
        self.cache_ttl = timedelta(hours=24)
        self.processing_metrics = {}
        
        # Initialize storage structure
        self._initialize_storage_structure()
    
    def _initialize_storage_structure(self):
        """Initialize Delta Lake-style directory structure"""
        storage_dirs = [
            "raw/golden_datasets",
            "raw/synthetic",
            "raw/production_samples",
            "processed/phase1_results",
            "processed/phase2_results", 
            "processed/intermediate",
            "analytics/performance_metrics",
            "analytics/quality_assessments",
            "analytics/comparison_reports",
            "partitioned",  # Partitioned storage
            "streaming",    # Real-time processing buffer
            "cache"         # Temporary computation cache
        ]
        
        for dir_path in storage_dirs:
            (self.base_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    async def process_single_phase(self, 
                                   phase_num: int,
                                   input_data: Dict[str, Any],
                                   models: List[str],
                                   processing_mode: ProcessingMode = ProcessingMode.STREAMING,
                                   cache_intermediate: bool = True) -> Dict[str, Any]:
        """
        Process single phase with configurable execution mode
        
        Args:
            phase_num: Phase number (1 or 2)
            input_data: Input data for processing
            models: AI models to use
            processing_mode: Execution mode (streaming/micro_batch/batch)
            cache_intermediate: Whether to cache results
        """
        logger.info(f"Processing Phase {phase_num} in {processing_mode.value} mode")
        
        # Generate data lineage
        lineage = await self._create_lineage(input_data, phase_num, models)
        
        # Check cache first (if enabled)
        if cache_intermediate:
            cached_result = await self._check_cache(lineage)
            if cached_result:
                logger.info("Using cached result")
                return cached_result
        
        try:
            # Process based on mode
            if processing_mode == ProcessingMode.STREAMING:
                result = await self._process_streaming(phase_num, input_data, models, lineage)
            elif processing_mode == ProcessingMode.MICRO_BATCH:
                result = await self._process_micro_batch(phase_num, [input_data], models, lineage)
                result = result[0]  # Extract single result
            else:  # BATCH
                result = await self._process_batch(phase_num, [input_data], models, lineage)
                result = result[0]  # Extract single result
            
            # Store result with partitioning
            await self._store_with_partitioning(result, lineage)
            
            # Cache if enabled
            if cache_intermediate:
                await self._cache_result(lineage, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Phase {phase_num} processing failed: {e}")
            # Store failure with metadata
            await self._store_failure(input_data, lineage, str(e))
            raise
    
    async def process_phase_transition(self,
                                       phase1_result: Dict[str, Any],
                                       models_phase2: List[str],
                                       validation_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Process Phase 1 → Phase 2 transition with quality gates
        
        Args:
            phase1_result: Phase 1 output
            models_phase2: Models to use for Phase 2
            validation_threshold: Minimum confidence threshold
        """
        logger.info("Processing Phase 1 → Phase 2 transition")
        
        # Quality gate validation
        if not await self._validate_phase_transition(phase1_result, validation_threshold):
            raise ValueError(f"Phase 1 result below quality threshold ({validation_threshold})")
        
        # Transform Phase 1 output to Phase 2 input format
        phase2_input = await self._transform_p1_to_p2(phase1_result)
        
        # Process Phase 2 with dependency tracking
        phase2_result = await self.process_single_phase(
            phase_num=2,
            input_data=phase2_input,
            models=models_phase2,
            processing_mode=ProcessingMode.STREAMING
        )
        
        # Create transition metadata
        transition_metadata = {
            'phase1_hash': hashlib.md5(json.dumps(phase1_result, sort_keys=True).encode()).hexdigest(),
            'phase2_hash': hashlib.md5(json.dumps(phase2_result, sort_keys=True).encode()).hexdigest(),
            'transition_timestamp': datetime.now().isoformat(),
            'quality_gate_passed': True,
            'validation_threshold': validation_threshold
        }
        
        # Store transition record
        await self._store_transition_record(phase1_result, phase2_result, transition_metadata)
        
        return {
            'phase1_result': phase1_result,
            'phase2_result': phase2_result,
            'transition_metadata': transition_metadata
        }
    
    async def create_test_dataset_from_template(self,
                                                template_name: str,
                                                variations: int = 10,
                                                confidence_distribution: Dict[str, float] = None) -> List[Dict]:
        """
        Generate synthetic test dataset from template with controlled variations
        """
        confidence_distribution = confidence_distribution or {
            'high': 0.6, 'medium': 0.3, 'low': 0.1
        }
        
        template_path = self.base_path / "raw" / "golden_datasets" / f"{template_name}.json"
        if not template_path.exists():
            raise FileNotFoundError(f"Template {template_name} not found")
        
        with open(template_path, 'r') as f:
            template_data = json.load(f)
        
        # Generate variations using statistical methods
        dataset = []
        for i in range(variations):
            # Vary room sizes, materials, complexity
            varied_data = await self._apply_controlled_variations(template_data, i)
            
            # Assign confidence tier based on distribution
            confidence_tier = self._sample_confidence_tier(confidence_distribution)
            varied_data['_test_metadata'] = {
                'template': template_name,
                'variation_id': i,
                'confidence_tier': confidence_tier,
                'generated_timestamp': datetime.now().isoformat()
            }
            
            dataset.append(varied_data)
        
        return dataset
    
    async def run_comparative_analysis(self,
                                       model_combinations: List[List[str]],
                                       test_scenarios: List[Dict],
                                       phases: List[int] = [1, 2]) -> Dict[str, Any]:
        """
        Run comparative analysis across model combinations and scenarios
        """
        logger.info(f"Running comparative analysis: {len(model_combinations)} combinations, {len(test_scenarios)} scenarios")
        
        results = {}
        
        # Process all combinations in parallel
        tasks = []
        for combo_idx, models in enumerate(model_combinations):
            for scenario_idx, scenario in enumerate(test_scenarios):
                for phase in phases:
                    task = self._process_combination_scenario(
                        combo_idx, models, scenario_idx, scenario, phase
                    )
                    tasks.append(task)
        
        # Execute with controlled concurrency
        completed_results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(asyncio.run, task) for task in tasks]
            for future in as_completed(futures):
                completed_results.append(await future.result())
        
        # Aggregate and analyze results
        analysis = await self._aggregate_comparative_results(completed_results)
        
        # Store analysis results
        analysis_path = self.base_path / "analytics" / "comparison_reports" / f"comparative_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return analysis
    
    async def _create_lineage(self, input_data: Dict, phase: int, models: List[str]) -> DataLineage:
        """Create data lineage record"""
        source_hash = hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
        
        return DataLineage(
            source_hash=source_hash,
            phase_dependency=None,  # Set by caller if needed
            processing_timestamp=datetime.now(),
            model_versions=models,
            validation_status="pending",
            confidence_tier="unknown",
            test_batch_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{source_hash[:8]}"
        )
    
    async def _process_streaming(self, phase: int, data: Dict, models: List[str], lineage: DataLineage) -> Dict:
        """Process single record in streaming mode"""
        from src.phases.phase1_processor import Phase1Processor
        from src.phases.phase2_processor import Phase2Processor
        
        if phase == 1:
            processor = Phase1Processor()
            result = await processor.process(data, models)
        else:  # phase == 2
            processor = Phase2Processor()
            result = await processor.process(data, models)
        
        # Update lineage with results
        lineage.validation_status = "success" if result.get('success') else "failure"
        lineage.confidence_tier = self._classify_confidence(result.get('confidence_score', 0))
        
        result['_lineage'] = lineage.__dict__
        return result
    
    async def _process_micro_batch(self, phase: int, data_batch: List[Dict], models: List[str], lineage: DataLineage) -> List[Dict]:
        """Process small batch of records"""
        results = []
        for data in data_batch:
            result = await self._process_streaming(phase, data, models, lineage)
            results.append(result)
        return results
    
    async def _process_batch(self, phase: int, data_batch: List[Dict], models: List[str], lineage: DataLineage) -> List[Dict]:
        """Process large batch with optimizations"""
        # Batch processing optimizations:
        # 1. Model connection pooling
        # 2. Parallel processing
        # 3. Result aggregation
        
        tasks = []
        for data in data_batch:
            task = self._process_streaming(phase, data, models, lineage)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
            else:
                successful_results.append(result)
        
        return successful_results
    
    def _classify_confidence(self, confidence_score: float) -> str:
        """Classify confidence into tiers"""
        if confidence_score >= 0.8:
            return "high"
        elif confidence_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    async def _store_with_partitioning(self, result: Dict, lineage: DataLineage):
        """Store result with hierarchical partitioning"""
        timestamp = lineage.processing_timestamp
        
        partition = TestDataPartition(
            year=timestamp.year,
            month=timestamp.month,
            day=timestamp.day,
            phase=f"phase{result.get('phase', 'unknown')}",
            model_combination="_".join(lineage.model_versions),
            confidence_tier=lineage.confidence_tier,
            validation_status=lineage.validation_status,
            test_batch_id=lineage.test_batch_id
        )
        
        partition_path = self.base_path / partition.to_path()
        partition_path.mkdir(parents=True, exist_ok=True)
        
        # Store result with metadata
        result_file = partition_path / f"{lineage.source_hash[:8]}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Update partition metadata
        await self._update_partition_metadata(partition, lineage)
    
    async def _update_partition_metadata(self, partition: TestDataPartition, lineage: DataLineage):
        """Update partition-level metadata for queries"""
        metadata_file = self.base_path / partition.to_path() / "_metadata.json"
        
        # Load existing metadata
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Update with new record
        if 'records' not in metadata:
            metadata['records'] = []
        
        metadata['records'].append({
            'source_hash': lineage.source_hash,
            'processing_timestamp': lineage.processing_timestamp.isoformat(),
            'confidence_tier': lineage.confidence_tier,
            'validation_status': lineage.validation_status
        })
        
        metadata['last_updated'] = datetime.now().isoformat()
        metadata['record_count'] = len(metadata['records'])
        
        # Write updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    # Additional helper methods would be implemented here...
    # _check_cache, _cache_result, _validate_phase_transition, etc.


# Usage example:
async def main():
    """Example usage of the data pipeline orchestrator"""
    orchestrator = DataPipelineOrchestrator()
    
    # Single phase testing
    result = await orchestrator.process_single_phase(
        phase_num=1,
        input_data={"rooms": [{"name": "living_room", "sqft": 200}]},
        models=["gpt4", "claude"],
        processing_mode=ProcessingMode.STREAMING
    )
    
    # Phase transition testing
    transition_result = await orchestrator.process_phase_transition(
        phase1_result=result,
        models_phase2=["gpt4", "gemini"],
        validation_threshold=0.7
    )
    
    print(f"Pipeline execution completed: {transition_result['transition_metadata']}")


if __name__ == "__main__":
    asyncio.run(main())