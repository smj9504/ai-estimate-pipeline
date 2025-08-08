"""
Model Combination Tester
Executes AI model combination tests with comprehensive result tracking.
"""
import asyncio
import time
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json

from .test_matrix import TestConfiguration, ModelType, ValidationMode, ProcessingMode
from ..utils.test_data_loader import TestDataLoader, TestDataSet
from src.models.model_interface import ModelOrchestrator
from src.models.data_models import ModelResponse, ProjectData, MergedEstimate
from src.processors.result_merger import ResultMerger
from src.validators.estimation_validator import ComprehensiveValidator, ValidationResult


logger = logging.getLogger(__name__)


@dataclass
class CombinationTestResult:
    """Results from a single model combination test"""
    configuration: TestConfiguration
    success: bool
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Model execution results
    model_responses: List[ModelResponse] = field(default_factory=list)
    models_responded: int = 0
    total_models: int = 0
    
    # Merged results
    merged_estimate: Optional[MergedEstimate] = None
    overall_confidence: float = 0.0
    consensus_level: float = 0.0
    
    # Validation results  
    validation_result: Optional[ValidationResult] = None
    validation_score: float = 0.0
    
    # Performance metrics
    processing_time_per_model: Dict[str, float] = field(default_factory=dict)
    total_work_items: int = 0
    unique_rooms_processed: int = 0
    
    # Error information
    error_message: Optional[str] = None
    failed_models: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Output paths (if saved)
    output_files: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived metrics after initialization"""
        self.total_models = len(self.configuration.models)
        
        if self.merged_estimate:
            self.overall_confidence = self.merged_estimate.overall_confidence
            self.consensus_level = self.merged_estimate.metadata.consensus_level
            self.total_work_items = self.merged_estimate.total_work_items
            self.unique_rooms_processed = len(self.merged_estimate.rooms)
        
        if self.validation_result:
            self.validation_score = self.validation_result.score
    
    @property
    def model_success_rate(self) -> float:
        """Calculate model success rate"""
        if self.total_models == 0:
            return 0.0
        return self.models_responded / self.total_models
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time per model"""
        if not self.processing_time_per_model:
            return 0.0
        return sum(self.processing_time_per_model.values()) / len(self.processing_time_per_model)
    
    @property
    def is_successful(self) -> bool:
        """Check if test was fully successful"""
        return (self.success and 
                self.models_responded > 0 and 
                self.merged_estimate is not None and
                self.overall_confidence > 0)
    
    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-1)"""
        if not self.is_successful:
            return 0.0
        
        # Weighted combination of various quality metrics
        confidence_weight = 0.4
        consensus_weight = 0.3
        validation_weight = 0.2
        success_rate_weight = 0.1
        
        quality = (
            self.overall_confidence * confidence_weight +
            self.consensus_level * consensus_weight +
            self.validation_score * validation_weight +
            self.model_success_rate * success_rate_weight
        )
        
        return min(1.0, max(0.0, quality))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "configuration": self.configuration.to_dict(),
            "success": self.success,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
            "models_responded": self.models_responded,
            "total_models": self.total_models,
            "overall_confidence": self.overall_confidence,
            "consensus_level": self.consensus_level,
            "validation_score": self.validation_score,
            "total_work_items": self.total_work_items,
            "unique_rooms_processed": self.unique_rooms_processed,
            "model_success_rate": self.model_success_rate,
            "average_processing_time": self.average_processing_time,
            "quality_score": self.quality_score,
            "error_message": self.error_message,
            "failed_models": self.failed_models,
            "warnings": self.warnings,
            "processing_time_per_model": self.processing_time_per_model,
            "output_files": self.output_files
        }


class ModelCombinationTester:
    """Main class for executing model combination tests"""
    
    def __init__(self, 
                 test_data_loader: Optional[TestDataLoader] = None,
                 output_directory: Optional[str] = None,
                 enable_caching: bool = True):
        """Initialize combination tester
        
        Args:
            test_data_loader: Test data loader instance
            output_directory: Directory for saving test outputs
            enable_caching: Enable caching of test results
        """
        self.test_data_loader = test_data_loader or TestDataLoader()
        self.output_directory = Path(output_directory) if output_directory else Path("test_outputs/model_combinations")
        self.enable_caching = enable_caching
        
        # Create output directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.orchestrator = None
        self.result_merger = None
        self.validator = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize testing components"""
        try:
            self.orchestrator = ModelOrchestrator()
            self.result_merger = ResultMerger()
            self.validator = ComprehensiveValidator()
            logger.info("Model combination tester initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def run_single_test(self, 
                            configuration: TestConfiguration,
                            test_dataset: TestDataSet,
                            prompt: str) -> CombinationTestResult:
        """Run a single model combination test
        
        Args:
            configuration: Test configuration to run
            test_dataset: Test data to use
            prompt: AI model prompt
            
        Returns:
            Test result with all metrics
        """
        start_time = time.time()
        result = CombinationTestResult(configuration=configuration)
        
        try:
            logger.info(f"Starting test: {configuration.test_name}")
            
            # Prepare project data
            project_data = ProjectData.from_json_list(test_dataset.combined_project_data)
            
            # Execute models based on configuration
            if configuration.processing_mode == ProcessingMode.SEQUENTIAL:
                model_responses = await self._run_models_sequential(
                    configuration.model_names, prompt, test_dataset.combined_project_data
                )
            else:  # PARALLEL or SINGLE_MODEL
                model_responses = await self._run_models_parallel(
                    configuration.model_names, prompt, test_dataset.combined_project_data
                )
            
            result.model_responses = model_responses
            result.models_responded = len([r for r in model_responses if r is not None])
            
            # Track per-model processing times
            for response in model_responses:
                if response:
                    result.processing_time_per_model[response.model_name] = response.processing_time
            
            # Merge results if multiple models
            if len(model_responses) > 1:
                merged_result = self.result_merger.merge_results(model_responses)
                result.merged_estimate = merged_result
            elif len(model_responses) == 1 and model_responses[0]:
                # Convert single model response to merged estimate format
                result.merged_estimate = self._convert_single_response_to_merged(model_responses[0])
            
            # Validate results
            if result.merged_estimate and result.merged_estimate.rooms:
                validation_result = await self._validate_results(project_data, result.merged_estimate)
                result.validation_result = validation_result
            
            # Save outputs if requested
            if configuration.save_outputs:
                await self._save_test_outputs(result)
            
            result.success = result.is_successful
            result.execution_time = time.time() - start_time
            
            logger.info(f"Test completed: {configuration.test_name} "
                       f"(success: {result.success}, time: {result.execution_time:.2f}s)")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.execution_time = time.time() - start_time
            logger.error(f"Test failed: {configuration.test_name} - {e}")
        
        return result
    
    async def _run_models_parallel(self, 
                                 model_names: List[str],
                                 prompt: str, 
                                 json_data: List[Dict[str, Any]]) -> List[ModelResponse]:
        """Run models in parallel"""
        return await self.orchestrator.run_parallel(prompt, json_data, model_names)
    
    async def _run_models_sequential(self,
                                   model_names: List[str],
                                   prompt: str,
                                   json_data: List[Dict[str, Any]]) -> List[ModelResponse]:
        """Run models sequentially"""
        responses = []
        for model_name in model_names:
            try:
                model_responses = await self.orchestrator.run_parallel(prompt, json_data, [model_name])
                if model_responses:
                    responses.extend(model_responses)
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                # Add None to track failed model
                responses.append(None)
        
        return responses
    
    def _convert_single_response_to_merged(self, response: ModelResponse) -> MergedEstimate:
        """Convert single model response to MergedEstimate format"""
        # This is a simplified conversion - in reality you might want more sophisticated logic
        from src.models.data_models import MergedEstimate, EstimateMetadata
        
        metadata = EstimateMetadata(
            models_used=[response.model_name],
            processing_time_total=response.processing_time,
            consensus_level=1.0,  # Single model = perfect consensus
            confidence_interval_lower=response.confidence_self_assessment * 0.9,
            confidence_interval_upper=response.confidence_self_assessment * 1.1
        )
        
        return MergedEstimate(
            rooms=response.room_estimates,
            total_work_items=response.total_work_items,
            overall_confidence=response.confidence_self_assessment,
            metadata=metadata
        )
    
    async def _validate_results(self, 
                              project_data: ProjectData,
                              merged_estimate: MergedEstimate) -> ValidationResult:
        """Validate merged results against project data"""
        try:
            # For now, validate the first room as an example
            if project_data.floors and project_data.floors[0].rooms and merged_estimate.rooms:
                first_room = project_data.floors[0].rooms[0]
                first_estimate = merged_estimate.rooms[0]
                
                # Extract work items from estimate
                work_items = first_estimate.get('tasks', [])
                
                return self.validator.validate_single_room(first_room, work_items)
            
            # Return default validation if no rooms to validate
            from src.validators.estimation_validator import ValidationResult
            return ValidationResult(
                is_valid=True,
                score=0.5,
                issues=[],
                details={"note": "No rooms available for validation"}
            )
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            from src.validators.estimation_validator import ValidationResult
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=[f"Validation error: {str(e)}"],
                details={}
            )
    
    async def _save_test_outputs(self, result: CombinationTestResult):
        """Save test outputs to files"""
        try:
            test_dir = self.output_directory / result.configuration.test_name
            test_dir.mkdir(parents=True, exist_ok=True)
            
            # Save result summary
            result_file = test_dir / "result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            result.output_files["result"] = str(result_file)
            
            # Save individual model responses
            if result.model_responses:
                responses_file = test_dir / "model_responses.json"
                responses_data = []
                for response in result.model_responses:
                    if response:
                        responses_data.append({
                            "model_name": response.model_name,
                            "processing_time": response.processing_time,
                            "total_work_items": response.total_work_items,
                            "confidence_self_assessment": response.confidence_self_assessment,
                            "room_estimates": response.room_estimates
                        })
                
                with open(responses_file, 'w', encoding='utf-8') as f:
                    json.dump(responses_data, f, indent=2, ensure_ascii=False)
                result.output_files["model_responses"] = str(responses_file)
            
            # Save merged estimate
            if result.merged_estimate:
                merged_file = test_dir / "merged_estimate.json"
                # Convert to dict for serialization
                merged_dict = {
                    "rooms": result.merged_estimate.rooms,
                    "total_work_items": result.merged_estimate.total_work_items,
                    "overall_confidence": result.merged_estimate.overall_confidence,
                    "metadata": {
                        "models_used": result.merged_estimate.metadata.models_used,
                        "processing_time_total": result.merged_estimate.metadata.processing_time_total,
                        "consensus_level": result.merged_estimate.metadata.consensus_level
                    }
                }
                
                with open(merged_file, 'w', encoding='utf-8') as f:
                    json.dump(merged_dict, f, indent=2, ensure_ascii=False)
                result.output_files["merged_estimate"] = str(merged_file)
            
            logger.info(f"Test outputs saved to: {test_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save test outputs: {e}")
            result.warnings.append(f"Output save failed: {str(e)}")
    
    async def run_test_batch(self,
                           configurations: List[TestConfiguration],
                           test_dataset: Optional[TestDataSet] = None,
                           prompt: Optional[str] = None,
                           max_concurrent: int = 3) -> List[CombinationTestResult]:
        """Run a batch of model combination tests
        
        Args:
            configurations: List of test configurations to run
            test_dataset: Test data (will load default if not provided)
            prompt: AI prompt (will use default if not provided)
            max_concurrent: Maximum concurrent tests
            
        Returns:
            List of test results
        """
        if not test_dataset:
            test_dataset = self.test_data_loader.get_test_dataset()
        
        if not prompt:
            prompt = self._get_default_prompt()
        
        logger.info(f"Starting batch of {len(configurations)} tests with max_concurrent={max_concurrent}")
        
        # Run tests with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_with_semaphore(config):
            async with semaphore:
                return await self.run_single_test(config, test_dataset, prompt)
        
        results = await asyncio.gather(
            *[run_single_with_semaphore(config) for config in configurations],
            return_exceptions=True
        )
        
        # Handle exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Test {configurations[i].test_name} failed with exception: {result}")
                # Create failed result
                failed_result = CombinationTestResult(
                    configuration=configurations[i],
                    success=False,
                    execution_time=0.0,
                    error_message=str(result)
                )
                valid_results.append(failed_result)
            else:
                valid_results.append(result)
        
        logger.info(f"Batch completed: {len(valid_results)} results")
        return valid_results
    
    def _get_default_prompt(self) -> str:
        """Get default prompt for testing"""
        return """You are a Senior Reconstruction Estimating Specialist in the DMV area.

Generate a detailed reconstruction estimate by analyzing the provided JSON data for each room.
Focus on:
1. Remove & Replace logic - remove existing materials, install new ones
2. Use provided measurements accurately
3. Include special work and premiums
4. Account for demo_scope (already completed demolition)
5. Include protection and detach/reset work from additional_notes

For each room, provide:
- Task name and description
- Necessity level (required/optional/nice-to-have)
- Quantity and unit
- Consider high ceiling premiums for walls/ceilings >9 feet

Return comprehensive work scope with proper quantities."""
    
    def get_test_statistics(self, results: List[CombinationTestResult]) -> Dict[str, Any]:
        """Calculate statistics for a batch of test results"""
        if not results:
            return {"error": "No results provided"}
        
        total_tests = len(results)
        successful_tests = len([r for r in results if r.success])
        failed_tests = total_tests - successful_tests
        
        # Performance metrics
        execution_times = [r.execution_time for r in results if r.success]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # Quality metrics
        quality_scores = [r.quality_score for r in results if r.success]
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Model performance
        model_usage = {}
        for result in results:
            for model in result.configuration.model_names:
                if model not in model_usage:
                    model_usage[model] = {"total": 0, "successful": 0}
                model_usage[model]["total"] += 1
                if result.success:
                    model_usage[model]["successful"] += 1
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "average_execution_time": avg_execution_time,
            "average_quality_score": avg_quality_score,
            "model_usage": model_usage,
            "best_performing_config": max(results, key=lambda r: r.quality_score).configuration.test_name if results else None,
            "fastest_config": min([r for r in results if r.success], 
                                key=lambda r: r.execution_time).configuration.test_name if successful_tests > 0 else None
        }