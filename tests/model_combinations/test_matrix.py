"""
Test Matrix for Model Combinations
Defines all possible AI model combinations and test configurations.
"""
import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterator, Optional, Set
from enum import Enum


class ModelType(Enum):
    """Available AI model types"""
    GPT4 = "gpt4"
    CLAUDE = "claude" 
    GEMINI = "gemini"


class ValidationMode(Enum):
    """Validation modes for testing"""
    STRICT = "strict"
    BALANCED = "balanced" 
    LENIENT = "lenient"


class ProcessingMode(Enum):
    """Processing modes for testing"""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    SINGLE_MODEL = "single"


@dataclass
class TestConfiguration:
    """Configuration for a single test run"""
    models: List[ModelType]
    validation_mode: ValidationMode
    processing_mode: ProcessingMode
    timeout_seconds: int = 300
    retry_attempts: int = 2
    save_outputs: bool = True
    test_name: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.models:
            raise ValueError("At least one model must be specified")
        
        if self.processing_mode == ProcessingMode.SINGLE_MODEL and len(self.models) > 1:
            raise ValueError("Single model processing mode requires exactly one model")
        
        if len(self.models) > 1 and self.processing_mode == ProcessingMode.SINGLE_MODEL:
            self.processing_mode = ProcessingMode.PARALLEL
        
        # Generate test name if not provided
        if not self.test_name:
            model_names = "_".join([m.value for m in self.models])
            self.test_name = f"{model_names}_{self.validation_mode.value}_{self.processing_mode.value}"
    
    @property
    def model_names(self) -> List[str]:
        """Get model names as strings"""
        return [model.value for model in self.models]
    
    @property
    def is_single_model(self) -> bool:
        """Check if this is a single model test"""
        return len(self.models) == 1
    
    @property
    def is_multi_model(self) -> bool:
        """Check if this is a multi-model test"""
        return len(self.models) > 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "models": self.model_names,
            "validation_mode": self.validation_mode.value,
            "processing_mode": self.processing_mode.value,
            "timeout_seconds": self.timeout_seconds,
            "retry_attempts": self.retry_attempts,
            "save_outputs": self.save_outputs,
            "test_name": self.test_name,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestConfiguration':
        """Create configuration from dictionary"""
        return cls(
            models=[ModelType(m) for m in data["models"]],
            validation_mode=ValidationMode(data["validation_mode"]),
            processing_mode=ProcessingMode(data["processing_mode"]),
            timeout_seconds=data.get("timeout_seconds", 300),
            retry_attempts=data.get("retry_attempts", 2),
            save_outputs=data.get("save_outputs", True),
            test_name=data.get("test_name"),
            description=data.get("description")
        )


class ModelTestMatrix:
    """Test matrix generator for all possible model combinations"""
    
    def __init__(self, 
                 available_models: Optional[List[ModelType]] = None,
                 validation_modes: Optional[List[ValidationMode]] = None,
                 processing_modes: Optional[List[ProcessingMode]] = None,
                 include_single_model: bool = True,
                 include_multi_model: bool = True):
        """Initialize test matrix generator
        
        Args:
            available_models: Models to include in testing (default: all)
            validation_modes: Validation modes to test (default: all)
            processing_modes: Processing modes to test (default: parallel/sequential)
            include_single_model: Include single model tests
            include_multi_model: Include multi-model combination tests
        """
        self.available_models = available_models or list(ModelType)
        self.validation_modes = validation_modes or list(ValidationMode)
        self.processing_modes = processing_modes or [ProcessingMode.PARALLEL, ProcessingMode.SEQUENTIAL]
        self.include_single_model = include_single_model
        self.include_multi_model = include_multi_model
        
        if ProcessingMode.SINGLE_MODEL in self.processing_modes and not include_single_model:
            self.processing_modes.remove(ProcessingMode.SINGLE_MODEL)
    
    def generate_single_model_configurations(self) -> List[TestConfiguration]:
        """Generate all single model test configurations"""
        configurations = []
        
        if not self.include_single_model:
            return configurations
        
        for model in self.available_models:
            for validation_mode in self.validation_modes:
                config = TestConfiguration(
                    models=[model],
                    validation_mode=validation_mode,
                    processing_mode=ProcessingMode.SINGLE_MODEL,
                    description=f"Single model test: {model.value} with {validation_mode.value} validation"
                )
                configurations.append(config)
        
        return configurations
    
    def generate_model_combinations(self, min_models: int = 2, max_models: int = 3) -> List[List[ModelType]]:
        """Generate all possible model combinations
        
        Args:
            min_models: Minimum number of models in combination
            max_models: Maximum number of models in combination
            
        Returns:
            List of model combinations
        """
        combinations = []
        
        for r in range(min_models, min(max_models + 1, len(self.available_models) + 1)):
            for combo in itertools.combinations(self.available_models, r):
                combinations.append(list(combo))
        
        return combinations
    
    def generate_multi_model_configurations(self) -> List[TestConfiguration]:
        """Generate all multi-model test configurations"""
        configurations = []
        
        if not self.include_multi_model:
            return configurations
        
        # Generate all model combinations (2 or 3 models)
        model_combinations = self.generate_model_combinations(min_models=2, max_models=3)
        
        for models in model_combinations:
            for validation_mode in self.validation_modes:
                # Only use parallel/sequential for multi-model
                processing_modes = [pm for pm in self.processing_modes 
                                  if pm != ProcessingMode.SINGLE_MODEL]
                
                for processing_mode in processing_modes:
                    model_names = " + ".join([m.value for m in models])
                    config = TestConfiguration(
                        models=models,
                        validation_mode=validation_mode,
                        processing_mode=processing_mode,
                        description=f"Multi-model test: {model_names} with {validation_mode.value} validation"
                    )
                    configurations.append(config)
        
        return configurations
    
    def generate_all_configurations(self) -> List[TestConfiguration]:
        """Generate all possible test configurations"""
        configurations = []
        
        if self.include_single_model:
            configurations.extend(self.generate_single_model_configurations())
        
        if self.include_multi_model:
            configurations.extend(self.generate_multi_model_configurations())
        
        return configurations
    
    def generate_essential_configurations(self) -> List[TestConfiguration]:
        """Generate essential test configurations for quick testing
        
        This creates a smaller subset focusing on:
        - Each individual model with balanced validation
        - All 3-model combination with balanced validation  
        - Performance comparison configurations
        """
        essential_configs = []
        
        # Single model tests with balanced validation
        for model in self.available_models:
            config = TestConfiguration(
                models=[model],
                validation_mode=ValidationMode.BALANCED,
                processing_mode=ProcessingMode.SINGLE_MODEL,
                description=f"Essential single model test: {model.value}"
            )
            essential_configs.append(config)
        
        # All models combination with balanced validation
        if len(self.available_models) >= 2:
            config = TestConfiguration(
                models=self.available_models,
                validation_mode=ValidationMode.BALANCED,
                processing_mode=ProcessingMode.PARALLEL,
                description="Essential all-models combination test"
            )
            essential_configs.append(config)
        
        # Performance comparison: best performing pairs
        high_performance_pairs = [
            [ModelType.GPT4, ModelType.CLAUDE],
            [ModelType.GPT4, ModelType.GEMINI], 
            [ModelType.CLAUDE, ModelType.GEMINI]
        ]
        
        for pair in high_performance_pairs:
            if all(model in self.available_models for model in pair):
                config = TestConfiguration(
                    models=pair,
                    validation_mode=ValidationMode.BALANCED,
                    processing_mode=ProcessingMode.PARALLEL,
                    description=f"Essential pair test: {' + '.join([m.value for m in pair])}"
                )
                essential_configs.append(config)
        
        return essential_configs
    
    def generate_performance_comparison_matrix(self) -> List[TestConfiguration]:
        """Generate configurations specifically for performance comparison
        
        This creates configurations designed to compare:
        - Individual model performance 
        - Pair combinations vs single models
        - All models vs pairs vs singles
        """
        comparison_configs = []
        
        # Individual models for baseline
        for model in self.available_models:
            config = TestConfiguration(
                models=[model],
                validation_mode=ValidationMode.BALANCED,
                processing_mode=ProcessingMode.SINGLE_MODEL,
                test_name=f"baseline_{model.value}",
                description=f"Baseline performance test: {model.value}"
            )
            comparison_configs.append(config)
        
        # All pair combinations
        for model1, model2 in itertools.combinations(self.available_models, 2):
            config = TestConfiguration(
                models=[model1, model2],
                validation_mode=ValidationMode.BALANCED,
                processing_mode=ProcessingMode.PARALLEL,
                test_name=f"pair_{model1.value}_{model2.value}",
                description=f"Pair comparison: {model1.value} + {model2.value}"
            )
            comparison_configs.append(config)
        
        # All models if 3 or more available
        if len(self.available_models) >= 3:
            config = TestConfiguration(
                models=self.available_models,
                validation_mode=ValidationMode.BALANCED,
                processing_mode=ProcessingMode.PARALLEL,
                test_name="all_models_comparison",
                description="All models performance comparison"
            )
            comparison_configs.append(config)
        
        return comparison_configs
    
    def filter_configurations(self, 
                            configurations: List[TestConfiguration],
                            model_filter: Optional[Set[ModelType]] = None,
                            validation_filter: Optional[Set[ValidationMode]] = None,
                            processing_filter: Optional[Set[ProcessingMode]] = None,
                            max_configs: Optional[int] = None) -> List[TestConfiguration]:
        """Filter configurations based on criteria
        
        Args:
            configurations: Configurations to filter
            model_filter: Only include configs with these models
            validation_filter: Only include configs with these validation modes
            processing_filter: Only include configs with these processing modes
            max_configs: Maximum number of configurations to return
            
        Returns:
            Filtered list of configurations
        """
        filtered = configurations
        
        if model_filter:
            filtered = [c for c in filtered 
                       if any(model in model_filter for model in c.models)]
        
        if validation_filter:
            filtered = [c for c in filtered 
                       if c.validation_mode in validation_filter]
        
        if processing_filter:
            filtered = [c for c in filtered 
                       if c.processing_mode in processing_filter]
        
        if max_configs and len(filtered) > max_configs:
            # Prioritize multi-model configurations for better coverage
            multi_model = [c for c in filtered if c.is_multi_model]
            single_model = [c for c in filtered if c.is_single_model]
            
            if len(multi_model) >= max_configs:
                filtered = multi_model[:max_configs]
            else:
                remaining = max_configs - len(multi_model)
                filtered = multi_model + single_model[:remaining]
        
        return filtered
    
    def get_configuration_summary(self, configurations: List[TestConfiguration]) -> Dict[str, Any]:
        """Get summary statistics for a list of configurations"""
        total_configs = len(configurations)
        single_model = len([c for c in configurations if c.is_single_model])
        multi_model = len([c for c in configurations if c.is_multi_model])
        
        validation_counts = {}
        for mode in ValidationMode:
            count = len([c for c in configurations if c.validation_mode == mode])
            validation_counts[mode.value] = count
        
        processing_counts = {}
        for mode in ProcessingMode:
            count = len([c for c in configurations if c.processing_mode == mode])
            processing_counts[mode.value] = count
        
        model_usage = {}
        for model in ModelType:
            count = len([c for c in configurations if model in c.models])
            model_usage[model.value] = count
        
        return {
            "total_configurations": total_configs,
            "single_model_tests": single_model,
            "multi_model_tests": multi_model,
            "validation_mode_distribution": validation_counts,
            "processing_mode_distribution": processing_counts,
            "model_usage": model_usage
        }


# Predefined matrix configurations
DEFAULT_MATRIX = ModelTestMatrix()

QUICK_MATRIX = ModelTestMatrix(
    validation_modes=[ValidationMode.BALANCED],
    processing_modes=[ProcessingMode.PARALLEL],
    include_single_model=True,
    include_multi_model=True
)

PERFORMANCE_MATRIX = ModelTestMatrix(
    validation_modes=[ValidationMode.BALANCED],
    processing_modes=[ProcessingMode.PARALLEL, ProcessingMode.SEQUENTIAL],
    include_single_model=True, 
    include_multi_model=True
)