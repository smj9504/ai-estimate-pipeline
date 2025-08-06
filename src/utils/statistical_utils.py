# src/utils/statistical_utils.py
import numpy as np
from typing import List, Tuple, Dict, Any

class StatisticalProcessor:
    """통계 처리 유틸리티"""
    
    @staticmethod
    def calculate_iqr_bounds(values: List[float], multiplier: float = 1.5) -> Tuple[float, float]:
        """IQR 방식으로 아웃라이어 경계값 계산"""
        if len(values) < 2:
            return float('-inf'), float('inf')
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        return lower_bound, upper_bound
    
    @staticmethod
    def remove_outliers_iqr(values: List[float], multiplier: float = 1.5) -> List[float]:
        """IQR 방식으로 아웃라이어 제거"""
        if len(values) <= 2:
            return values
        
        lower_bound, upper_bound = StatisticalProcessor.calculate_iqr_bounds(values, multiplier)
        return [v for v in values if lower_bound <= v <= upper_bound]
    
    @staticmethod
    def calculate_z_scores(values: List[float]) -> List[float]:
        """Z-score 계산"""
        if len(values) <= 1:
            return [0.0] * len(values)
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        if std == 0:
            return [0.0] * len(values)
        
        return [(v - mean) / std for v in values]
    
    @staticmethod
    def remove_outliers_zscore(values: List[float], threshold: float = 3.0) -> List[float]:
        """Z-score 방식으로 아웃라이어 제거"""
        if len(values) <= 2:
            return values
        
        z_scores = StatisticalProcessor.calculate_z_scores(values)
        return [values[i] for i, z in enumerate(z_scores) if abs(z) <= threshold]
    
    @staticmethod
    def weighted_average(values: List[float], weights: List[float]) -> float:
        """가중 평균 계산"""
        if not values or not weights or len(values) != len(weights):
            return 0.0
        
        if sum(weights) == 0:
            return np.mean(values)
        
        return np.average(values, weights=weights)
    
    @staticmethod
    def calculate_variance_metrics(values: List[float]) -> Dict[str, float]:
        """분산 메트릭스 계산"""
        if len(values) <= 1:
            return {
                'mean': values[0] if values else 0.0,
                'std': 0.0,
                'cv': 0.0,  # coefficient of variation
                'range': 0.0,
                'iqr': 0.0
            }
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        return {
            'mean': mean,
            'std': std,
            'cv': (std / mean * 100) if mean != 0 else 0.0,
            'range': max(values) - min(values),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25)
        }
    
    @staticmethod
    def calculate_consensus_score(agreement_matrix: List[List[bool]]) -> float:
        """합의 점수 계산 (0-1)"""
        if not agreement_matrix:
            return 0.0
        
        total_comparisons = 0
        agreements = 0
        
        for i in range(len(agreement_matrix)):
            for j in range(i + 1, len(agreement_matrix)):
                total_comparisons += len(agreement_matrix[i])
                agreements += sum(1 for k in range(len(agreement_matrix[i])) 
                                if agreement_matrix[i][k] == agreement_matrix[j][k])
        
        return agreements / total_comparisons if total_comparisons > 0 else 0.0
    
    @staticmethod
    def calculate_percentile_range(values: List[float], lower_pct: float = 25, upper_pct: float = 75) -> Tuple[float, float]:
        """백분위수 범위 계산"""
        if not values:
            return 0.0, 0.0
        
        return np.percentile(values, lower_pct), np.percentile(values, upper_pct)
    
    @staticmethod
    def robust_mean(values: List[float], trim_percent: float = 0.1) -> float:
        """로버스트 평균 (양 끝단 제거 후 평균)"""
        if not values:
            return 0.0
        
        if len(values) <= 2:
            return np.mean(values)
        
        trim_count = max(1, int(len(values) * trim_percent))
        sorted_values = sorted(values)
        trimmed_values = sorted_values[trim_count:-trim_count] if trim_count < len(values)//2 else sorted_values
        
        return np.mean(trimmed_values)
    
    @staticmethod
    def calculate_confidence_interval(values: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
        """신뢰구간 계산"""
        if len(values) < 2:
            mean_val = values[0] if values else 0.0
            return mean_val, mean_val
        
        mean = np.mean(values)
        std_err = np.std(values, ddof=1) / np.sqrt(len(values))
        
        # 정규분포 가정하에 95% 신뢰구간 (간단한 근사)
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 99%인 경우
        margin_error = z_score * std_err
        
        return mean - margin_error, mean + margin_error
    
    @staticmethod
    def detect_outliers_multiple_methods(values: List[float]) -> Dict[str, List[int]]:
        """여러 방법으로 아웃라이어 감지"""
        if len(values) < 3:
            return {'iqr': [], 'zscore': []}
        
        outlier_indices = {'iqr': [], 'zscore': []}
        
        # IQR 방법
        lower_bound, upper_bound = StatisticalProcessor.calculate_iqr_bounds(values)
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                outlier_indices['iqr'].append(i)
        
        # Z-score 방법
        z_scores = StatisticalProcessor.calculate_z_scores(values)
        for i, z_score in enumerate(z_scores):
            if abs(z_score) > 3.0:
                outlier_indices['zscore'].append(i)
        
        return outlier_indices
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """안전한 나누기 (0으로 나누기 방지)"""
        return numerator / denominator if denominator != 0 else default
    
    @staticmethod
    def normalize_values(values: List[float], method: str = 'minmax') -> List[float]:
        """값들을 정규화"""
        if not values:
            return []
        
        if len(values) == 1:
            return [1.0]
        
        if method == 'minmax':
            min_val = min(values)
            max_val = max(values)
            if max_val == min_val:
                return [1.0] * len(values)
            return [(v - min_val) / (max_val - min_val) for v in values]
        
        elif method == 'zscore':
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            if std == 0:
                return [0.0] * len(values)
            return [(v - mean) / std for v in values]
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")