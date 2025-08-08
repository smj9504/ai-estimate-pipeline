# src/tracking/usage_reporter.py
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

from src.tracking.token_tracker import TokenTracker, TokenPricingManager
from src.utils.logger import get_logger

class UsageReporter:
    """Comprehensive usage reporting and analytics system"""
    
    def __init__(self, token_tracker: Optional[TokenTracker] = None):
        self.logger = get_logger('usage_reporter')
        self.token_tracker = token_tracker or TokenTracker()
        self.pricing_manager = TokenPricingManager()
    
    def generate_daily_report(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive daily usage report"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        stats = self.token_tracker.get_usage_stats(days=1)
        
        if stats["summary"]["total_requests"] == 0:
            return {
                "date": date,
                "summary": "No usage recorded for this date",
                "stats": stats
            }
        
        # Get detailed breakdown
        detailed_stats = self._get_detailed_breakdown(date)
        
        # Calculate efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(stats)
        
        # Cost analysis
        cost_analysis = self._analyze_costs(stats)
        
        return {
            "date": date,
            "summary": {
                "total_requests": stats["summary"]["total_requests"],
                "successful_requests": stats["summary"]["successful_requests"],
                "total_cost": stats["summary"]["total_cost"],
                "total_tokens": stats["summary"]["total_tokens"],
                "success_rate": stats["summary"]["success_rate"],
                "avg_processing_time": stats["summary"]["avg_processing_time"]
            },
            "breakdown": stats["breakdown"],
            "detailed_stats": detailed_stats,
            "efficiency_metrics": efficiency_metrics,
            "cost_analysis": cost_analysis,
            "projections": self.token_tracker.get_cost_projection()
        }
    
    def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate weekly usage report with trends"""
        stats = self.token_tracker.get_usage_stats(days=7)
        
        # Daily breakdown for the week
        daily_breakdown = self._get_daily_breakdown(days=7)
        
        # Trend analysis
        trends = self._analyze_trends(daily_breakdown)
        
        # Top models and phases
        top_performers = self._get_top_performers(stats)
        
        # Cost efficiency analysis
        efficiency_analysis = self._analyze_cost_efficiency(stats)
        
        return {
            "period": "Weekly (7 days)",
            "summary": stats["summary"],
            "daily_breakdown": daily_breakdown,
            "trends": trends,
            "top_performers": top_performers,
            "efficiency_analysis": efficiency_analysis,
            "breakdown": stats["breakdown"],
            "projections": self.token_tracker.get_cost_projection()
        }
    
    def generate_monthly_report(self) -> Dict[str, Any]:
        """Generate comprehensive monthly report"""
        stats = self.token_tracker.get_usage_stats(days=30)
        
        # Weekly breakdown for the month
        weekly_breakdown = self._get_weekly_breakdown()
        
        # Model performance comparison
        model_comparison = self._compare_model_performance(stats)
        
        # Phase analysis
        phase_analysis = self._analyze_phase_performance(stats)
        
        # Cost optimization recommendations
        optimization_recommendations = self._generate_optimization_recommendations(stats)
        
        return {
            "period": "Monthly (30 days)",
            "summary": stats["summary"],
            "weekly_breakdown": weekly_breakdown,
            "model_comparison": model_comparison,
            "phase_analysis": phase_analysis,
            "optimization_recommendations": optimization_recommendations,
            "breakdown": stats["breakdown"],
            "projections": self.token_tracker.get_cost_projection()
        }
    
    def generate_custom_report(self, 
                             start_date: str, 
                             end_date: str,
                             model_filter: Optional[str] = None,
                             phase_filter: Optional[str] = None) -> Dict[str, Any]:
        """Generate custom date range report with filters"""
        
        # Calculate days between dates
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        days = (end_dt - start_dt).days + 1
        
        # Get filtered stats
        stats = self.token_tracker.get_usage_stats(
            days=days, 
            model_name=model_filter,
            phase=phase_filter
        )
        
        # Custom analysis based on date range
        range_analysis = self._analyze_date_range(start_date, end_date, stats)
        
        return {
            "period": f"Custom ({start_date} to {end_date})",
            "filters": {
                "model": model_filter,
                "phase": phase_filter
            },
            "summary": stats["summary"],
            "range_analysis": range_analysis,
            "breakdown": stats["breakdown"],
            "total_days": days
        }
    
    def export_usage_csv(self, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        output_path: Optional[str] = None) -> str:
        """Export usage data to CSV format"""
        
        csv_data = self.token_tracker.export_usage_data(start_date, end_date, format="csv")
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"usage_export_{timestamp}.csv"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            f.write(csv_data)
        
        self.logger.info(f"Usage data exported to: {output_file}")
        return str(output_file)
    
    def export_usage_json(self, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         output_path: Optional[str] = None) -> str:
        """Export usage data to JSON format"""
        
        json_data = self.token_tracker.export_usage_data(start_date, end_date, format="json")
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"usage_export_{timestamp}.json"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_data)
        
        self.logger.info(f"Usage data exported to: {output_file}")
        return str(output_file)
    
    def generate_cost_breakdown_chart(self, days: int = 30, save_path: Optional[str] = None) -> str:
        """Generate cost breakdown visualization"""
        stats = self.token_tracker.get_usage_stats(days=days)
        
        if stats["summary"]["total_requests"] == 0:
            self.logger.warning("No data available for chart generation")
            return ""
        
        try:
            # Prepare data for plotting
            model_data = stats["breakdown"]["by_model"]
            models = list(model_data.keys())
            costs = [data["cost"] for data in model_data.values()]
            requests = [data["requests"] for data in model_data.values()]
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Cost breakdown pie chart
            ax1.pie(costs, labels=models, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Cost Breakdown by Model')
            
            # Request count bar chart
            ax2.bar(models, requests, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            ax2.set_title('Request Count by Model')
            ax2.set_ylabel('Number of Requests')
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save chart
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f"cost_breakdown_{timestamp}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Cost breakdown chart saved to: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate chart: {e}")
            return ""
    
    def _get_detailed_breakdown(self, date: str) -> Dict[str, Any]:
        """Get detailed breakdown for a specific date"""
        # This would query the database for more detailed information
        # For now, return placeholder structure
        return {
            "hourly_distribution": {},
            "error_breakdown": {},
            "performance_metrics": {}
        }
    
    def _calculate_efficiency_metrics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate efficiency metrics from usage stats"""
        summary = stats["summary"]
        
        if summary["total_requests"] == 0:
            return {"tokens_per_request": 0, "cost_per_token": 0, "cost_per_request": 0}
        
        tokens_per_request = summary["total_tokens"] / summary["total_requests"]
        cost_per_token = summary["total_cost"] / summary["total_tokens"] if summary["total_tokens"] > 0 else 0
        cost_per_request = summary["total_cost"] / summary["total_requests"]
        
        return {
            "tokens_per_request": round(tokens_per_request, 2),
            "cost_per_token": round(cost_per_token, 8),
            "cost_per_request": round(cost_per_request, 6),
            "success_rate": summary["success_rate"]
        }
    
    def _analyze_costs(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cost patterns and trends"""
        model_breakdown = stats["breakdown"]["by_model"]
        
        if not model_breakdown:
            return {"most_expensive": None, "most_efficient": None, "cost_distribution": {}}
        
        # Find most expensive and most efficient models
        most_expensive = max(model_breakdown.items(), key=lambda x: x[1]["cost"])
        most_efficient = min(model_breakdown.items(), key=lambda x: x[1]["cost"] / x[1]["requests"] if x[1]["requests"] > 0 else float('inf'))
        
        # Cost distribution
        total_cost = sum(data["cost"] for data in model_breakdown.values())
        cost_distribution = {
            model: {"cost": data["cost"], "percentage": (data["cost"] / total_cost * 100) if total_cost > 0 else 0}
            for model, data in model_breakdown.items()
        }
        
        return {
            "most_expensive": {"model": most_expensive[0], "cost": most_expensive[1]["cost"]},
            "most_efficient": {"model": most_efficient[0], "cost_per_request": most_efficient[1]["cost"] / most_efficient[1]["requests"] if most_efficient[1]["requests"] > 0 else 0},
            "cost_distribution": cost_distribution
        }
    
    def _get_daily_breakdown(self, days: int) -> List[Dict[str, Any]]:
        """Get daily breakdown for the specified number of days"""
        # This would query the database for daily stats
        # For now, return placeholder
        return []
    
    def _analyze_trends(self, daily_breakdown: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze usage trends from daily data"""
        if not daily_breakdown:
            return {"trend_direction": "stable", "growth_rate": 0}
        
        # Placeholder trend analysis
        return {
            "trend_direction": "stable",
            "growth_rate": 0,
            "peak_usage_day": "",
            "lowest_usage_day": ""
        }
    
    def _get_top_performers(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Identify top performing models and phases"""
        model_breakdown = stats["breakdown"]["by_model"]
        phase_breakdown = stats["breakdown"]["by_phase"]
        
        if not model_breakdown:
            return {"top_model": None, "top_phase": None}
        
        # Top model by usage
        top_model = max(model_breakdown.items(), key=lambda x: x[1]["requests"])
        
        # Top phase by cost
        top_phase = None
        if phase_breakdown:
            top_phase = max(phase_breakdown.items(), key=lambda x: x[1]["cost"])
        
        return {
            "top_model": {"name": top_model[0], "requests": top_model[1]["requests"]},
            "top_phase": {"name": top_phase[0], "cost": top_phase[1]["cost"]} if top_phase else None
        }
    
    def _analyze_cost_efficiency(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cost efficiency patterns"""
        model_breakdown = stats["breakdown"]["by_model"]
        
        if not model_breakdown:
            return {"efficiency_ranking": []}
        
        # Calculate efficiency score for each model
        efficiency_ranking = []
        for model, data in model_breakdown.items():
            if data["requests"] > 0:
                cost_per_request = data["cost"] / data["requests"]
                success_rate = data["success_rate"]
                efficiency_score = success_rate / cost_per_request if cost_per_request > 0 else 0
                
                efficiency_ranking.append({
                    "model": model,
                    "efficiency_score": efficiency_score,
                    "cost_per_request": cost_per_request,
                    "success_rate": success_rate
                })
        
        # Sort by efficiency score
        efficiency_ranking.sort(key=lambda x: x["efficiency_score"], reverse=True)
        
        return {
            "efficiency_ranking": efficiency_ranking[:5]  # Top 5
        }
    
    def _get_weekly_breakdown(self) -> List[Dict[str, Any]]:
        """Get weekly breakdown for the past month"""
        # Placeholder for weekly data
        return []
    
    def _compare_model_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance across different models"""
        model_breakdown = stats["breakdown"]["by_model"]
        
        comparison = {}
        for model, data in model_breakdown.items():
            comparison[model] = {
                "requests": data["requests"],
                "tokens": data["tokens"],
                "cost": data["cost"],
                "avg_time": data["avg_processing_time"],
                "success_rate": data["success_rate"],
                "cost_efficiency": data["cost"] / data["requests"] if data["requests"] > 0 else 0
            }
        
        return comparison
    
    def _analyze_phase_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance by phase"""
        phase_breakdown = stats["breakdown"]["by_phase"]
        
        if not phase_breakdown:
            return {"phase_efficiency": {}, "recommendations": []}
        
        phase_efficiency = {}
        for phase, data in phase_breakdown.items():
            phase_efficiency[phase] = {
                "requests": data["requests"],
                "cost": data["cost"],
                "cost_per_request": data["cost"] / data["requests"] if data["requests"] > 0 else 0
            }
        
        return {
            "phase_efficiency": phase_efficiency,
            "recommendations": self._generate_phase_recommendations(phase_efficiency)
        }
    
    def _generate_optimization_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        model_breakdown = stats["breakdown"]["by_model"]
        
        if not model_breakdown:
            return recommendations
        
        # Analyze model costs
        for model, data in model_breakdown.items():
            cost_per_request = data["cost"] / data["requests"] if data["requests"] > 0 else 0
            
            if cost_per_request > 0.01:  # Threshold for expensive requests
                recommendations.append(f"Consider optimizing {model} usage - high cost per request: ${cost_per_request:.6f}")
            
            if data["success_rate"] < 0.9:  # Low success rate
                recommendations.append(f"Investigate {model} reliability - success rate: {data['success_rate']:.1%}")
        
        return recommendations
    
    def _generate_phase_recommendations(self, phase_efficiency: Dict[str, Any]) -> List[str]:
        """Generate phase-specific recommendations"""
        recommendations = []
        
        for phase, data in phase_efficiency.items():
            if data["cost_per_request"] > 0.005:  # Threshold
                recommendations.append(f"Phase {phase} has high cost per request: ${data['cost_per_request']:.6f}")
        
        return recommendations
    
    def _analyze_date_range(self, start_date: str, end_date: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze specific date range"""
        return {
            "date_range": f"{start_date} to {end_date}",
            "total_days": (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days + 1,
            "daily_average": {
                "requests": stats["summary"]["total_requests"] / max(1, (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days + 1),
                "cost": stats["summary"]["total_cost"] / max(1, (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days + 1)
            }
        }

class ConsoleReporter:
    """Console-based reporting for real-time usage display"""
    
    def __init__(self, usage_reporter: UsageReporter):
        self.usage_reporter = usage_reporter
    
    def display_current_usage(self):
        """Display current usage in console"""
        stats = self.usage_reporter.token_tracker.get_usage_stats(days=1)
        
        print("\n" + "=" * 80)
        print("📊 CURRENT USAGE SUMMARY")
        print("=" * 80)
        
        if stats["summary"]["total_requests"] == 0:
            print("No usage recorded today.")
            return
        
        summary = stats["summary"]
        print(f"🔢 Total Requests: {summary['total_requests']}")
        print(f"✅ Successful: {summary['successful_requests']} ({summary['success_rate']:.1%})")
        print(f"🎯 Total Tokens: {summary['total_tokens']:,}")
        print(f"💰 Total Cost: ${summary['total_cost']:.6f}")
        print(f"⏱️  Avg Time: {summary['avg_processing_time']:.2f}s")
        
        # Model breakdown
        if stats["breakdown"]["by_model"]:
            print(f"\n📋 By Model:")
            for model, data in stats["breakdown"]["by_model"].items():
                print(f"  • {model:30} | {data['requests']:>3} reqs | ${data['cost']:>8.6f} | {data['success_rate']:>5.1%}")
        
        # Phase breakdown
        if stats["breakdown"]["by_phase"]:
            print(f"\n🔄 By Phase:")
            for phase, data in stats["breakdown"]["by_phase"].items():
                print(f"  • {phase:15} | {data['requests']:>3} reqs | ${data['cost']:>8.6f}")
        
        # Projections
        projections = self.usage_reporter.token_tracker.get_cost_projection()
        if projections:
            print(f"\n📈 Cost Projections:")
            print(f"  • Daily: ${projections['daily_projection']:.6f}")
            print(f"  • Monthly: ${projections['monthly_projection']:.4f}")
            print(f"  • Yearly: ${projections['yearly_projection']:.2f}")
        
        print("=" * 80 + "\n")
    
    def display_model_pricing(self):
        """Display current model pricing information"""
        pricing = TokenPricingManager.get_all_pricing()
        
        print("\n" + "=" * 80)
        print("💰 CURRENT MODEL PRICING (per 1K tokens)")
        print("=" * 80)
        
        for model, prices in pricing.items():
            print(f"{model:35} | Input: ${prices['input']:>8.6f} | Output: ${prices['output']:>8.6f}")
        
        print("=" * 80 + "\n")
    
    def display_quick_stats(self, days: int = 7):
        """Display quick statistics for specified days"""
        stats = self.usage_reporter.token_tracker.get_usage_stats(days=days)
        
        if stats["summary"]["total_requests"] == 0:
            print(f"No usage recorded in the last {days} days.")
            return
        
        summary = stats["summary"]
        print(f"\n📊 Last {days} days: {summary['total_requests']} requests, "
              f"${summary['total_cost']:.4f} total, "
              f"{summary['success_rate']:.1%} success rate")