# src/api/tracking_endpoints.py
from fastapi import APIRouter, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import tempfile
import os

from src.tracking.token_tracker import TokenTracker, TokenPricingManager
from src.tracking.usage_reporter import UsageReporter, ConsoleReporter
from src.utils.logger import get_logger

# Create router for tracking endpoints
tracking_router = APIRouter(prefix="/api/tracking", tags=["Token Tracking"])

# Initialize components
logger = get_logger('tracking_endpoints')
token_tracker = None
usage_reporter = None

def get_token_tracker():
    """Get or create token tracker instance"""
    global token_tracker, usage_reporter
    if token_tracker is None:
        try:
            token_tracker = TokenTracker()
            usage_reporter = UsageReporter(token_tracker)
            logger.info("Token tracking components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize token tracker: {e}")
            raise HTTPException(status_code=500, detail="Token tracking service unavailable")
    return token_tracker, usage_reporter

@tracking_router.get("/health")
async def tracking_health():
    """Check tracking system health"""
    try:
        tracker, _ = get_token_tracker()
        db_info = tracker.get_database_size()
        
        return {
            "status": "healthy",
            "database": {
                "size_mb": db_info["size_mb"],
                "total_records": db_info["total_records"],
                "usage_records": db_info["usage_records"]
            },
            "pricing_models": len(TokenPricingManager.get_all_pricing())
        }
    except Exception as e:
        logger.error(f"Tracking health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@tracking_router.get("/stats")
async def get_usage_stats(
    days: int = Query(7, ge=1, le=365, description="Number of days to analyze"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    api_provider: Optional[str] = Query(None, description="Filter by API provider"),
    phase: Optional[str] = Query(None, description="Filter by phase")
):
    """Get comprehensive usage statistics"""
    try:
        tracker, _ = get_token_tracker()
        
        stats = tracker.get_usage_stats(
            days=days,
            model_name=model_name,
            api_provider=api_provider,
            phase=phase
        )
        
        return {
            "success": True,
            "data": stats,
            "filters": {
                "days": days,
                "model_name": model_name,
                "api_provider": api_provider,
                "phase": phase
            }
        }
    except Exception as e:
        logger.error(f"Failed to get usage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@tracking_router.get("/recent")
async def get_recent_usage(
    limit: int = Query(100, ge=1, le=1000, description="Number of recent records")
):
    """Get recent usage records"""
    try:
        tracker, _ = get_token_tracker()
        recent_records = tracker.get_recent_usage(limit=limit)
        
        return {
            "success": True,
            "data": recent_records,
            "count": len(recent_records)
        }
    except Exception as e:
        logger.error(f"Failed to get recent usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@tracking_router.get("/projections")
async def get_cost_projections():
    """Get cost projections based on recent usage"""
    try:
        tracker, _ = get_token_tracker()
        projections = tracker.get_cost_projection()
        
        if not projections:
            return {
                "success": True,
                "message": "No recent usage data available for projections",
                "data": {
                    "daily_projection": 0.0,
                    "weekly_projection": 0.0,
                    "monthly_projection": 0.0,
                    "yearly_projection": 0.0
                }
            }
        
        return {
            "success": True,
            "data": projections
        }
    except Exception as e:
        logger.error(f"Failed to get cost projections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@tracking_router.get("/pricing")
async def get_model_pricing():
    """Get current model pricing information"""
    try:
        pricing = TokenPricingManager.get_all_pricing()
        
        return {
            "success": True,
            "data": {
                "pricing_table": pricing,
                "last_updated": "2024-12-01",  # Update this when pricing changes
                "currency": "USD",
                "unit": "per 1K tokens"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get pricing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@tracking_router.get("/reports/daily")
async def get_daily_report(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format")
):
    """Generate daily usage report"""
    try:
        _, reporter = get_token_tracker()
        
        if date:
            try:
                datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        report = reporter.generate_daily_report(date)
        
        return {
            "success": True,
            "report_type": "daily",
            "data": report
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate daily report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@tracking_router.get("/reports/weekly")
async def get_weekly_report():
    """Generate weekly usage report"""
    try:
        _, reporter = get_token_tracker()
        report = reporter.generate_weekly_report()
        
        return {
            "success": True,
            "report_type": "weekly",
            "data": report
        }
    except Exception as e:
        logger.error(f"Failed to generate weekly report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@tracking_router.get("/reports/monthly")
async def get_monthly_report():
    """Generate monthly usage report"""
    try:
        _, reporter = get_token_tracker()
        report = reporter.generate_monthly_report()
        
        return {
            "success": True,
            "report_type": "monthly",
            "data": report
        }
    except Exception as e:
        logger.error(f"Failed to generate monthly report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@tracking_router.get("/reports/custom")
async def get_custom_report(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    model_filter: Optional[str] = Query(None, description="Filter by model name"),
    phase_filter: Optional[str] = Query(None, description="Filter by phase")
):
    """Generate custom date range report"""
    try:
        # Validate dates
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            if start_dt > end_dt:
                raise HTTPException(status_code=400, detail="Start date must be before end date")
            
            if (end_dt - start_dt).days > 365:
                raise HTTPException(status_code=400, detail="Date range cannot exceed 365 days")
                
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        _, reporter = get_token_tracker()
        report = reporter.generate_custom_report(start_date, end_date, model_filter, phase_filter)
        
        return {
            "success": True,
            "report_type": "custom",
            "data": report
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate custom report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@tracking_router.get("/export/csv")
async def export_usage_csv(
    background_tasks: BackgroundTasks,
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format")
):
    """Export usage data as CSV file"""
    try:
        # Validate dates if provided
        if start_date:
            try:
                datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")
        
        if end_date:
            try:
                datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")
        
        _, reporter = get_token_tracker()
        
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"usage_export_{timestamp}.csv"
        temp_path = os.path.join(temp_dir, filename)
        
        # Export data
        csv_path = reporter.export_usage_csv(start_date, end_date, temp_path)
        
        # Clean up temp file after response
        background_tasks.add_task(os.remove, csv_path)
        
        return FileResponse(
            path=csv_path,
            media_type='text/csv',
            filename=filename
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@tracking_router.get("/export/json")
async def export_usage_json(
    background_tasks: BackgroundTasks,
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format")
):
    """Export usage data as JSON file"""
    try:
        # Validate dates if provided
        if start_date:
            try:
                datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")
        
        if end_date:
            try:
                datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")
        
        _, reporter = get_token_tracker()
        
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"usage_export_{timestamp}.json"
        temp_path = os.path.join(temp_dir, filename)
        
        # Export data
        json_path = reporter.export_usage_json(start_date, end_date, temp_path)
        
        # Clean up temp file after response
        background_tasks.add_task(os.remove, json_path)
        
        return FileResponse(
            path=json_path,
            media_type='application/json',
            filename=filename
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export JSON: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@tracking_router.post("/cleanup")
async def cleanup_old_data(
    days_to_keep: int = Query(90, ge=7, le=1000, description="Number of days of data to keep")
):
    """Clean up old usage data"""
    try:
        tracker, _ = get_token_tracker()
        
        # Get current record count
        db_info_before = tracker.get_database_size()
        
        # Perform cleanup
        tracker.cleanup_old_data(days_to_keep)
        
        # Get new record count
        db_info_after = tracker.get_database_size()
        
        records_deleted = db_info_before["total_records"] - db_info_after["total_records"]
        
        return {
            "success": True,
            "message": f"Cleanup completed. Removed data older than {days_to_keep} days.",
            "records_deleted": records_deleted,
            "database_size": {
                "before_mb": db_info_before["size_mb"],
                "after_mb": db_info_after["size_mb"],
                "space_saved_mb": round(db_info_before["size_mb"] - db_info_after["size_mb"], 2)
            }
        }
    except Exception as e:
        logger.error(f"Failed to cleanup data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@tracking_router.get("/database/info")
async def get_database_info():
    """Get database size and statistics"""
    try:
        tracker, _ = get_token_tracker()
        db_info = tracker.get_database_size()
        
        return {
            "success": True,
            "data": db_info
        }
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@tracking_router.get("/models/efficiency")
async def get_model_efficiency():
    """Get model efficiency analysis"""
    try:
        tracker, reporter = get_token_tracker()
        
        # Get recent stats for efficiency analysis
        stats = tracker.get_usage_stats(days=30)
        
        if stats["summary"]["total_requests"] == 0:
            return {
                "success": True,
                "message": "No usage data available for efficiency analysis",
                "data": {"efficiency_ranking": []}
            }
        
        # Calculate efficiency metrics
        efficiency_data = reporter._analyze_cost_efficiency(stats)
        
        return {
            "success": True,
            "data": efficiency_data
        }
    except Exception as e:
        logger.error(f"Failed to get model efficiency: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@tracking_router.get("/dashboard/summary")
async def get_dashboard_summary():
    """Get summary data for usage dashboard"""
    try:
        tracker, reporter = get_token_tracker()
        
        # Get various time periods
        today_stats = tracker.get_usage_stats(days=1)
        week_stats = tracker.get_usage_stats(days=7)
        month_stats = tracker.get_usage_stats(days=30)
        
        # Get projections
        projections = tracker.get_cost_projection()
        
        # Get recent usage
        recent_usage = tracker.get_recent_usage(limit=10)
        
        # Get database info
        db_info = tracker.get_database_size()
        
        return {
            "success": True,
            "data": {
                "timeframes": {
                    "today": today_stats["summary"],
                    "week": week_stats["summary"],
                    "month": month_stats["summary"]
                },
                "projections": projections,
                "recent_usage": recent_usage,
                "database_info": db_info,
                "model_breakdown": week_stats["breakdown"]["by_model"],
                "phase_breakdown": week_stats["breakdown"]["by_phase"]
            }
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))