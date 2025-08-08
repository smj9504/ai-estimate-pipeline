# src/tracking/token_tracker.py
import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from contextlib import contextmanager

from src.utils.logger import get_logger

@dataclass
class TokenUsage:
    """Token usage data structure"""
    timestamp: str
    model_name: str
    api_provider: str  # openai, anthropic, google
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    phase: str  # phase0, phase1, phase2, etc.
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

class TokenPricingManager:
    """Manages API pricing for different models"""
    
    # Current pricing as of 2024 (per 1K tokens)
    PRICING_TABLE = {
        # OpenAI GPT-4 models
        'gpt-4o': {
            'input': 0.0025,   # $2.50 per 1M input tokens
            'output': 0.01     # $10.00 per 1M output tokens
        },
        'gpt-4o-mini': {
            'input': 0.000150,  # $0.150 per 1M input tokens
            'output': 0.0006    # $0.600 per 1M output tokens
        },
        'gpt-4-turbo-preview': {
            'input': 0.01,      # $10.00 per 1M input tokens
            'output': 0.03      # $30.00 per 1M output tokens
        },
        
        # Anthropic Claude models
        'claude-3-5-sonnet-20241022': {
            'input': 0.003,     # $3.00 per 1M input tokens
            'output': 0.015     # $15.00 per 1M output tokens
        },
        'claude-3-sonnet-20240229': {
            'input': 0.003,     # $3.00 per 1M input tokens
            'output': 0.015     # $15.00 per 1M output tokens
        },
        'claude-3-haiku-20240307': {
            'input': 0.00025,   # $0.25 per 1M input tokens
            'output': 0.00125   # $1.25 per 1M output tokens
        },
        
        # Google Gemini models
        'gemini-1.5-pro': {
            'input': 0.0035,    # $3.50 per 1M input tokens (up to 128K)
            'output': 0.0105    # $10.50 per 1M output tokens
        },
        'gemini-1.5-flash': {
            'input': 0.000075,  # $0.075 per 1M input tokens (up to 128K)
            'output': 0.0003    # $0.30 per 1M output tokens
        }
    }
    
    @classmethod
    def calculate_cost(cls, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for given token usage"""
        pricing = cls.PRICING_TABLE.get(model_name)
        if not pricing:
            # Fallback to average pricing if model not found
            return (prompt_tokens * 0.001 + completion_tokens * 0.002) / 1000
        
        input_cost = (prompt_tokens / 1000) * pricing['input']
        output_cost = (completion_tokens / 1000) * pricing['output']
        return input_cost + output_cost
    
    @classmethod
    def get_model_pricing(cls, model_name: str) -> Optional[Dict[str, float]]:
        """Get pricing info for a specific model"""
        return cls.PRICING_TABLE.get(model_name)
    
    @classmethod
    def get_all_pricing(cls) -> Dict[str, Dict[str, float]]:
        """Get all pricing information"""
        return cls.PRICING_TABLE.copy()

class TokenTracker:
    """Comprehensive token usage and cost tracking system"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.logger = get_logger('token_tracker')
        
        # Setup database path
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "token_usage.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Threading lock for database operations
        self._db_lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Pricing manager
        self.pricing_manager = TokenPricingManager()
        
        self.logger.info(f"TokenTracker initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Main usage table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    api_provider TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    estimated_cost REAL NOT NULL,
                    phase TEXT,
                    session_id TEXT,
                    request_id TEXT,
                    processing_time REAL DEFAULT 0.0,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Summary table for quick aggregations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS usage_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    api_provider TEXT NOT NULL,
                    total_requests INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    success_rate REAL DEFAULT 0.0,
                    avg_processing_time REAL DEFAULT 0.0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, model_name, api_provider)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON token_usage(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_name ON token_usage(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_provider ON token_usage(api_provider)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_phase ON token_usage(phase)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON token_usage(session_id)')
            
            conn.commit()
            self.logger.info("Database tables initialized successfully")
    
    @contextmanager
    def _get_db_connection(self):
        """Thread-safe database connection context manager"""
        with self._db_lock:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            try:
                yield conn
            finally:
                conn.close()
    
    def track_usage(self, 
                   model_name: str,
                   prompt_tokens: int,
                   completion_tokens: int,
                   processing_time: float = 0.0,
                   phase: str = "unknown",
                   session_id: Optional[str] = None,
                   request_id: Optional[str] = None,
                   success: bool = True,
                   error_message: Optional[str] = None) -> TokenUsage:
        """Track token usage for a model call"""
        
        # Determine API provider from model name
        api_provider = self._get_api_provider(model_name)
        
        # Calculate cost
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost = self.pricing_manager.calculate_cost(
            model_name, prompt_tokens, completion_tokens
        )
        
        # Create usage record
        usage = TokenUsage(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            api_provider=api_provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            phase=phase,
            session_id=session_id,
            request_id=request_id,
            processing_time=processing_time,
            success=success,
            error_message=error_message
        )
        
        # Save to database
        self._save_usage(usage)
        
        # Update summary
        self._update_summary(usage)
        
        # Log usage
        self.logger.info(
            f"📊 Token usage tracked: {model_name} | "
            f"{total_tokens} tokens | ${estimated_cost:.6f} | "
            f"{phase} | {processing_time:.2f}s"
        )
        
        return usage
    
    def _get_api_provider(self, model_name: str) -> str:
        """Determine API provider from model name"""
        model_name_lower = model_name.lower()
        
        if 'gpt' in model_name_lower or 'davinci' in model_name_lower:
            return 'openai'
        elif 'claude' in model_name_lower:
            return 'anthropic'
        elif 'gemini' in model_name_lower or 'palm' in model_name_lower:
            return 'google'
        else:
            return 'unknown'
    
    def _save_usage(self, usage: TokenUsage):
        """Save usage record to database"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO token_usage (
                    timestamp, model_name, api_provider, prompt_tokens,
                    completion_tokens, total_tokens, estimated_cost,
                    phase, session_id, request_id, processing_time,
                    success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                usage.timestamp, usage.model_name, usage.api_provider,
                usage.prompt_tokens, usage.completion_tokens, usage.total_tokens,
                usage.estimated_cost, usage.phase, usage.session_id,
                usage.request_id, usage.processing_time, usage.success,
                usage.error_message
            ))
            conn.commit()
    
    def _update_summary(self, usage: TokenUsage):
        """Update daily summary statistics"""
        date = usage.timestamp.split('T')[0]  # Get date part
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get current summary
            cursor.execute('''
                SELECT total_requests, total_tokens, total_cost, success_rate, avg_processing_time
                FROM usage_summary 
                WHERE date = ? AND model_name = ? AND api_provider = ?
            ''', (date, usage.model_name, usage.api_provider))
            
            row = cursor.fetchone()
            
            if row:
                # Update existing summary
                total_requests, total_tokens, total_cost, success_rate, avg_processing_time = row
                new_requests = total_requests + 1
                new_tokens = total_tokens + usage.total_tokens
                new_cost = total_cost + usage.estimated_cost
                new_success_rate = ((success_rate * total_requests) + (1 if usage.success else 0)) / new_requests
                new_avg_time = ((avg_processing_time * total_requests) + usage.processing_time) / new_requests
                
                cursor.execute('''
                    UPDATE usage_summary 
                    SET total_requests = ?, total_tokens = ?, total_cost = ?,
                        success_rate = ?, avg_processing_time = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE date = ? AND model_name = ? AND api_provider = ?
                ''', (new_requests, new_tokens, new_cost, new_success_rate, 
                     new_avg_time, date, usage.model_name, usage.api_provider))
            else:
                # Create new summary
                cursor.execute('''
                    INSERT INTO usage_summary (
                        date, model_name, api_provider, total_requests, total_tokens,
                        total_cost, success_rate, avg_processing_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (date, usage.model_name, usage.api_provider, 1,
                     usage.total_tokens, usage.estimated_cost, 
                     1.0 if usage.success else 0.0, usage.processing_time))
            
            conn.commit()
    
    def get_usage_stats(self, 
                       days: int = 7,
                       model_name: Optional[str] = None,
                       api_provider: Optional[str] = None,
                       phase: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build WHERE clause
            where_conditions = ["timestamp >= ?"]
            params = [start_date.isoformat()]
            
            if model_name:
                where_conditions.append("model_name = ?")
                params.append(model_name)
            if api_provider:
                where_conditions.append("api_provider = ?")
                params.append(api_provider)
            if phase:
                where_conditions.append("phase = ?")
                params.append(phase)
            
            where_clause = " AND ".join(where_conditions)
            
            # Get basic stats
            cursor.execute(f'''
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(total_tokens) as total_tokens,
                    SUM(estimated_cost) as total_cost,
                    AVG(processing_time) as avg_processing_time,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_requests,
                    MIN(timestamp) as first_request,
                    MAX(timestamp) as last_request
                FROM token_usage
                WHERE {where_clause}
            ''', params)
            
            basic_stats = cursor.fetchone()
            
            if not basic_stats or basic_stats[0] == 0:
                return {
                    "period": {"start": start_date.isoformat(), "end": end_date.isoformat(), "days": days},
                    "summary": {"total_requests": 0, "total_tokens": 0, "total_cost": 0.0},
                    "breakdown": {}
                }
            
            total_requests, total_tokens, total_cost, avg_time, successful_requests, first_req, last_req = basic_stats
            success_rate = (successful_requests / total_requests) if total_requests > 0 else 0
            
            # Get breakdown by model
            cursor.execute(f'''
                SELECT 
                    model_name,
                    api_provider,
                    COUNT(*) as requests,
                    SUM(total_tokens) as tokens,
                    SUM(estimated_cost) as cost,
                    AVG(processing_time) as avg_time,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
                FROM token_usage
                WHERE {where_clause}
                GROUP BY model_name, api_provider
                ORDER BY cost DESC
            ''', params)
            
            model_breakdown = {}
            for row in cursor.fetchall():
                model, provider, requests, tokens, cost, avg_time_model, successful = row
                success_rate_model = (successful / requests) if requests > 0 else 0
                model_breakdown[f"{provider}:{model}"] = {
                    "requests": requests,
                    "tokens": tokens,
                    "cost": round(cost, 6),
                    "avg_processing_time": round(avg_time_model or 0, 3),
                    "success_rate": round(success_rate_model, 3)
                }
            
            # Get breakdown by phase
            cursor.execute(f'''
                SELECT 
                    phase,
                    COUNT(*) as requests,
                    SUM(total_tokens) as tokens,
                    SUM(estimated_cost) as cost
                FROM token_usage
                WHERE {where_clause}
                GROUP BY phase
                ORDER BY cost DESC
            ''', params)
            
            phase_breakdown = {}
            for row in cursor.fetchall():
                phase_name, requests, tokens, cost = row
                phase_breakdown[phase_name or "unknown"] = {
                    "requests": requests,
                    "tokens": tokens,
                    "cost": round(cost, 6)
                }
            
            return {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": days,
                    "first_request": first_req,
                    "last_request": last_req
                },
                "summary": {
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "total_tokens": total_tokens,
                    "total_cost": round(total_cost or 0, 6),
                    "avg_processing_time": round(avg_time or 0, 3),
                    "success_rate": round(success_rate, 3)
                },
                "breakdown": {
                    "by_model": model_breakdown,
                    "by_phase": phase_breakdown
                }
            }
    
    def get_recent_usage(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent usage records"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM token_usage 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_cost_projection(self, days: int = 30) -> Dict[str, float]:
        """Project costs based on recent usage patterns"""
        stats = self.get_usage_stats(days=7)  # Use last week for projection
        
        if stats["summary"]["total_requests"] == 0:
            return {"daily_projection": 0.0, "monthly_projection": 0.0, "yearly_projection": 0.0}
        
        daily_avg_cost = stats["summary"]["total_cost"] / 7
        
        return {
            "daily_projection": round(daily_avg_cost, 6),
            "weekly_projection": round(daily_avg_cost * 7, 6),
            "monthly_projection": round(daily_avg_cost * days, 6),
            "yearly_projection": round(daily_avg_cost * 365, 2)
        }
    
    def export_usage_data(self, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         format: str = "json") -> str:
        """Export usage data to JSON or CSV format"""
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build query
            where_conditions = []
            params = []
            
            if start_date:
                where_conditions.append("timestamp >= ?")
                params.append(start_date)
            if end_date:
                where_conditions.append("timestamp <= ?")
                params.append(end_date)
            
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            cursor.execute(f'''
                SELECT * FROM token_usage 
                {where_clause}
                ORDER BY timestamp DESC
            ''', params)
            
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            if format.lower() == "json":
                data = [dict(zip(columns, row)) for row in rows]
                return json.dumps(data, indent=2)
            elif format.lower() == "csv":
                import csv
                import io
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(columns)
                writer.writerows(rows)
                return output.getvalue()
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old usage data"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Delete old usage records
            cursor.execute('DELETE FROM token_usage WHERE timestamp < ?', (cutoff_date.isoformat(),))
            usage_deleted = cursor.rowcount
            
            # Delete old summary records
            cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
            cursor.execute('DELETE FROM usage_summary WHERE date < ?', (cutoff_date_str,))
            summary_deleted = cursor.rowcount
            
            conn.commit()
            
            self.logger.info(f"Cleaned up {usage_deleted} usage records and {summary_deleted} summary records older than {days_to_keep} days")
    
    def get_database_size(self) -> Dict[str, Any]:
        """Get database size and record counts"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get record counts
            cursor.execute('SELECT COUNT(*) FROM token_usage')
            usage_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM usage_summary')
            summary_count = cursor.fetchone()[0]
            
            # Get database file size
            db_size_bytes = self.db_path.stat().st_size
            db_size_mb = round(db_size_bytes / (1024 * 1024), 2)
            
            return {
                "database_path": str(self.db_path),
                "size_bytes": db_size_bytes,
                "size_mb": db_size_mb,
                "usage_records": usage_count,
                "summary_records": summary_count,
                "total_records": usage_count + summary_count
            }