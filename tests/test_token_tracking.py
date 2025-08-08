# tests/test_token_tracking.py
import pytest
import tempfile
import sqlite3
from pathlib import Path
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Import tracking components
from src.tracking.token_tracker import TokenTracker, TokenPricingManager, TokenUsage
from src.tracking.usage_reporter import UsageReporter, ConsoleReporter
from src.tracking.response_parser import TokenResponseParser
from src.tracking.tracked_orchestrator import TrackedModelOrchestrator

class TestTokenPricingManager:
    """Test token pricing calculations"""
    
    def test_calculate_cost_gpt4o_mini(self):
        """Test cost calculation for GPT-4o mini"""
        cost = TokenPricingManager.calculate_cost("gpt-4o-mini", 1000, 500)
        expected = (1000/1000 * 0.000150) + (500/1000 * 0.0006)
        assert cost == pytest.approx(expected, rel=1e-6)
    
    def test_calculate_cost_claude_sonnet(self):
        """Test cost calculation for Claude Sonnet"""
        cost = TokenPricingManager.calculate_cost("claude-3-5-sonnet-20241022", 1000, 500)
        expected = (1000/1000 * 0.003) + (500/1000 * 0.015)
        assert cost == pytest.approx(expected, rel=1e-6)
    
    def test_calculate_cost_gemini_flash(self):
        """Test cost calculation for Gemini Flash"""
        cost = TokenPricingManager.calculate_cost("gemini-1.5-flash", 1000, 500)
        expected = (1000/1000 * 0.000075) + (500/1000 * 0.0003)
        assert cost == pytest.approx(expected, rel=1e-6)
    
    def test_calculate_cost_unknown_model(self):
        """Test fallback pricing for unknown models"""
        cost = TokenPricingManager.calculate_cost("unknown-model", 1000, 500)
        expected = (1000 * 0.001 + 500 * 0.002) / 1000
        assert cost == pytest.approx(expected, rel=1e-6)
    
    def test_get_model_pricing(self):
        """Test getting pricing for specific model"""
        pricing = TokenPricingManager.get_model_pricing("gpt-4o-mini")
        assert pricing is not None
        assert "input" in pricing
        assert "output" in pricing
        assert pricing["input"] == 0.000150
    
    def test_get_all_pricing(self):
        """Test getting all pricing information"""
        all_pricing = TokenPricingManager.get_all_pricing()
        assert isinstance(all_pricing, dict)
        assert len(all_pricing) > 0
        assert "gpt-4o-mini" in all_pricing

class TestTokenTracker:
    """Test token tracking functionality"""
    
    @pytest.fixture
    def temp_tracker(self):
        """Create a temporary token tracker for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_tokens.db"
            tracker = TokenTracker(str(db_path))
            yield tracker
    
    def test_init_database(self, temp_tracker):
        """Test database initialization"""
        # Check if tables exist
        with temp_tracker._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check token_usage table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='token_usage'")
            assert cursor.fetchone() is not None
            
            # Check usage_summary table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usage_summary'")
            assert cursor.fetchone() is not None
    
    def test_track_usage(self, temp_tracker):
        """Test tracking token usage"""
        usage = temp_tracker.track_usage(
            model_name="gpt-4o-mini",
            prompt_tokens=1000,
            completion_tokens=500,
            processing_time=2.5,
            phase="test_phase",
            success=True
        )
        
        assert isinstance(usage, TokenUsage)
        assert usage.model_name == "gpt-4o-mini"
        assert usage.prompt_tokens == 1000
        assert usage.completion_tokens == 500
        assert usage.total_tokens == 1500
        assert usage.processing_time == 2.5
        assert usage.phase == "test_phase"
        assert usage.success == True
        assert usage.estimated_cost > 0
    
    def test_track_failed_usage(self, temp_tracker):
        """Test tracking failed API calls"""
        usage = temp_tracker.track_usage(
            model_name="gpt-4o-mini",
            prompt_tokens=1000,
            completion_tokens=0,
            processing_time=1.0,
            phase="test_phase",
            success=False,
            error_message="API timeout"
        )
        
        assert usage.success == False
        assert usage.error_message == "API timeout"
        assert usage.completion_tokens == 0
    
    def test_get_usage_stats(self, temp_tracker):
        """Test getting usage statistics"""
        # Track some usage first
        temp_tracker.track_usage("gpt-4o-mini", 1000, 500, 2.0, "phase1", success=True)
        temp_tracker.track_usage("claude-3-5-sonnet-20241022", 800, 400, 1.8, "phase1", success=True)
        temp_tracker.track_usage("gemini-1.5-flash", 1200, 600, 1.5, "phase2", success=False)
        
        stats = temp_tracker.get_usage_stats(days=1)
        
        assert stats["summary"]["total_requests"] == 3
        assert stats["summary"]["successful_requests"] == 2
        assert stats["summary"]["success_rate"] == pytest.approx(2/3, rel=1e-2)
        assert len(stats["breakdown"]["by_model"]) == 3
        assert len(stats["breakdown"]["by_phase"]) == 2
    
    def test_get_usage_stats_with_filters(self, temp_tracker):
        """Test filtered usage statistics"""
        temp_tracker.track_usage("gpt-4o-mini", 1000, 500, 2.0, "phase1")
        temp_tracker.track_usage("gpt-4o-mini", 800, 400, 1.8, "phase2")
        temp_tracker.track_usage("claude-3-5-sonnet-20241022", 1200, 600, 1.5, "phase1")
        
        # Filter by model
        stats = temp_tracker.get_usage_stats(days=1, model_name="gpt-4o-mini")
        assert stats["summary"]["total_requests"] == 2
        
        # Filter by phase
        stats = temp_tracker.get_usage_stats(days=1, phase="phase1")
        assert stats["summary"]["total_requests"] == 2
    
    def test_get_recent_usage(self, temp_tracker):
        """Test getting recent usage records"""
        temp_tracker.track_usage("gpt-4o-mini", 1000, 500, 2.0, "phase1")
        temp_tracker.track_usage("claude-3-5-sonnet-20241022", 800, 400, 1.8, "phase2")
        
        recent = temp_tracker.get_recent_usage(limit=5)
        assert len(recent) == 2
        assert recent[0]["model_name"] in ["gpt-4o-mini", "claude-3-5-sonnet-20241022"]
    
    def test_cost_projection(self, temp_tracker):
        """Test cost projection calculations"""
        # Add some usage data
        for i in range(7):
            temp_tracker.track_usage("gpt-4o-mini", 1000, 500, 2.0, "phase1")
        
        projections = temp_tracker.get_cost_projection()
        
        assert "daily_projection" in projections
        assert "monthly_projection" in projections
        assert "yearly_projection" in projections
        assert all(isinstance(v, float) and v >= 0 for v in projections.values())
    
    def test_export_usage_data(self, temp_tracker):
        """Test exporting usage data"""
        temp_tracker.track_usage("gpt-4o-mini", 1000, 500, 2.0, "phase1")
        
        # Test JSON export
        json_data = temp_tracker.export_usage_data(format="json")
        parsed_data = json.loads(json_data)
        assert isinstance(parsed_data, list)
        assert len(parsed_data) == 1
        
        # Test CSV export
        csv_data = temp_tracker.export_usage_data(format="csv")
        assert "model_name" in csv_data
        assert "gpt-4o-mini" in csv_data
    
    def test_cleanup_old_data(self, temp_tracker):
        """Test cleanup of old data"""
        # Add some data
        temp_tracker.track_usage("gpt-4o-mini", 1000, 500, 2.0, "phase1")
        
        # Cleanup (should not delete recent data)
        temp_tracker.cleanup_old_data(days_to_keep=30)
        
        recent = temp_tracker.get_recent_usage(limit=10)
        assert len(recent) == 1  # Data should still be there
    
    def test_get_database_size(self, temp_tracker):
        """Test database size information"""
        temp_tracker.track_usage("gpt-4o-mini", 1000, 500, 2.0, "phase1")
        
        db_info = temp_tracker.get_database_size()
        
        assert "database_path" in db_info
        assert "size_bytes" in db_info
        assert "size_mb" in db_info
        assert "usage_records" in db_info
        assert db_info["usage_records"] == 1

class TestTokenResponseParser:
    """Test token response parsing"""
    
    @pytest.fixture
    def parser(self):
        return TokenResponseParser()
    
    def test_parse_openai_response(self, parser):
        """Test parsing OpenAI API response"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_usage = Mock()
        mock_usage.prompt_tokens = 1000
        mock_usage.completion_tokens = 500
        mock_response.usage = mock_usage
        
        prompt_tokens, completion_tokens = parser.parse_openai_response(mock_response)
        
        assert prompt_tokens == 1000
        assert completion_tokens == 500
    
    def test_parse_openai_response_missing_usage(self, parser):
        """Test parsing OpenAI response without usage info"""
        mock_response = Mock(spec=[])  # No usage attribute
        
        prompt_tokens, completion_tokens = parser.parse_openai_response(mock_response)
        
        assert prompt_tokens == 0
        assert completion_tokens == 0
    
    def test_parse_anthropic_response(self, parser):
        """Test parsing Anthropic API response"""
        # Mock Anthropic response
        mock_response = Mock()
        mock_usage = Mock()
        mock_usage.input_tokens = 800
        mock_usage.output_tokens = 400
        mock_response.usage = mock_usage
        
        prompt_tokens, completion_tokens = parser.parse_anthropic_response(mock_response, "test prompt")
        
        assert prompt_tokens == 800
        assert completion_tokens == 400
    
    def test_parse_anthropic_response_fallback(self, parser):
        """Test Anthropic response parsing with fallback estimation"""
        # Mock response without usage
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "This is a test response"
        mock_response.content = [mock_content]
        
        # Remove usage attribute
        del mock_response.usage
        
        prompt_tokens, completion_tokens = parser.parse_anthropic_response(mock_response, "test prompt")
        
        assert prompt_tokens > 0  # Should estimate based on prompt
        assert completion_tokens > 0  # Should estimate based on response
    
    def test_parse_google_response(self, parser):
        """Test parsing Google Gemini API response"""
        # Mock Google response
        mock_response = Mock()
        mock_usage = Mock()
        mock_usage.prompt_token_count = 1200
        mock_usage.candidates_token_count = 600
        mock_response.usage_metadata = mock_usage
        
        prompt_tokens, completion_tokens = parser.parse_google_response(mock_response, "test prompt")
        
        assert prompt_tokens == 1200
        assert completion_tokens == 600
    
    def test_estimate_tokens(self, parser):
        """Test token estimation"""
        text = "This is a test sentence with multiple words."
        tokens = parser._estimate_tokens(text)
        
        assert tokens > 0
        assert tokens < len(text)  # Should be less than character count
    
    def test_estimate_tokens_empty(self, parser):
        """Test token estimation for empty text"""
        tokens = parser._estimate_tokens("")
        assert tokens == 0
    
    def test_estimate_tokens_json(self, parser):
        """Test token estimation for JSON content"""
        json_text = '{"key": "value", "number": 123}'
        tokens = parser._estimate_tokens(json_text)
        
        # JSON should have higher token density
        regular_tokens = parser._estimate_tokens("regular text with same length as json")
        assert tokens >= regular_tokens * 1.1  # Should be at least 10% more

class TestUsageReporter:
    """Test usage reporting functionality"""
    
    @pytest.fixture
    def temp_reporter(self):
        """Create a temporary usage reporter for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_tokens.db"
            tracker = TokenTracker(str(db_path))
            reporter = UsageReporter(tracker)
            yield reporter, tracker
    
    def test_generate_daily_report_empty(self, temp_reporter):
        """Test daily report with no data"""
        reporter, _ = temp_reporter
        
        report = reporter.generate_daily_report()
        
        assert "date" in report
        assert "summary" in report
        assert report["summary"] == "No usage recorded for this date"
    
    def test_generate_daily_report_with_data(self, temp_reporter):
        """Test daily report with usage data"""
        reporter, tracker = temp_reporter
        
        # Add some usage data
        tracker.track_usage("gpt-4o-mini", 1000, 500, 2.0, "phase1", success=True)
        tracker.track_usage("claude-3-5-sonnet-20241022", 800, 400, 1.8, "phase2", success=True)
        
        report = reporter.generate_daily_report()
        
        assert "date" in report
        assert "summary" in report
        assert report["summary"]["total_requests"] == 2
        assert "breakdown" in report
        assert "projections" in report
    
    def test_generate_weekly_report(self, temp_reporter):
        """Test weekly report generation"""
        reporter, tracker = temp_reporter
        
        # Add usage data
        tracker.track_usage("gpt-4o-mini", 1000, 500, 2.0, "phase1")
        
        report = reporter.generate_weekly_report()
        
        assert "period" in report
        assert report["period"] == "Weekly (7 days)"
        assert "summary" in report
        assert "breakdown" in report
    
    def test_generate_monthly_report(self, temp_reporter):
        """Test monthly report generation"""
        reporter, tracker = temp_reporter
        
        # Add usage data
        tracker.track_usage("gpt-4o-mini", 1000, 500, 2.0, "phase1")
        
        report = reporter.generate_monthly_report()
        
        assert "period" in report
        assert report["period"] == "Monthly (30 days)"
        assert "summary" in report
        assert "optimization_recommendations" in report
    
    def test_generate_custom_report(self, temp_reporter):
        """Test custom date range report"""
        reporter, tracker = temp_reporter
        
        # Add usage data
        tracker.track_usage("gpt-4o-mini", 1000, 500, 2.0, "phase1")
        
        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        report = reporter.generate_custom_report(
            start_date=start_date,
            end_date=end_date,
            model_filter="gpt-4o-mini"
        )
        
        assert "period" in report
        assert "filters" in report
        assert report["filters"]["model"] == "gpt-4o-mini"
        assert "summary" in report

class TestTrackedModelOrchestrator:
    """Test tracked model orchestrator"""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock tracked orchestrator"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_tokens.db"
            
            # Mock the parent class initialization
            with patch('src.tracking.tracked_orchestrator.ModelOrchestrator.__init__'):
                orchestrator = TrackedModelOrchestrator(
                    enable_tracking=True,
                    phase="test_phase"
                )
                
                # Set up required attributes
                orchestrator.models = {
                    'gpt4': Mock(),
                    'claude': Mock(),
                    'gemini': Mock()
                }
                orchestrator.enable_validation = False
                orchestrator.validation_orchestrator = None
                
                # Initialize token tracker with temp path
                orchestrator.token_tracker = TokenTracker(str(db_path))
                orchestrator.response_parser = TokenResponseParser()
                
                yield orchestrator
    
    def test_set_phase(self, mock_orchestrator):
        """Test setting phase"""
        mock_orchestrator.set_phase("new_phase")
        assert mock_orchestrator.current_phase == "new_phase"
    
    def test_set_session_id(self, mock_orchestrator):
        """Test setting session ID"""
        mock_orchestrator.set_session_id("test_session_123")
        assert mock_orchestrator.session_id == "test_session_123"
    
    def test_get_api_provider_for_model(self, mock_orchestrator):
        """Test API provider detection"""
        assert mock_orchestrator._get_api_provider_for_model("gpt-4o-mini") == "openai"
        assert mock_orchestrator._get_api_provider_for_model("claude-3-5-sonnet") == "anthropic"
        assert mock_orchestrator._get_api_provider_for_model("gemini-1.5-pro") == "google"
        assert mock_orchestrator._get_api_provider_for_model("unknown-model") == "unknown"
    
    @pytest.mark.asyncio
    async def test_run_single_model_tracked_success(self, mock_orchestrator):
        """Test successful single model tracking"""
        # Mock model interface
        mock_model = Mock()
        mock_model._prepare_prompt = Mock(return_value="prepared prompt")
        mock_model.call_model = AsyncMock(return_value=Mock(
            confidence_self_assessment=0.85,
            raw_response="test response"
        ))
        mock_model._last_api_response = Mock()
        
        mock_orchestrator.models['gpt4'] = mock_model
        
        # Mock response parser
        mock_orchestrator.response_parser.parse_response_by_provider = Mock(return_value=(1000, 500))
        
        result = await mock_orchestrator.run_single_model_tracked(
            "gpt4", "test prompt", {"test": "data"}
        )
        
        assert result is not None
        assert mock_model.call_model.called
    
    @pytest.mark.asyncio
    async def test_run_single_model_tracked_failure(self, mock_orchestrator):
        """Test failed single model tracking"""
        # Mock model interface that fails
        mock_model = Mock()
        mock_model._prepare_prompt = Mock(return_value="prepared prompt")
        mock_model.call_model = AsyncMock(side_effect=Exception("API Error"))
        
        mock_orchestrator.models['gpt4'] = mock_model
        
        # Should track usage even on failure
        with pytest.raises(Exception, match="API Error"):
            await mock_orchestrator.run_single_model_tracked(
                "gpt4", "test prompt", {"test": "data"}
            )
        
        # Check that usage was tracked
        stats = mock_orchestrator.token_tracker.get_usage_stats(days=1)
        assert stats["summary"]["total_requests"] == 1
        assert stats["summary"]["successful_requests"] == 0

class TestIntegration:
    """Integration tests for the complete tracking system"""
    
    @pytest.fixture
    def integration_setup(self):
        """Set up complete integration test environment"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "integration_test.db"
            
            # Create components
            tracker = TokenTracker(str(db_path))
            reporter = UsageReporter(tracker)
            console_reporter = ConsoleReporter(reporter)
            
            yield {
                'tracker': tracker,
                'reporter': reporter,
                'console_reporter': console_reporter,
                'db_path': db_path
            }
    
    def test_full_tracking_workflow(self, integration_setup):
        """Test complete workflow from tracking to reporting"""
        components = integration_setup
        tracker = components['tracker']
        reporter = components['reporter']
        
        # Track various usage scenarios
        tracker.track_usage("gpt-4o-mini", 1000, 500, 2.5, "phase1", success=True)
        tracker.track_usage("claude-3-5-sonnet-20241022", 800, 400, 1.8, "phase1", success=True)
        tracker.track_usage("gemini-1.5-flash", 1200, 600, 1.5, "phase2", success=False, error_message="Timeout")
        tracker.track_usage("gpt-4o-mini", 900, 450, 2.1, "phase2", success=True)
        
        # Generate reports
        daily_report = reporter.generate_daily_report()
        weekly_report = reporter.generate_weekly_report()
        
        # Validate tracking
        assert daily_report["summary"]["total_requests"] == 4
        assert daily_report["summary"]["successful_requests"] == 3
        assert daily_report["summary"]["success_rate"] == 0.75
        
        # Validate cost calculations
        total_cost = daily_report["summary"]["total_cost"]
        assert total_cost > 0
        
        # Validate breakdowns
        assert len(daily_report["breakdown"]["by_model"]) == 3
        assert len(daily_report["breakdown"]["by_phase"]) == 2
        
        # Test data export
        json_export = tracker.export_usage_data(format="json")
        exported_data = json.loads(json_export)
        assert len(exported_data) == 4
        
        csv_export = tracker.export_usage_data(format="csv")
        assert "model_name" in csv_export
        assert len(csv_export.split('\n')) == 6  # Header + 4 data rows + empty line
    
    def test_database_persistence(self, integration_setup):
        """Test database persistence across tracker instances"""
        db_path = integration_setup['db_path']
        
        # Create first tracker instance and add data
        tracker1 = TokenTracker(str(db_path))
        tracker1.track_usage("gpt-4o-mini", 1000, 500, 2.0, "phase1")
        
        # Create second tracker instance with same database
        tracker2 = TokenTracker(str(db_path))
        stats = tracker2.get_usage_stats(days=1)
        
        # Should see the data from first tracker
        assert stats["summary"]["total_requests"] == 1
    
    def test_concurrent_tracking(self, integration_setup):
        """Test concurrent access to tracking system"""
        tracker = integration_setup['tracker']
        
        # Simulate concurrent requests
        import threading
        import time
        
        def track_usage(model_name, request_id):
            tracker.track_usage(
                model_name=model_name,
                prompt_tokens=1000,
                completion_tokens=500,
                processing_time=1.0,
                phase="concurrent_test",
                request_id=f"req_{request_id}"
            )
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=track_usage, args=("gpt-4o-mini", i))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Verify all requests were tracked
        stats = tracker.get_usage_stats(days=1)
        assert stats["summary"]["total_requests"] == 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])