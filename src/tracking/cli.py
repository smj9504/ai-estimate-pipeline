# src/tracking/cli.py
#!/usr/bin/env python3
"""
Command-line interface for token usage tracking and reporting
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.tracking.token_tracker import TokenTracker
from src.tracking.usage_reporter import UsageReporter, ConsoleReporter

def setup_argparser():
    """Set up command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="AI Estimate Pipeline - Token Usage Tracking CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current usage
  python -m src.tracking.cli stats

  # Show detailed daily report
  python -m src.tracking.cli report daily

  # Show usage for specific model
  python -m src.tracking.cli stats --model gpt-4o-mini

  # Export last 30 days to CSV
  python -m src.tracking.cli export csv --days 30

  # Clean up old data (keep last 90 days)
  python -m src.tracking.cli cleanup --days 90

  # Show model pricing
  python -m src.tracking.cli pricing
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show usage statistics')
    stats_parser.add_argument('--days', type=int, default=7, help='Number of days to analyze (default: 7)')
    stats_parser.add_argument('--model', help='Filter by model name')
    stats_parser.add_argument('--provider', help='Filter by API provider (openai, anthropic, google)')
    stats_parser.add_argument('--phase', help='Filter by phase')
    stats_parser.add_argument('--detailed', action='store_true', help='Show detailed breakdown')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate usage reports')
    report_subparsers = report_parser.add_subparsers(dest='report_type', help='Report types')
    
    daily_parser = report_subparsers.add_parser('daily', help='Generate daily report')
    daily_parser.add_argument('--date', help='Date in YYYY-MM-DD format (default: today)')
    
    report_subparsers.add_parser('weekly', help='Generate weekly report')
    report_subparsers.add_parser('monthly', help='Generate monthly report')
    
    custom_parser = report_subparsers.add_parser('custom', help='Generate custom date range report')
    custom_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    custom_parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    custom_parser.add_argument('--model', help='Filter by model name')
    custom_parser.add_argument('--phase', help='Filter by phase')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export usage data')
    export_parser.add_argument('format', choices=['csv', 'json'], help='Export format')
    export_parser.add_argument('--days', type=int, default=30, help='Number of days to export (default: 30)')
    export_parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    export_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    export_parser.add_argument('--output', help='Output file path')
    
    # Recent command
    recent_parser = subparsers.add_parser('recent', help='Show recent usage records')
    recent_parser.add_argument('--limit', type=int, default=20, help='Number of records to show (default: 20)')
    
    # Projections command
    subparsers.add_parser('projections', help='Show cost projections')
    
    # Pricing command
    subparsers.add_parser('pricing', help='Show current model pricing')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old usage data')
    cleanup_parser.add_argument('--days', type=int, default=90, help='Days of data to keep (default: 90)')
    cleanup_parser.add_argument('--confirm', action='store_true', help='Confirm cleanup without prompt')
    
    # Database command
    subparsers.add_parser('database', help='Show database information')
    
    # Live command (real-time monitoring)
    live_parser = subparsers.add_parser('live', help='Show live usage monitoring')
    live_parser.add_argument('--refresh', type=int, default=5, help='Refresh interval in seconds (default: 5)')
    
    return parser

def format_currency(amount):
    """Format currency amount"""
    if amount == 0:
        return "$0.00"
    elif amount < 0.001:
        return f"${amount:.6f}"
    elif amount < 0.1:
        return f"${amount:.4f}"
    else:
        return f"${amount:.2f}"

def format_number(number):
    """Format large numbers with commas"""
    if isinstance(number, int):
        return f"{number:,}"
    elif isinstance(number, float):
        if number >= 1000:
            return f"{number:,.0f}"
        else:
            return f"{number:.2f}"
    return str(number)

def print_stats(args, tracker, reporter):
    """Print usage statistics"""
    stats = tracker.get_usage_stats(
        days=args.days,
        model_name=args.model,
        api_provider=args.provider,
        phase=args.phase
    )
    
    if stats["summary"]["total_requests"] == 0:
        print("[STATS] No usage data found for the specified criteria.")
        return
    
    summary = stats["summary"]
    period = stats["period"]
    
    print(f"[STATS] Usage Statistics ({period['start'][:10]} to {period['end'][:10]})")
    print("=" * 80)
    print(f"Total Requests: {format_number(summary['total_requests'])}")
    print(f"Successful: {format_number(summary['successful_requests'])} ({summary['success_rate']:.1%})")
    print(f"Total Tokens: {format_number(summary['total_tokens'])}")
    print(f"Total Cost: {format_currency(summary['total_cost'])}")
    print(f"Average Time: {summary['avg_processing_time']:.2f}s")
    
    if args.detailed or len(stats["breakdown"]["by_model"]) <= 5:
        print(f"\n[MODELS] By Model:")
        for model, data in stats["breakdown"]["by_model"].items():
            success_rate = f"{data['success_rate']:.1%}" if 'success_rate' in data else "N/A"
            print(f"  • {model[:35]:<35} | {format_number(data['requests']):>6} reqs | "
                  f"{format_currency(data['cost']):>10} | {success_rate:>6}")
    
    if args.detailed and stats["breakdown"]["by_phase"]:
        print(f"\n🔄 By Phase:")
        for phase, data in stats["breakdown"]["by_phase"].items():
            print(f"  • {phase[:20]:<20} | {format_number(data['requests']):>6} reqs | "
                  f"{format_currency(data['cost']):>10}")
    
    print("=" * 80)

def print_report(args, reporter):
    """Print usage reports"""
    if args.report_type == 'daily':
        report = reporter.generate_daily_report(args.date)
        print_daily_report(report)
    elif args.report_type == 'weekly':
        report = reporter.generate_weekly_report()
        print_weekly_report(report)
    elif args.report_type == 'monthly':
        report = reporter.generate_monthly_report()
        print_monthly_report(report)
    elif args.report_type == 'custom':
        report = reporter.generate_custom_report(
            args.start, args.end, args.model, args.phase
        )
        print_custom_report(report)

def print_daily_report(report):
    """Print daily report"""
    if isinstance(report["summary"], str):
        print(f"📅 Daily Report - {report['date']}")
        print(report["summary"])
        return
    
    print(f"📅 Daily Report - {report['date']}")
    print("=" * 80)
    
    summary = report["summary"]
    print(f"Requests: {format_number(summary['total_requests'])} | "
          f"Success: {summary['success_rate']:.1%} | "
          f"Cost: {format_currency(summary['total_cost'])} | "
          f"Tokens: {format_number(summary['total_tokens'])}")
    
    if "efficiency_metrics" in report:
        metrics = report["efficiency_metrics"]
        print(f"\n📈 Efficiency Metrics:")
        print(f"  • Cost per request: {format_currency(metrics['cost_per_request'])}")
        print(f"  • Tokens per request: {format_number(metrics['tokens_per_request'])}")
        print(f"  • Cost per token: {format_currency(metrics['cost_per_token'])}")
    
    if "projections" in report and report["projections"]:
        proj = report["projections"]
        print(f"\n📊 Cost Projections:")
        print(f"  • Daily: {format_currency(proj['daily_projection'])}")
        print(f"  • Monthly: {format_currency(proj['monthly_projection'])}")
        print(f"  • Yearly: {format_currency(proj['yearly_projection'])}")
    
    print("=" * 80)

def print_weekly_report(report):
    """Print weekly report"""
    print("📅 Weekly Report (Last 7 Days)")
    print("=" * 80)
    
    summary = report["summary"]
    print(f"Total Requests: {format_number(summary['total_requests'])}")
    print(f"Total Cost: {format_currency(summary['total_cost'])}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    
    if "top_performers" in report and report["top_performers"]["top_model"]:
        top_model = report["top_performers"]["top_model"]
        print(f"\n🏆 Top Model: {top_model['name']} ({format_number(top_model['requests'])} requests)")
    
    print("=" * 80)

def print_monthly_report(report):
    """Print monthly report"""
    print("📅 Monthly Report (Last 30 Days)")
    print("=" * 80)
    
    summary = report["summary"]
    print(f"Total Requests: {format_number(summary['total_requests'])}")
    print(f"Total Cost: {format_currency(summary['total_cost'])}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Average Processing Time: {summary['avg_processing_time']:.2f}s")
    
    if "optimization_recommendations" in report and report["optimization_recommendations"]:
        print(f"\n💡 Optimization Recommendations:")
        for rec in report["optimization_recommendations"][:3]:  # Top 3
            print(f"  • {rec}")
    
    print("=" * 80)

def print_custom_report(report):
    """Print custom report"""
    print(f"📅 Custom Report - {report['period']}")
    if "filters" in report:
        filters = report["filters"]
        if filters["model"] or filters["phase"]:
            print(f"Filters: Model={filters['model'] or 'All'}, Phase={filters['phase'] or 'All'}")
    
    print("=" * 80)
    
    summary = report["summary"]
    print(f"Total Requests: {format_number(summary['total_requests'])}")
    print(f"Total Cost: {format_currency(summary['total_cost'])}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    
    if "range_analysis" in report:
        analysis = report["range_analysis"]
        daily_avg = analysis["daily_average"]
        print(f"\nDaily Averages:")
        print(f"  • Requests: {format_number(daily_avg['requests'])}")
        print(f"  • Cost: {format_currency(daily_avg['cost'])}")
    
    print("=" * 80)

def export_data(args, reporter):
    """Export usage data"""
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    elif args.days:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    else:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    if args.format == 'csv':
        output_path = reporter.export_usage_csv(start_date, end_date, args.output)
        print(f"📄 Data exported to CSV: {output_path}")
    else:
        output_path = reporter.export_usage_json(start_date, end_date, args.output)
        print(f"📄 Data exported to JSON: {output_path}")

def show_recent(args, tracker):
    """Show recent usage records"""
    records = tracker.get_recent_usage(limit=args.limit)
    
    if not records:
        print("📊 No recent usage records found.")
        return
    
    print(f"📊 Recent Usage Records (Last {len(records)})")
    print("=" * 80)
    print(f"{'Time':<20} {'Model':<25} {'Tokens':<10} {'Cost':<12} {'Phase':<15} {'Status'}")
    print("-" * 80)
    
    for record in records:
        timestamp = record['timestamp'][:19].replace('T', ' ')
        model = record['model_name'][:24]
        tokens = f"{record['total_tokens']:,}"
        cost = format_currency(record['estimated_cost'])
        phase = (record['phase'] or 'N/A')[:14]
        status = "✅" if record['success'] else "❌"
        
        print(f"{timestamp:<20} {model:<25} {tokens:<10} {cost:<12} {phase:<15} {status}")
    
    print("=" * 80)

def show_projections(tracker):
    """Show cost projections"""
    projections = tracker.get_cost_projection()
    
    if not projections or all(v == 0 for v in projections.values()):
        print("📊 No recent usage data available for projections.")
        return
    
    print("📊 Cost Projections (Based on Recent Usage)")
    print("=" * 80)
    print(f"Daily:   {format_currency(projections['daily_projection'])}")
    print(f"Weekly:  {format_currency(projections['weekly_projection'])}")
    print(f"Monthly: {format_currency(projections['monthly_projection'])}")
    print(f"Yearly:  {format_currency(projections['yearly_projection'])}")
    print("=" * 80)

def show_pricing():
    """Show model pricing information"""
    from src.tracking.token_tracker import TokenPricingManager
    
    pricing = TokenPricingManager.get_all_pricing()
    
    print("💰 Current Model Pricing (per 1K tokens)")
    print("=" * 80)
    print(f"{'Model':<40} {'Input':<12} {'Output':<12}")
    print("-" * 80)
    
    for model, prices in pricing.items():
        input_cost = format_currency(prices['input'])
        output_cost = format_currency(prices['output'])
        print(f"{model:<40} {input_cost:<12} {output_cost:<12}")
    
    print("=" * 80)
    print("Note: Prices are subject to change. Check API provider documentation for latest rates.")

def cleanup_data(args, tracker):
    """Clean up old data"""
    if not args.confirm:
        response = input(f"⚠️  This will delete usage data older than {args.days} days. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cleanup cancelled.")
            return
    
    print(f"🧹 Cleaning up data older than {args.days} days...")
    tracker.cleanup_old_data(args.days)
    print("✅ Cleanup completed.")

def show_database_info(tracker):
    """Show database information"""
    db_info = tracker.get_database_size()
    
    print("💾 Database Information")
    print("=" * 80)
    print(f"Path: {db_info['database_path']}")
    print(f"Size: {db_info['size_mb']:.2f} MB ({format_number(db_info['size_bytes'])} bytes)")
    print(f"Usage Records: {format_number(db_info['usage_records'])}")
    print(f"Summary Records: {format_number(db_info['summary_records'])}")
    print(f"Total Records: {format_number(db_info['total_records'])}")
    print("=" * 80)

def live_monitoring(args, tracker, reporter):
    """Live monitoring mode"""
    import time
    import os
    
    try:
        while True:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("📊 Live Token Usage Monitoring")
            print(f"Refreshing every {args.refresh} seconds... (Press Ctrl+C to exit)")
            print("=" * 80)
            
            # Show current stats
            console_reporter = ConsoleReporter(reporter)
            console_reporter.display_current_usage()
            
            print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(args.refresh)
    
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")

def main():
    """Main CLI function"""
    parser = setup_argparser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Initialize tracking components
        tracker = TokenTracker()
        reporter = UsageReporter(tracker)
        
        # Execute commands
        if args.command == 'stats':
            print_stats(args, tracker, reporter)
        
        elif args.command == 'report':
            if not args.report_type:
                print("Error: Please specify a report type (daily, weekly, monthly, custom)")
                return
            print_report(args, reporter)
        
        elif args.command == 'export':
            export_data(args, reporter)
        
        elif args.command == 'recent':
            show_recent(args, tracker)
        
        elif args.command == 'projections':
            show_projections(tracker)
        
        elif args.command == 'pricing':
            show_pricing()
        
        elif args.command == 'cleanup':
            cleanup_data(args, tracker)
        
        elif args.command == 'database':
            show_database_info(tracker)
        
        elif args.command == 'live':
            live_monitoring(args, tracker, reporter)
        
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted.")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())