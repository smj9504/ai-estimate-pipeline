"""
Centralized Test Data Loader
Provides unified access to actual test data files instead of dummy fixtures.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging

from src.models.data_models import ProjectData


logger = logging.getLogger(__name__)


@dataclass
class TestDataSet:
    """Complete test data set for estimation pipeline testing"""
    demo_data: List[Dict[str, Any]]
    measurement_data: List[Dict[str, Any]] 
    intake_form: str
    combined_project_data: List[Dict[str, Any]]
    name: str
    description: str


class TestDataLoader:
    """Centralized loader for actual test data files"""
    
    def __init__(self, test_data_dir: Optional[str] = None):
        """Initialize test data loader
        
        Args:
            test_data_dir: Override default test_data directory path
        """
        if test_data_dir:
            self.test_data_dir = Path(test_data_dir)
        else:
            # Default to project root test_data directory
            project_root = Path(__file__).parent.parent.parent
            self.test_data_dir = project_root / "test_data"
        
        self._validate_test_data_dir()
        self._cached_data = {}
    
    def _validate_test_data_dir(self) -> None:
        """Validate test data directory exists and contains required files"""
        if not self.test_data_dir.exists():
            raise FileNotFoundError(f"Test data directory not found: {self.test_data_dir}")
        
        required_files = [
            "sample_demo.json",
            "sample_measurement.json", 
            "sample_intake_form.txt"
        ]
        
        for file_name in required_files:
            file_path = self.test_data_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required test data file not found: {file_path}")
    
    def load_demo_data(self, cached: bool = True) -> List[Dict[str, Any]]:
        """Load demolition scope test data
        
        Args:
            cached: Use cached data if available
            
        Returns:
            List of demolition data by location/floor
        """
        cache_key = "demo_data"
        if cached and cache_key in self._cached_data:
            return self._cached_data[cache_key]
        
        demo_file = self.test_data_dir / "sample_demo.json"
        try:
            with open(demo_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if cached:
                self._cached_data[cache_key] = data
            
            logger.info(f"Loaded demo data: {len(data)} floors with total "
                       f"{sum(len(floor.get('rooms', [])) for floor in data)} rooms")
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in demo data file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load demo data: {e}")
    
    def load_measurement_data(self, cached: bool = True) -> List[Dict[str, Any]]:
        """Load measurement test data
        
        Args:
            cached: Use cached data if available
            
        Returns:
            List of measurement data by location/floor
        """
        cache_key = "measurement_data"
        if cached and cache_key in self._cached_data:
            return self._cached_data[cache_key]
        
        measurement_file = self.test_data_dir / "sample_measurement.json"
        try:
            with open(measurement_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if cached:
                self._cached_data[cache_key] = data
            
            logger.info(f"Loaded measurement data: {len(data)} floors with total "
                       f"{sum(len(floor.get('rooms', [])) for floor in data)} rooms")
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in measurement data file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load measurement data: {e}")
    
    def load_intake_form(self, cached: bool = True) -> str:
        """Load intake form test data
        
        Args:
            cached: Use cached data if available
            
        Returns:
            Intake form text content
        """
        cache_key = "intake_form"
        if cached and cache_key in self._cached_data:
            return self._cached_data[cache_key]
        
        intake_file = self.test_data_dir / "sample_intake_form.txt"
        try:
            with open(intake_file, 'r', encoding='utf-8') as f:
                data = f.read().strip()
            
            if cached:
                self._cached_data[cache_key] = data
            
            logger.info(f"Loaded intake form: {len(data)} characters")
            return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load intake form: {e}")
    
    def create_combined_project_data(self) -> List[Dict[str, Any]]:
        """Create combined project data by merging demo, measurement, and intake data
        
        This combines the three separate test data files into the format expected
        by the ProjectData model for end-to-end testing.
        
        Returns:
            Combined project data in ProjectData JSON format
        """
        demo_data = self.load_demo_data()
        measurement_data = self.load_measurement_data()
        intake_form = self.load_intake_form()
        
        # Create jobsite info from intake form
        jobsite_info = {
            "Jobsite": "456 Elm Street Test Project",
            "occupancy": "Single Family",
            "company": {},
            "intake_form": intake_form
        }
        
        # Merge demo and measurement data by matching floors/rooms
        combined_floors = []
        
        # Create floor mapping for efficient lookup
        demo_floor_map = {floor["location"]: floor for floor in demo_data}
        measurement_floor_map = {floor["location"]: floor for floor in measurement_data}
        
        # Process all unique floor locations
        all_locations = set(demo_floor_map.keys()) | set(measurement_floor_map.keys())
        
        for location in sorted(all_locations):
            demo_floor = demo_floor_map.get(location, {"location": location, "rooms": []})
            measurement_floor = measurement_floor_map.get(location, {"location": location, "rooms": []})
            
            # Create room mapping for this floor
            demo_room_map = {room["name"]: room for room in demo_floor.get("rooms", [])}
            measurement_room_map = {room["name"]: room for room in measurement_floor.get("rooms", [])}
            
            # Process all unique room names for this floor
            all_room_names = set(demo_room_map.keys()) | set(measurement_room_map.keys())
            
            combined_rooms = []
            for room_name in sorted(all_room_names):
                demo_room = demo_room_map.get(room_name, {"name": room_name})
                measurement_room = measurement_room_map.get(room_name, {"name": room_name})
                
                # Combine room data
                combined_room = {
                    "name": room_name,
                    "material": self._extract_default_materials(location),
                    "work_scope": self._extract_default_work_scope(location),
                    "measurements": measurement_room.get("measurements", {}),
                    "demo_scope(already demo'd)": demo_room.get("demo_scope", {}),
                    "additional_notes": self._extract_additional_notes_from_intake(room_name, intake_form)
                }
                
                combined_rooms.append(combined_room)
            
            combined_floors.append({
                "location": location,
                "rooms": combined_rooms
            })
        
        return [jobsite_info] + combined_floors
    
    def _extract_default_materials(self, location: str) -> Dict[str, str]:
        """Extract default materials for a floor from intake form patterns"""
        # These would normally be parsed from intake form, but for testing use defaults
        if "1st Floor" in location:
            return {
                "Floor": "Laminate Wood",
                "wall": "drywall", 
                "ceiling": "drywall",
                "Baseboard": "wood",
                "Quarter Round": "wood"
            }
        elif "2nd Floor" in location:
            return {
                "Floor": "Carpet",
                "wall": "drywall",
                "ceiling": "drywall", 
                "Baseboard": "wood",
                "Quarter Round": "wood"
            }
        elif "Basement" in location:
            return {
                "Floor": "Vinyl Plank",
                "wall": "drywall",
                "ceiling": "drop-ceiling",
                "Baseboard": "vinyl",
                "Quarter Round": "N/A"
            }
        else:
            return {
                "Floor": "Unknown",
                "wall": "drywall",
                "ceiling": "drywall",
                "Baseboard": "wood",
                "Quarter Round": "wood"
            }
    
    def _extract_default_work_scope(self, location: str) -> Dict[str, str]:
        """Extract default work scope for a floor"""
        if "1st Floor" in location:
            return {
                "Flooring": "Remove & Replace",
                "Wall": "Patch",
                "Ceiling": "Patch", 
                "Baseboard": "Remove & Replace",
                "Quarter Round": "Remove & Replace",
                "Paint Scope": "Wall, Ceiling"
            }
        elif "2nd Floor" in location:
            return {
                "Flooring": "Remove & Replace",
                "Wall": "Paint Only",
                "Ceiling": "Paint Only",
                "Baseboard": "Clean & Reset", 
                "Quarter Round": "Clean & Reset",
                "Paint Scope": "Wall, Ceiling, Trim"
            }
        elif "Basement" in location:
            return {
                "Flooring": "Remove & Replace",
                "Wall": "Remove & Replace (lower 4ft)",
                "Ceiling": "Remove",
                "Baseboard": "Remove & Replace",
                "Quarter Round": "N/A",
                "Paint Scope": "Wall, Ceiling"
            }
        else:
            return {
                "Flooring": "Remove & Replace",
                "Wall": "Paint",
                "Ceiling": "Paint", 
                "Baseboard": "Remove & Replace",
                "Quarter Round": "Remove & Replace",
                "Paint Scope": "Full Room"
            }
    
    def _extract_additional_notes_from_intake(self, room_name: str, intake_form: str) -> Dict[str, List[str]]:
        """Extract additional notes for a specific room from intake form"""
        # Parse intake form for room-specific notes
        protection_notes = []
        detach_reset_notes = []
        
        # Look for room-specific sections in intake form
        lines = intake_form.split('\n')
        in_room_section = False
        current_room = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Room:'):
                current_room = line.replace('Room:', '').strip()
                in_room_section = (current_room.lower() == room_name.lower())
            elif line.startswith('Room:') or (line.startswith('[') and line.endswith(']')):
                in_room_section = False
            elif in_room_section:
                if 'Protection:' in line or 'protection' in line.lower():
                    # Extract protection items
                    if 'cover' in line.lower() or 'seal' in line.lower() or 'plastic' in line.lower():
                        protection_notes.append(line.replace('-', '').replace('Protection:', '').strip())
                elif 'Detach & Reset:' in line or 'detach' in line.lower():
                    # Extract detach/reset items
                    if 'detach' in line.lower() or 'reset' in line.lower():
                        detach_reset_notes.append(line.replace('-', '').replace('Detach & Reset:', '').strip())
        
        # Default notes if none found
        if not protection_notes and not detach_reset_notes:
            if 'living' in room_name.lower():
                protection_notes = ["cover furniture", "seal doorways"]
                detach_reset_notes = ["wall-mounted TV", "artwork"]
            elif 'kitchen' in room_name.lower():
                protection_notes = ["plastic wrap appliances"]
                detach_reset_notes = ["dishwasher", "refrigerator water line"]
            elif 'bathroom' in room_name.lower():
                protection_notes = ["cover fixtures"]
                detach_reset_notes = ["toilet", "vanity"]
        
        return {
            "protection": protection_notes,
            "detach_reset": detach_reset_notes
        }
    
    def get_test_dataset(self, name: str = "default") -> TestDataSet:
        """Get complete test data set
        
        Args:
            name: Name identifier for the test data set
            
        Returns:
            TestDataSet with all test data loaded and combined
        """
        return TestDataSet(
            demo_data=self.load_demo_data(),
            measurement_data=self.load_measurement_data(),
            intake_form=self.load_intake_form(),
            combined_project_data=self.create_combined_project_data(),
            name=name,
            description="Complete test data set from actual project files"
        )
    
    def validate_project_data(self, project_data: List[Dict[str, Any]]) -> bool:
        """Validate that project data can be converted to ProjectData model
        
        Args:
            project_data: Project data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Try to create ProjectData object
            project = ProjectData.from_json_list(project_data)
            
            # Basic validation checks
            if not project.jobsite_info.Jobsite:
                logger.error("Missing jobsite information")
                return False
            
            if not project.floors:
                logger.error("No floors found in project data")
                return False
            
            total_rooms = sum(len(floor.rooms) for floor in project.floors)
            if total_rooms == 0:
                logger.error("No rooms found in project data")
                return False
            
            logger.info(f"Validated project data: {len(project.floors)} floors, {total_rooms} rooms")
            return True
            
        except Exception as e:
            logger.error(f"Project data validation failed: {e}")
            return False
    
    def clear_cache(self) -> None:
        """Clear cached test data"""
        self._cached_data.clear()
        logger.info("Test data cache cleared")


# Convenience functions for backward compatibility
def load_test_demo_data() -> List[Dict[str, Any]]:
    """Load demo test data (backward compatibility)"""
    loader = TestDataLoader()
    return loader.load_demo_data()


def load_test_measurement_data() -> List[Dict[str, Any]]:
    """Load measurement test data (backward compatibility)"""
    loader = TestDataLoader()
    return loader.load_measurement_data()


def load_test_intake_form() -> str:
    """Load intake form test data (backward compatibility)"""
    loader = TestDataLoader()
    return loader.load_intake_form()


def get_combined_test_project_data() -> List[Dict[str, Any]]:
    """Get combined project data for testing (backward compatibility)"""
    loader = TestDataLoader()
    return loader.create_combined_project_data()


# Global test data loader instance
_global_loader = None

def get_test_data_loader() -> TestDataLoader:
    """Get global test data loader instance"""
    global _global_loader
    if _global_loader is None:
        _global_loader = TestDataLoader()
    return _global_loader