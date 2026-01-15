"""
Asana Registry
Central registry for all yoga pose definitions
Allows manual selection and future auto-recognition
"""

from typing import Dict, Optional
from src.services.asana_base import AsanaBase
from src.asanas.mountain import MountainPose
from src.asanas.warrior_2 import WarriorII
from src.asanas.tree import TreePose


class AsanaRegistry:
    """
    Registry for all available yoga poses
    
    Provides:
    - Manual pose selection by name
    - Pose metadata
    - Future: auto-recognition integration point
    """
    
    def __init__(self):
        self._asanas: Dict[str, AsanaBase] = {}
        self._register_default_asanas()
    
    def _register_default_asanas(self):
        """Register all available asanas"""
        # Mountain Pose
        self.register('mountain', MountainPose())
        self.register('tadasana', MountainPose())  # Sanskrit alias
        
        # Warrior II
        self.register('warrior_2', WarriorII())
        self.register('warrior_ii', WarriorII())
        self.register('virabhadrasana_2', WarriorII())  # Sanskrit alias
        
        # Tree Pose (both sides)
        self.register('tree_right', TreePose(standing_leg='right'))
        self.register('tree_left', TreePose(standing_leg='left'))
        self.register('vrksasana_right', TreePose(standing_leg='right'))
        self.register('vrksasana_left', TreePose(standing_leg='left'))
    
    def register(self, name: str, asana: AsanaBase):
        """
        Register an asana with a name
        
        Args:
            name: Identifier for the asana (lowercase, underscores)
            asana: Asana instance
        """
        self._asanas[name.lower()] = asana
    
    def get(self, name: str) -> Optional[AsanaBase]:
        """
        Get asana by name
        
        Args:
            name: Asana identifier
            
        Returns:
            Asana instance or None if not found
        """
        return self._asanas.get(name.lower())
    
    def list_available(self) -> list:
        """
        Get list of all available asana names
        
        Returns:
            List of asana identifiers
        """
        # Return unique asanas (not aliases)
        seen = set()
        unique = []
        
        for name, asana in self._asanas.items():
            asana_id = id(asana)
            if asana_id not in seen:
                seen.add(asana_id)
                unique.append({
                    'id': name,
                    'name': asana.name,
                    'sanskrit': asana.sanskrit_name
                })
        
        return unique
    
    def get_by_id(self, asana_id: str) -> Optional[AsanaBase]:
        """
        Get asana by ID (alias for get)
        
        Args:
            asana_id: Asana identifier
            
        Returns:
            Asana instance or None
        """
        return self.get(asana_id)


# Global registry instance
_registry = None


def get_registry() -> AsanaRegistry:
    """
    Get the global asana registry (singleton)
    
    Returns:
        AsanaRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = AsanaRegistry()
    return _registry


def get_asana(name: str) -> Optional[AsanaBase]:
    """
    Convenience function to get asana from global registry
    
    Args:
        name: Asana identifier
        
    Returns:
        Asana instance or None
    """
    return get_registry().get(name)


def list_asanas() -> list:
    """
    Convenience function to list all available asanas
    
    Returns:
        List of asana metadata dicts
    """
    return get_registry().list_available()
