"""
Cross-modal relation discovery engine for TiMMInsight.

This module implements algorithms to automatically discover temporal and spatial
relationships between different modalities based on their metadata.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import re

from .metadata_graph import (
    ModalityMetadata, ModalityType, CrossModalRelation, RelationType,
    TemporalInfo, SpatialInfo, MultimodalMetadataGraph
)

logger = logging.getLogger(__name__)

try:
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    logger.warning("geopy not available. GPS-based spatial calculations will be disabled.")


@dataclass
class RelationDiscoveryConfig:
    """Configuration for relation discovery algorithms."""
    temporal_threshold: float = 0.3
    spatial_threshold: float = 0.3
    confidence_threshold: float = 0.2
    max_spatial_distance_km: float = 50.0
    enable_fuzzy_location_matching: bool = True
    enable_temporal_expression_matching: bool = True


class CrossModalRelationDiscoverer:
    """
    Engine for discovering cross-modal relationships between different modalities.
    
    This class implements algorithms to automatically detect temporal and spatial
    relationships based on the metadata extracted from various modalities.
    """
    
    def __init__(self, config: Optional[RelationDiscoveryConfig] = None):
        """Initialize the relation discoverer."""
        self.config = config or RelationDiscoveryConfig()
        self.discovered_relations = []
        logger.info("CrossModalRelationDiscoverer initialized")
    
    def discover_relations(self, metadata_graph: MultimodalMetadataGraph) -> List[CrossModalRelation]:
        """
        Discover all cross-modal relationships in the metadata graph.
        
        Args:
            metadata_graph: The multimodal metadata graph
            
        Returns:
            List of discovered cross-modal relations
        """
        logger.info("Starting cross-modal relation discovery")
        relations = []
        
        modalities = list(metadata_graph.modalities.values())
        total_pairs = (len(modalities) * (len(modalities) - 1)) // 2
        
        logger.info(f"Analyzing {len(modalities)} modalities ({total_pairs} pairs)")
        
        # Analyze all pairs of modalities
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                modal1, modal2 = modalities[i], modalities[j]
                
                try:
                    pair_relations = self._discover_relation_pair(modal1, modal2)
                    relations.extend(pair_relations)
                    
                    if pair_relations:
                        logger.debug(f"Found {len(pair_relations)} relations between "
                                   f"{modal1.modality_id} and {modal2.modality_id}")
                        
                except Exception as e:
                    logger.warning(f"Error analyzing pair {modal1.modality_id}-{modal2.modality_id}: {e}")
        
        self.discovered_relations = relations
        logger.info(f"Discovered {len(relations)} cross-modal relations")
        return relations
    
    def _discover_relation_pair(self, modal1: ModalityMetadata, modal2: ModalityMetadata) -> List[CrossModalRelation]:
        """
        Discover relationships between two specific modalities.
        
        Args:
            modal1, modal2: The two modalities to analyze
            
        Returns:
            List of relations found between the modalities
        """
        relations = []
        
        # Temporal relationship analysis
        temporal_score = self._calculate_temporal_relation_score(modal1, modal2)
        if temporal_score >= self.config.temporal_threshold:
            relation = self._create_temporal_relation(modal1, modal2, temporal_score)
            if relation:
                relations.append(relation)
        
        # Spatial relationship analysis
        spatial_score = self._calculate_spatial_relation_score(modal1, modal2)
        if spatial_score >= self.config.spatial_threshold:
            relation = self._create_spatial_relation(modal1, modal2, spatial_score)
            if relation:
                relations.append(relation)
        
        return relations
    
    def _calculate_temporal_relation_score(self, modal1: ModalityMetadata, modal2: ModalityMetadata) -> float:
        """Calculate temporal relationship score between two modalities."""
        if not modal1.temporal_info or not modal2.temporal_info:
            return 0.0
        
        return self.calculate_temporal_overlap(modal1.temporal_info, modal2.temporal_info)
    
    def _calculate_spatial_relation_score(self, modal1: ModalityMetadata, modal2: ModalityMetadata) -> float:
        """Calculate spatial relationship score between two modalities."""
        if not modal1.spatial_info or not modal2.spatial_info:
            return 0.0
        
        return self.calculate_spatial_proximity(modal1.spatial_info, modal2.spatial_info)
    
    def calculate_temporal_overlap(self, temporal_info1: TemporalInfo, temporal_info2: TemporalInfo) -> float:
        """
        Calculate temporal overlap/similarity between two temporal information objects.
        
        Args:
            temporal_info1, temporal_info2: Temporal information to compare
            
        Returns:
            Overlap score between 0.0 and 1.0
        """
        if not temporal_info1 or not temporal_info2:
            return 0.0
        
        # Case 1: Both have explicit time ranges
        if (temporal_info1.start_time and temporal_info1.end_time and
            temporal_info2.start_time and temporal_info2.end_time):
            return self._calculate_time_range_overlap(temporal_info1, temporal_info2)
        
        # Case 2: Both have single timestamps
        if (temporal_info1.start_time and temporal_info2.start_time and
            not temporal_info1.end_time and not temporal_info2.end_time):
            return self._calculate_timestamp_proximity(temporal_info1.start_time, temporal_info2.start_time)
        
        # Case 3: Mixed cases or time expression matching
        if self.config.enable_temporal_expression_matching:
            return self._calculate_time_expression_similarity(temporal_info1, temporal_info2)
        
        return 0.0
    
    def _calculate_time_range_overlap(self, temporal_info1: TemporalInfo, temporal_info2: TemporalInfo) -> float:
        """Calculate overlap between two time ranges."""
        start1, end1 = temporal_info1.start_time, temporal_info1.end_time
        start2, end2 = temporal_info2.start_time, temporal_info2.end_time
        
        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return 0.0  # No overlap
        
        # Calculate overlap duration
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        
        # Calculate union duration
        union_start = min(start1, start2)
        union_end = max(end1, end2)
        union_duration = (union_end - union_start).total_seconds()
        
        if union_duration == 0:
            return 1.0
        
        # Jaccard similarity for time ranges
        return overlap_duration / union_duration
    
    def _calculate_timestamp_proximity(self, time1: datetime, time2: datetime) -> float:
        """Calculate proximity between two timestamps."""
        time_diff = abs((time1 - time2).total_seconds())
        
        # Define proximity thresholds (in seconds)
        thresholds = [
            (3600, 1.0),      # Same hour: 1.0
            (86400, 0.8),     # Same day: 0.8
            (604800, 0.6),    # Same week: 0.6
            (2592000, 0.4),   # Same month: 0.4
            (31536000, 0.2)   # Same year: 0.2
        ]
        
        for threshold, score in thresholds:
            if time_diff <= threshold:
                return score
        
        return 0.0
    
    def _calculate_time_expression_similarity(self, temporal_info1: TemporalInfo, temporal_info2: TemporalInfo) -> float:
        """Calculate similarity based on time expressions."""
        expressions1 = set(temporal_info1.time_expressions) if temporal_info1.time_expressions else set()
        expressions2 = set(temporal_info2.time_expressions) if temporal_info2.time_expressions else set()
        
        if not expressions1 or not expressions2:
            return 0.0
        
        # Normalize expressions for better matching
        expressions1_norm = {self._normalize_time_expression(expr) for expr in expressions1}
        expressions2_norm = {self._normalize_time_expression(expr) for expr in expressions2}
        
        # Jaccard similarity
        intersection = expressions1_norm.intersection(expressions2_norm)
        union = expressions1_norm.union(expressions2_norm)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _normalize_time_expression(self, expression: str) -> str:
        """Normalize time expressions for better matching."""
        # Convert to lowercase and strip whitespace
        expr = expression.lower().strip()
        
        # Extract years (4 digits)
        year_match = re.search(r'\b(\d{4})\b', expr)
        if year_match:
            return f"year_{year_match.group(1)}"
        
        # Extract months
        month_patterns = [
            r'\b(january|jan)\b', r'\b(february|feb)\b', r'\b(march|mar)\b',
            r'\b(april|apr)\b', r'\b(may)\b', r'\b(june|jun)\b',
            r'\b(july|jul)\b', r'\b(august|aug)\b', r'\b(september|sep)\b',
            r'\b(october|oct)\b', r'\b(november|nov)\b', r'\b(december|dec)\b'
        ]
        
        for i, pattern in enumerate(month_patterns, 1):
            if re.search(pattern, expr):
                return f"month_{i:02d}"
        
        return expr
    
    def calculate_spatial_proximity(self, spatial_info1: SpatialInfo, spatial_info2: SpatialInfo) -> float:
        """
        Calculate spatial proximity between two spatial information objects.
        
        Args:
            spatial_info1, spatial_info2: Spatial information to compare
            
        Returns:
            Proximity score between 0.0 and 1.0
        """
        if not spatial_info1 or not spatial_info2:
            return 0.0
        
        # Case 1: Both have GPS coordinates
        if spatial_info1.coordinates and spatial_info2.coordinates and GEOPY_AVAILABLE:
            coord_score = self._calculate_coordinate_proximity(
                spatial_info1.coordinates, spatial_info2.coordinates
            )
            if coord_score > 0:
                return coord_score
        
        # Case 2: Location name matching
        if spatial_info1.locations and spatial_info2.locations:
            return self._calculate_location_name_similarity(
                spatial_info1.locations, spatial_info2.locations
            )
        
        return 0.0
    
    def _calculate_coordinate_proximity(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate proximity based on GPS coordinates."""
        if not GEOPY_AVAILABLE:
            return 0.0
        
        try:
            distance_km = geodesic(coord1, coord2).kilometers
            
            if distance_km > self.config.max_spatial_distance_km:
                return 0.0
            
            # Linear proximity: closer = higher score
            return max(0.0, 1.0 - (distance_km / self.config.max_spatial_distance_km))
            
        except Exception as e:
            logger.warning(f"Error calculating coordinate proximity: {e}")
            return 0.0
    
    def _calculate_location_name_similarity(self, locations1: List[str], locations2: List[str]) -> float:
        """Calculate similarity based on location names."""
        if not locations1 or not locations2:
            return 0.0
        
        # Normalize location names
        locations1_norm = {self._normalize_location(loc) for loc in locations1}
        locations2_norm = {self._normalize_location(loc) for loc in locations2}
        
        # Exact matching
        exact_matches = locations1_norm.intersection(locations2_norm)
        if exact_matches:
            return len(exact_matches) / max(len(locations1_norm), len(locations2_norm))
        
        # Fuzzy matching if enabled
        if self.config.enable_fuzzy_location_matching:
            return self._calculate_fuzzy_location_similarity(locations1_norm, locations2_norm)
        
        return 0.0
    
    def _normalize_location(self, location: str) -> str:
        """Normalize location names for better matching."""
        return location.lower().strip().replace(',', '').replace('.', '')
    
    def _calculate_fuzzy_location_similarity(self, locations1: set, locations2: set) -> float:
        """Calculate fuzzy similarity between location sets."""
        fuzzy_score = 0.0
        max_locations = max(len(locations1), len(locations2))
        
        for loc1 in locations1:
            for loc2 in locations2:
                # Check substring relationships
                if loc1 in loc2 or loc2 in loc1:
                    fuzzy_score += 0.5
                # Check common words
                elif self._have_common_location_words(loc1, loc2):
                    fuzzy_score += 0.3
        
        return min(1.0, fuzzy_score / max_locations) if max_locations > 0 else 0.0
    
    def _have_common_location_words(self, loc1: str, loc2: str) -> bool:
        """Check if two locations have common significant words."""
        # Split and filter common words
        stop_words = {'the', 'of', 'in', 'at', 'on', 'by', 'for', 'with', 'and', 'or'}
        
        words1 = {word for word in loc1.split() if word not in stop_words and len(word) > 2}
        words2 = {word for word in loc2.split() if word not in stop_words and len(word) > 2}
        
        return bool(words1.intersection(words2))
    
    def _create_temporal_relation(self, modal1: ModalityMetadata, modal2: ModalityMetadata, score: float) -> Optional[CrossModalRelation]:
        """Create a temporal relation between two modalities."""
        relation_id = f"temporal_{uuid.uuid4().hex[:8]}"
        
        evidence = {
            'temporal_score': score,
            'modal1_temporal': self._serialize_temporal_info(modal1.temporal_info),
            'modal2_temporal': self._serialize_temporal_info(modal2.temporal_info),
            'analysis_method': 'temporal_overlap_analysis'
        }
        
        return CrossModalRelation(
            relation_id=relation_id,
            relation_type=RelationType.TEMPORAL_RELATED,
            source_modality=modal1.modality_id,
            target_modality=modal2.modality_id,
            confidence=score,
            evidence=evidence,
            properties={
                'discovery_timestamp': datetime.now().isoformat(),
                'discoverer_version': '1.0'
            }
        )
    
    def _create_spatial_relation(self, modal1: ModalityMetadata, modal2: ModalityMetadata, score: float) -> Optional[CrossModalRelation]:
        """Create a spatial relation between two modalities."""
        relation_id = f"spatial_{uuid.uuid4().hex[:8]}"
        
        evidence = {
            'spatial_score': score,
            'modal1_spatial': self._serialize_spatial_info(modal1.spatial_info),
            'modal2_spatial': self._serialize_spatial_info(modal2.spatial_info),
            'analysis_method': 'spatial_proximity_analysis'
        }
        
        return CrossModalRelation(
            relation_id=relation_id,
            relation_type=RelationType.SPATIAL_RELATED,
            source_modality=modal1.modality_id,
            target_modality=modal2.modality_id,
            confidence=score,
            evidence=evidence,
            properties={
                'discovery_timestamp': datetime.now().isoformat(),
                'discoverer_version': '1.0'
            }
        )
    
    def _serialize_temporal_info(self, temporal_info: Optional[TemporalInfo]) -> Dict[str, Any]:
        """Serialize temporal info for evidence storage."""
        if not temporal_info:
            return {}
        
        return {
            'start_time': temporal_info.start_time.isoformat() if temporal_info.start_time else None,
            'end_time': temporal_info.end_time.isoformat() if temporal_info.end_time else None,
            'time_expressions': temporal_info.time_expressions,
            'temporal_scope': temporal_info.temporal_scope
        }
    
    def _serialize_spatial_info(self, spatial_info: Optional[SpatialInfo]) -> Dict[str, Any]:
        """Serialize spatial info for evidence storage."""
        if not spatial_info:
            return {}
        
        return {
            'locations': spatial_info.locations,
            'coordinates': spatial_info.coordinates,
            'spatial_scope': spatial_info.spatial_scope,
            'spatial_relationships': spatial_info.spatial_relationships
        }
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get a summary of the discovery process."""
        temporal_relations = [r for r in self.discovered_relations if r.relation_type == RelationType.TEMPORAL_RELATED]
        spatial_relations = [r for r in self.discovered_relations if r.relation_type == RelationType.SPATIAL_RELATED]
        
        return {
            'total_relations': len(self.discovered_relations),
            'temporal_relations': len(temporal_relations),
            'spatial_relations': len(spatial_relations),
            'avg_temporal_confidence': sum(r.confidence for r in temporal_relations) / len(temporal_relations) if temporal_relations else 0.0,
            'avg_spatial_confidence': sum(r.confidence for r in spatial_relations) / len(spatial_relations) if spatial_relations else 0.0,
            'config': {
                'temporal_threshold': self.config.temporal_threshold,
                'spatial_threshold': self.config.spatial_threshold,
                'max_spatial_distance_km': self.config.max_spatial_distance_km
            }
        }