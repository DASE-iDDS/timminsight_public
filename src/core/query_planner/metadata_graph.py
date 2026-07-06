"""
Multimodal metadata graph for cross-modal data analysis and query planning.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import pickle
from datetime import datetime
from collections import defaultdict
import networkx as nx

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """TiMMInsight系统支持的模态类型枚举。
    
    定义了系统能够处理的不同数据模态类型，每种模态都有专门的处理器
    来提取其特定的元数据和语义特征。
    """
    IMAGE = "image"    # 图像模态：包含视觉内容，支持对象检测、场景识别等
    TEXT = "text"      # 文本模态：包含文字内容，支持实体识别、情感分析等  
    TABLE = "table"    # 表格模态：包含结构化数据，支持统计分析、关系发现等


class EntityType(Enum):
    """Cross-modal entity types."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    OBJECT = "OBJECT"
    TIME = "TIME"
    QUANTITY = "QUANTITY"


class RelationType(Enum):
    """Types of relationships between modalities."""
    # CONTAINS = "contains"
    # DESCRIBES = "describes"
    # QUANTIFIES = "quantifies"
    # TEMPORAL_OVERLAP = "temporal_overlap"
    # SPATIAL_COLOCATED = "spatial_colocated"
    # CAUSAL = "causal"
    # SIMILAR_TO = "similar_to"
    # PART_OF = "part_of"
    # REFERENCES = "references"

    # 时空维度关系
    TEMPORAL_RELATED = "temporal_related"    # 时间相关性
    SPATIAL_RELATED = "spatial_related"      # 空间相关性

    # 语义内容关系
    REPRESENTS = "represents"                # 表示/描述关系
    CONTAINS = "contains"                    # 包含关系
    BELONGS_TO = "belongs_to"               # 从属关系

    # 逻辑结构关系
    REFERENCES = "references"                # 引用/指向关系
    SIMILAR_TO = "similar_to"               # 相似性关系

    # 处理转换关系
    DERIVED_FROM = "derived_from"           # 派生/生成关系
    ASSOCIATED_WITH = "associated_with"     # 关联关系


@dataclass
class EntityMention:
    """表示实体在特定模态中的一次出现记录。
    
    这是跨模态实体链接的基本单元，记录同一实体在不同模态中的具体表现形式。
    在TiMMInsight系统的步骤1-2（元数据提取与图构建）中生成，
    为步骤3（查询澄清）提供实体级别的语义理解支持。
    """
    modality_id: str                                     # 所属模态的唯一标识符，对应ModalityMetadata.modality_id
                                                         # 建立实体提及与模态数据间的关联
    
    modality_type: ModalityType                          # 模态类型（图像/文本/表格/音频/视频）
                                                         # 用于区分实体在不同模态中的表现形式
    
    mention_text: str                                    # 实体在该模态中的文本表示
                                                         # 图像：检测到的对象标签（如"person", "car"）
                                                         # 文本：实际提及的文字（如"李明", "Apple Inc."）
                                                         # 表格：单元格中的数据值
    
    position: Optional[Dict[str, Any]] = None            # 实体在模态中的位置信息（方便定位和验证）
                                                         # 图像：{"bbox": [x1,y1,x2,y2], "center": [x,y]}
                                                         # 文本：{"start": 10, "end": 15, "sentence": 2}
                                                         # 表格：{"row": 3, "column": "name", "sheet": "data"}
    
    confidence: float = 1.0                              # 该次实体提及的置信度分数（0.0-1.0）
                                                         # 反映实体识别算法的准确性，用于结果排序和质量控制
    
    context: Optional[str] = None                        # 实体提及的上下文信息（提供语义消歧支持）
                                                         # 文本：周围句子或段落
                                                         # 表格：相关列或行的信息
                                                         # 图像：场景描述或其他对象信息


@dataclass
class CrossModalEntity:
    """表示跨多个模态出现的统一实体。
    
    这是TiMMInsight系统实现真正跨模态数据理解的核心抽象。通过聚合同一实体
    在不同模态中的所有提及，实现统一的实体表示和跨模态关系发现。
    
    在系统的步骤3（查询澄清）中，系统基于这些跨模态实体信息理解用户查询意图，
    生成针对性的澄清问题，如"您说的是哪个李明？"、"您关心的是这个球队的哪个方面？"。
    """
    entity_id: str                                       # 跨模态实体的唯一标识符，全局唯一
                                                         # 命名规则："entity_" + 实体类型 + "_" + 规范化名称
                                                         # 如："entity_person_leonardo", "entity_location_beijing"
    
    entity_type: EntityType                              # 实体类型枚举值，支持的类型包括：
                                                         # PERSON(人物), ORGANIZATION(组织), LOCATION(地点)
                                                         # EVENT(事件), CONCEPT(概念), OBJECT(物体)
                                                         # TIME(时间), QUANTITY(数量)
    
    canonical_name: str                                  # 实体的规范化名称（标准化表示）
                                                         # 用于统一不同模态中的同一实体的不同表达
                                                         # 如："李明" vs "Li Ming" vs "Ming Li"
                                                         # 对于查询澄清和结果展示至关重要
    
    mentions: List[EntityMention] = field(default_factory=list)  # 该实体在所有模态中的出现记录列表
                                                                 # 每个EntityMention记录一次具体的出现
                                                                 # 支持按模态类型、置信度等维度进行筛选和分析
    
    semantic_embedding: Optional[np.ndarray] = None      # 实体的语义嵌入向量表示（高维数组）
                                                         # 用于计算实体间的语义相似度和关系强度
                                                         # 支持基于语义的智能查询和推荐系统
    
    confidence_score: float = 1.0                        # 该跨模态实体的整体置信度分数（0.0-1.0）
                                                         # 基于所有mentions的置信度和跨模态一致性计算
                                                         # 用于查询结果排序和不确定性传播
    
    properties: Dict[str, Any] = field(default_factory=dict)    # 实体的扩展属性信息（灵活扩展）
                                                                # 可包含：别名列表、描述信息、关联网址等
                                                                # 如：{"aliases": ["达芬奇"], "birth_year": "1452"}

    def add_mention(self, mention: EntityMention):
        """添加一个实体提及记录。"""
        self.mentions.append(mention)

    def get_mentions_by_modality(self, modality_type: ModalityType) -> List[EntityMention]:
        """获取该实体在特定模态类型中的所有提及记录。"""
        return [m for m in self.mentions if m.modality_type == modality_type]


@dataclass
class TemporalInfo:
    """从内容中提取的时间信息。
    
    用于表示模态内容的时间维度特征，支持时间范围查询和时序关系分析。
    在多模态查询澄清阶段，帮助系统理解用户查询的时间约束。
    """
    start_time: Optional[datetime] = None                    # 内容的开始时间（如事件发生时间、文档创建时间）
    end_time: Optional[datetime] = None                      # 内容的结束时间（如事件结束时间、有效期限）
    time_expressions: List[str] = field(default_factory=list)  # 提取的时间表达式列表（如"昨天"、"2023年"）
    temporal_scope: Optional[str] = None                     # 时间范围类型："past"(过去), "present"(现在), "future"(未来)


@dataclass
class SpatialInfo:
    """从内容中提取的空间信息。
    
    用于表示模态内容的空间维度特征，支持地理位置查询和空间关系分析。
    在多模态查询澄清阶段，帮助系统理解用户查询的地理约束。
    """
    locations: List[str] = field(default_factory=list)         # 提取的地点名称列表（如"北京"、"清华大学"）
    coordinates: Optional[Tuple[float, float]] = None           # GPS坐标（纬度，经度）
    spatial_scope: Optional[str] = None                         # 空间范围类型："local"(本地), "regional"(区域), "global"(全球)
    spatial_relationships: List[str] = field(default_factory=list)  # 空间关系描述（如"附近"、"包含"、"相邻"）


@dataclass
class QualityMetrics:
    """模态内容的质量度量指标。
    
    用于评估模态数据的质量，指导查询优化和结果排序。
    所有指标范围为0.0-1.0，1.0表示最高质量。
    """
    completeness: float = 1.0     # 完整性：数据是否完整，无缺失值或损坏
    accuracy: float = 1.0         # 准确性：提取的元数据与实际内容的匹配度
    consistency: float = 1.0      # 一致性：与其他模态数据的一致性程度
    timeliness: float = 1.0       # 时效性：数据的新鲜度和时效性
    overall_quality: float = 1.0  # 综合质量：基于以上指标计算的总体质量分数


@dataclass
class ModalityMetadata:
    """所有模态类型的统一元数据结构。
    
    这是TiMMInsight系统的核心数据结构，用于统一表示不同模态（图像、文本、表格等）
    的元数据信息。设计遵循系统的6步工作流程，特别支持步骤1（元数据提取）和
    步骤2（元数据图构建）的需求。
    
    在查询澄清阶段（步骤3），系统会基于这些元数据信息生成针对性的澄清问题，
    帮助用户明确查询意图。
    """
    modality_id: str                                            # 模态的唯一标识符，用于在元数据图中定位和引用
    modality_type: ModalityType                                 # 模态类型（图像/文本/表格/音频/视频）
    source_path: str                                            # 源数据文件路径，记录原始数据位置
    
    content_summary: Dict[str, Any] = field(default_factory=dict)     # 内容摘要信息，包含核心内容的结构化描述
                                                                       # 如：图像的对象列表、文本的主题、表格的统计信息
    
    semantic_features: Dict[str, Any] = field(default_factory=dict)   # 语义特征向量和高级语义信息
                                                                       # 如：CLIP特征、词嵌入、语义标签等
    
    entities: List[str] = field(default_factory=list)         # 关联的实体ID列表，用于跨模态实体链接
                                                               # 支持基于实体的查询和关系发现
    
    temporal_info: Optional[TemporalInfo] = None               # 时间维度信息，支持时序查询和时间关系分析
    spatial_info: Optional[SpatialInfo] = None                 # 空间维度信息，支持地理查询和空间关系分析
    quality_metrics: Optional[QualityMetrics] = None           # 质量度量指标，用于查询优化和结果排序
    
    extracted_at: datetime = field(default_factory=datetime.now)      # 元数据提取时间戳，记录处理时间
    processor_version: str = "1.0"                             # 处理器版本，用于兼容性管理和结果重现
    raw_metadata: Dict[str, Any] = field(default_factory=dict)        # 原始元数据，保存处理器提取的所有原始信息


@dataclass
class CrossModalRelation:
    """Represents a relationship between modalities."""
    relation_id: str
    relation_type: RelationType
    source_modality: str
    target_modality: str
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)


class MultimodalMetadataGraph:
    """
    Main class for managing multimodal metadata and cross-modal relationships.

    This class provides a unified view of metadata across different modalities
    and enables complex query planning and relationship discovery.
    """

    def __init__(self, graph_id: str = None):
        """Initialize the metadata graph."""
        self.graph_id = graph_id or f"metadata_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Core data structures
        self.modalities: Dict[str, ModalityMetadata] = {}
        self.entities: Dict[str, CrossModalEntity] = {}
        self.relations: Dict[str, CrossModalRelation] = {}

        # Graph structures for efficient querying
        self.modality_graph = nx.MultiDiGraph()
        self.entity_graph = nx.Graph()

        # Indices for fast lookup
        self.modality_by_type: Dict[ModalityType, List[str]] = defaultdict(list)
        self.entities_by_type: Dict[EntityType, List[str]] = defaultdict(list)
        self.relations_by_type: Dict[RelationType, List[str]] = defaultdict(list)

        # Statistics
        self.stats = {
            'total_modalities': 0,
            'total_entities': 0,
            'total_relations': 0,
            'modality_distribution': {},
            'entity_distribution': {},
            'relation_distribution': {}
        }

        logger.info(f"Initialized MultimodalMetadataGraph: {self.graph_id}")

    def add_modality(self, metadata: ModalityMetadata) -> None:
        """Add a modality to the graph."""
        self.modalities[metadata.modality_id] = metadata
        self.modality_by_type[metadata.modality_type].append(metadata.modality_id)

        # Add to graph
        self.modality_graph.add_node(
            metadata.modality_id,
            modality_type=metadata.modality_type,
            metadata=metadata
        )

        self._update_stats()
        logger.debug(f"Added modality {metadata.modality_id} of type {metadata.modality_type}")

    def add_entity(self, entity: CrossModalEntity) -> None:
        """Add a cross-modal entity to the graph."""
        self.entities[entity.entity_id] = entity
        self.entities_by_type[entity.entity_type].append(entity.entity_id)

        # Add to entity graph
        self.entity_graph.add_node(
            entity.entity_id,
            entity_type=entity.entity_type,
            entity=entity
        )

        # Link entity to modalities
        for mention in entity.mentions:
            if mention.modality_id in self.modalities:
                self.modalities[mention.modality_id].entities.append(entity.entity_id)

        self._update_stats()
        logger.debug(f"Added entity {entity.entity_id} of type {entity.entity_type}")

    def add_relation(self, relation: CrossModalRelation) -> None:
        """Add a cross-modal relation to the graph."""
        self.relations[relation.relation_id] = relation
        self.relations_by_type[relation.relation_type].append(relation.relation_id)

        # Add to modality graph
        self.modality_graph.add_edge(
            relation.source_modality,
            relation.target_modality,
            relation_id=relation.relation_id,
            relation_type=relation.relation_type,
            confidence=relation.confidence,
            relation=relation
        )

        self._update_stats()
        logger.debug(f"Added relation {relation.relation_id} of type {relation.relation_type}")

    def get_modalities_by_type(self, modality_type: ModalityType) -> List[ModalityMetadata]:
        """Get all modalities of a specific type."""
        modality_ids = self.modality_by_type.get(modality_type, [])
        return [self.modalities[mid] for mid in modality_ids]

    def get_entities_by_type(self, entity_type: EntityType) -> List[CrossModalEntity]:
        """Get all entities of a specific type."""
        entity_ids = self.entities_by_type.get(entity_type, [])
        return [self.entities[eid] for eid in entity_ids]

    def get_relations_by_type(self, relation_type: RelationType) -> List[CrossModalRelation]:
        """Get all relations of a specific type."""
        relation_ids = self.relations_by_type.get(relation_type, [])
        return [self.relations[rid] for rid in relation_ids]

    def find_connected_modalities(self, modality_id: str, max_hops: int = 2) -> List[Tuple[str, int]]:
        """Find modalities connected to the given modality within max_hops."""
        if modality_id not in self.modality_graph:
            return []

        connected = []
        for target, path_length in nx.single_source_shortest_path_length(
            self.modality_graph.to_undirected(), modality_id, cutoff=max_hops
        ).items():
            if target != modality_id:
                connected.append((target, path_length))

        return connected

    def find_common_entities(self, modality_ids: List[str]) -> List[CrossModalEntity]:
        """Find entities that appear in all specified modalities."""
        if not modality_ids:
            return []

        # Get entities for each modality
        modality_entities = []
        for mid in modality_ids:
            if mid in self.modalities:
                entities = set(self.modalities[mid].entities)
                modality_entities.append(entities)

        if not modality_entities:
            return []

        # Find intersection
        common_entity_ids = set.intersection(*modality_entities)
        return [self.entities[eid] for eid in common_entity_ids if eid in self.entities]

    def get_semantic_similarity(self, modality_id1: str, modality_id2: str) -> float:
        """Calculate semantic similarity between two modalities."""
        if modality_id1 not in self.modalities or modality_id2 not in self.modalities:
            return 0.0

        meta1 = self.modalities[modality_id1]
        meta2 = self.modalities[modality_id2]

        # Simple semantic similarity based on common entities
        entities1 = set(meta1.entities)
        entities2 = set(meta2.entities)

        if not entities1 or not entities2:
            return 0.0

        intersection = entities1.intersection(entities2)
        union = entities1.union(entities2)

        return len(intersection) / len(union) if union else 0.0

    def query_by_entity(self, entity_name: str, entity_type: EntityType = None) -> List[ModalityMetadata]:
        """Query modalities that contain a specific entity."""
        matching_modalities = []

        for entity in self.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue

            if entity.canonical_name.lower() == entity_name.lower():
                for mention in entity.mentions:
                    if mention.modality_id in self.modalities:
                        matching_modalities.append(self.modalities[mention.modality_id])

        return matching_modalities

    def query_by_temporal_range(self, start_time: datetime, end_time: datetime) -> List[ModalityMetadata]:
        """Query modalities within a temporal range."""
        matching_modalities = []

        for metadata in self.modalities.values():
            if metadata.temporal_info:
                temp_info = metadata.temporal_info
                if (temp_info.start_time and temp_info.start_time >= start_time and temp_info.start_time <= end_time) or \
                   (temp_info.end_time and temp_info.end_time >= start_time and temp_info.end_time <= end_time):
                    matching_modalities.append(metadata)

        return matching_modalities

    def query_by_spatial_location(self, location: str) -> List[ModalityMetadata]:
        """Query modalities related to a specific location."""
        matching_modalities = []

        for metadata in self.modalities.values():
            if metadata.spatial_info and location.lower() in [loc.lower() for loc in metadata.spatial_info.locations]:
                matching_modalities.append(metadata)

        return matching_modalities

    def get_relationship_paths(self, source_modality: str, target_modality: str) -> List[List[str]]:
        """Find all relationship paths between two modalities."""
        if source_modality not in self.modality_graph or target_modality not in self.modality_graph:
            return []

        try:
            paths = list(nx.all_simple_paths(
                self.modality_graph,
                source_modality,
                target_modality,
                cutoff=3
            ))
            return paths
        except nx.NetworkXNoPath:
            return []

    def export_graph(self, file_path: str, format: str = "json") -> None:
        """Export the metadata graph to a file."""
        export_data = {
            'graph_id': self.graph_id,
            'modalities': {mid: self._serialize_modality(meta) for mid, meta in self.modalities.items()},
            'entities': {eid: self._serialize_entity(entity) for eid, entity in self.entities.items()},
            'relations': {rid: self._serialize_relation(rel) for rid, rel in self.relations.items()},
            'stats': self.stats,
            'exported_at': datetime.now().isoformat()
        }

        if format.lower() == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        elif format.lower() == "pickle":
            with open(file_path, 'wb') as f:
                pickle.dump(export_data, f)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported metadata graph to {file_path}")

    def load_graph(self, file_path: str, format: str = "json") -> None:
        """Load a metadata graph from a file."""
        if format.lower() == "json":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif format.lower() == "pickle":
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported load format: {format}")

        self.graph_id = data['graph_id']

        # Load modalities
        for mid, meta_data in data['modalities'].items():
            metadata = self._deserialize_modality(meta_data)
            self.add_modality(metadata)

        # Load entities
        for eid, entity_data in data['entities'].items():
            entity = self._deserialize_entity(entity_data)
            self.add_entity(entity)

        # Load relations
        for rid, rel_data in data['relations'].items():
            relation = self._deserialize_relation(rel_data)
            self.add_relation(relation)

        logger.info(f"Loaded metadata graph from {file_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the metadata graph."""
        return {
            'graph_id': self.graph_id,
            'statistics': self.stats,
            'modality_types': list(self.modality_by_type.keys()),
            'entity_types': list(self.entities_by_type.keys()),
            'relation_types': list(self.relations_by_type.keys()),
            'graph_connectivity': {
                'modality_nodes': self.modality_graph.number_of_nodes(),
                'modality_edges': self.modality_graph.number_of_edges(),
                'entity_nodes': self.entity_graph.number_of_nodes(),
                'entity_edges': self.entity_graph.number_of_edges()
            }
        }

    def _update_stats(self) -> None:
        """Update graph statistics."""
        self.stats['total_modalities'] = len(self.modalities)
        self.stats['total_entities'] = len(self.entities)
        self.stats['total_relations'] = len(self.relations)

        # Distribution statistics
        self.stats['modality_distribution'] = {
            str(mtype): len(mids) for mtype, mids in self.modality_by_type.items()
        }
        self.stats['entity_distribution'] = {
            str(etype): len(eids) for etype, eids in self.entities_by_type.items()
        }
        self.stats['relation_distribution'] = {
            str(rtype): len(rids) for rtype, rids in self.relations_by_type.items()
        }

    def _serialize_modality(self, metadata: ModalityMetadata) -> Dict[str, Any]:
        """Serialize modality metadata for export."""
        return {
            'modality_id': metadata.modality_id,
            'modality_type': metadata.modality_type.value,
            'source_path': metadata.source_path,
            'content_summary': metadata.content_summary,
            'semantic_features': metadata.semantic_features,
            'entities': metadata.entities,
            'temporal_info': metadata.temporal_info.__dict__ if metadata.temporal_info else None,
            'spatial_info': metadata.spatial_info.__dict__ if metadata.spatial_info else None,
            'quality_metrics': metadata.quality_metrics.__dict__ if metadata.quality_metrics else None,
            'extracted_at': metadata.extracted_at.isoformat(),
            'processor_version': metadata.processor_version,
            'raw_metadata': metadata.raw_metadata
        }

    def _serialize_entity(self, entity: CrossModalEntity) -> Dict[str, Any]:
        """Serialize entity for export."""
        return {
            'entity_id': entity.entity_id,
            'entity_type': entity.entity_type.value,
            'canonical_name': entity.canonical_name,
            'mentions': [mention.__dict__ for mention in entity.mentions],
            'semantic_embedding': entity.semantic_embedding.tolist() if entity.semantic_embedding is not None else None,
            'confidence_score': entity.confidence_score,
            'properties': entity.properties
        }

    def _serialize_relation(self, relation: CrossModalRelation) -> Dict[str, Any]:
        """Serialize relation for export."""
        return {
            'relation_id': relation.relation_id,
            'relation_type': relation.relation_type.value,
            'source_modality': relation.source_modality,
            'target_modality': relation.target_modality,
            'confidence': relation.confidence,
            'evidence': relation.evidence,
            'properties': relation.properties
        }

    def _deserialize_modality(self, data: Dict[str, Any]) -> ModalityMetadata:
        """Deserialize modality metadata from export data."""
        temporal_info = None
        if data['temporal_info']:
            temporal_info = TemporalInfo(**data['temporal_info'])

        spatial_info = None
        if data['spatial_info']:
            spatial_info = SpatialInfo(**data['spatial_info'])

        quality_metrics = None
        if data['quality_metrics']:
            quality_metrics = QualityMetrics(**data['quality_metrics'])

        return ModalityMetadata(
            modality_id=data['modality_id'],
            modality_type=ModalityType(data['modality_type']),
            source_path=data['source_path'],
            content_summary=data['content_summary'],
            semantic_features=data['semantic_features'],
            entities=data['entities'],
            temporal_info=temporal_info,
            spatial_info=spatial_info,
            quality_metrics=quality_metrics,
            extracted_at=datetime.fromisoformat(data['extracted_at']),
            processor_version=data['processor_version'],
            raw_metadata=data['raw_metadata']
        )

    def _deserialize_entity(self, data: Dict[str, Any]) -> CrossModalEntity:
        """Deserialize entity from export data."""
        mentions = [EntityMention(**mention_data) for mention_data in data['mentions']]
        semantic_embedding = np.array(data['semantic_embedding']) if data['semantic_embedding'] else None

        return CrossModalEntity(
            entity_id=data['entity_id'],
            entity_type=EntityType(data['entity_type']),
            canonical_name=data['canonical_name'],
            mentions=mentions,
            semantic_embedding=semantic_embedding,
            confidence_score=data['confidence_score'],
            properties=data['properties']
        )

    def _deserialize_relation(self, data: Dict[str, Any]) -> CrossModalRelation:
        """Deserialize relation from export data."""
        return CrossModalRelation(
            relation_id=data['relation_id'],
            relation_type=RelationType(data['relation_type']),
            source_modality=data['source_modality'],
            target_modality=data['target_modality'],
            confidence=data['confidence'],
            evidence=data['evidence'],
            properties=data['properties']
        )
    
    def build_cross_modal_relations(self, config=None) -> int:
        """
        Automatically build cross-modal relations using the relation discoverer.
        
        Args:
            config: Optional configuration for relation discovery
            
        Returns:
            Number of relations added to the graph
        """
        try:
            from .relation_discoverer import CrossModalRelationDiscoverer, RelationDiscoveryConfig
            
            # Initialize discoverer with config
            discoverer_config = config or RelationDiscoveryConfig()
            discoverer = CrossModalRelationDiscoverer(discoverer_config)
            
            # Discover relations
            relations = discoverer.discover_relations(self)
            
            # Add relations to the graph
            relations_added = 0
            for relation in relations:
                self.add_relation(relation)
                relations_added += 1
            
            logger.info(f"Built {relations_added} cross-modal relations")
            
            # Log discovery summary
            summary = discoverer.get_discovery_summary()
            logger.info(f"Relation discovery summary: {summary}")
            
            return relations_added
            
        except ImportError as e:
            logger.error(f"Failed to import relation discoverer: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error building cross-modal relations: {e}")
            return 0
    
    def build_entity_relations(self, config=None) -> int:
        """
        Automatically build entity relationships using the entity relationship discoverer.
        
        Args:
            config: Optional configuration for entity relationship discovery
            
        Returns:
            Number of entity relations added to the graph
        """
        try:
            from .entity_relationship_discoverer import EntityRelationshipDiscoverer, EntityRelationConfig
            
            # Initialize discoverer with config
            discoverer_config = config or EntityRelationConfig()
            discoverer = EntityRelationshipDiscoverer(discoverer_config)
            
            # Discover entity relationships
            entity_relations = discoverer.discover_entity_relations(self)
            
            # Add relations to the graph
            relations_added = 0
            for relation in entity_relations:
                self.add_relation(relation)
                relations_added += 1
            
            logger.info(f"Built {relations_added} entity relationships")
            
            # Log discovery summary
            summary = discoverer.get_entity_relationship_summary()
            logger.info(f"Entity relationship discovery summary: {summary}")
            
            return relations_added
            
        except ImportError as e:
            logger.error(f"Failed to import entity relationship discoverer: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error building entity relationships: {e}")
            return 0
    
    def build_all_relations(self, modal_config=None, entity_config=None) -> Dict[str, int]:
        """
        Build both cross-modal and entity relationships.
        
        Args:
            modal_config: Configuration for cross-modal relation discovery
            entity_config: Configuration for entity relationship discovery
            
        Returns:
            Dictionary with counts of different relation types built
        """
        results = {
            'cross_modal_relations': 0,
            'entity_relations': 0,
            'total_relations': 0
        }
        
        try:
            # Build cross-modal relations (temporal/spatial)
            results['cross_modal_relations'] = self.build_cross_modal_relations(modal_config)
            
            # Build entity relationships
            results['entity_relations'] = self.build_entity_relations(entity_config)
            
            results['total_relations'] = results['cross_modal_relations'] + results['entity_relations']
            
            logger.info(f"Built all relations: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error building all relations: {e}")
            return results