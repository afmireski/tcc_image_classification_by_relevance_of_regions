import numpy as np

from typing import Dict, List, Tuple, TypedDict

SpecialistSet = Tuple[
    Tuple[Dict[str, np.ndarray], int], Tuple[Dict[str, np.ndarray], int], Dict[str, int]
]


class FoldData(TypedDict):
    """Estrutura de dados para cada fold de validação cruzada"""

    fold_id: int
    train_class_features: Dict[str, np.ndarray]
    train_no_class_features: Dict[str, np.ndarray]
    train_true_map: Dict[str, int]
    test_class_features: Dict[str, np.ndarray]
    test_no_class_features: Dict[str, np.ndarray]
    test_true_map: Dict[str, int]
    train_class_count: int
    train_no_class_count: int
    test_class_count: int
    test_no_class_count: int
    train_total: int
    test_total: int


ClassificationData = Tuple[np.ndarray, np.ndarray, Dict[int, str]]

# Aliases para tipos complexos
ClassificationFold = Tuple[
    ClassificationData,  # dados de treino
    ClassificationData,  # dados de teste
]

ClassificationDataset = List[ClassificationFold]

PreparedSetsForClassification = List[ClassificationDataset]

ModelResults = Dict[str, np.ndarray]  # {img_id: [prob_segment_0, prob_segment_1, ...]}

PredictResults = Dict[str, int]  # {model_name: predicted_class}

ModelMetrics = Tuple[float, float, float, float]  # (accuracy, f1, recall, precision)

ModelLabels = Tuple[List[int], List[int]]  # (true_labels, predicted_labels)

RelevanceResults = Tuple[
    ModelResults,
    ModelResults,
    ModelResults,
    ModelResults,
    ModelResults,
    ModelResults,
    PredictResults,
    ModelLabels,
    ModelMetrics,
]
