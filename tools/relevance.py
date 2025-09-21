import importlib
import numpy as np
from typing import List
from sklearn.base import BaseEstimator, clone

import mytypes as mtp
importlib.reload(mtp)

from mytypes import PreparedSetsForClassification, ClassificationDataset, ModelResults


def extract_model_results(
    base_model: BaseEstimator, folded_dataset: ClassificationDataset, title: str = ""
) -> ModelResults:
    """
    Extrai probabilidades positivas de um modelo usando validação cruzada.

    Args:
        base_model: Modelo sklearn configurado (ex: GridSearchCV)
        folded_dataset: Lista de tuplas ((X_train, y_train), (X_test, y_test))
        title: Nome do modelo para logs

    Returns:
        Array 1D com probabilidades da classe positiva (0) para todas as amostras
    """

    probabilities = {}

    # Validação cruzada para avaliação
    print(f"=== Iniciando treinamento assistido: {title} ===")
    print(f"Validação cruzada: {len(folded_dataset)} folds")
    print("-" * 50)

    # Validação cruzada para avaliação
    for train_set, test_set in folded_dataset:
        X_train, y_train, train_pieces_map = train_set
        X_test, y_test, test_pieces_map = test_set

        # Clona o modelo base para cada fold
        fold_model = clone(base_model)
        fold_model.fit(X_train, y_train)

        print("Melhores parâmetros:", fold_model.best_params_)

        predict = fold_model.predict_proba(X_test)

        for i, p in enumerate(predict):
            img = test_pieces_map[i]

            if probabilities.get(img) is None:
                probabilities[img] = []

            probabilities[img].append(p[0])  # Probabilidade da classe positiva (0)

    print("-" * 50)
    return probabilities

def consolidate_model_results(specialists_results: List[ModelResults]) -> ModelResults:
    """
    Consolida resultados de múltiplos folds em um único dicionário.

    Args:
        specialists_results: Lista de dicionários {img_id: [prob_segment_0, prob_segment_1, ...]}

    Returns:
        Dicionário consolidado {img_id: [[specialist0_prob_segment_0, specialist0_prob_segment_1, ...], [specialist1_prob_segment_0, ...], ...]}
    """
    consolidated = {}

    for results in specialists_results:
        for img, probs in results.items():
            if consolidated.get(img) is None:
                consolidated[img] = np.array(probs)
            else:
                consolidated[img] = np.vstack((consolidated[img], probs))
        consolidated[img] = consolidated[img].T # Transpõe para ter shape (n_probabilities, n_specialists)

    return consolidated


def extract_specialists_probabilities(
    base_model: BaseEstimator,
    extract_func: callable,
    specialist_sets: PreparedSetsForClassification,
    class_names: List[str],
    model_name: str = "Specialist",
    k_folds: int = 5,
) -> ModelResults:
    """
    Treina múltiplos especialistas e extrai suas probabilidades normalizadas.

    Args:
        base_model: Modelo base para treinar especialistas
        extract_func: Função para extrair probabilidades (ex: extract_model_results)
        specialist_sets: Lista de datasets divididos em folds para cada especialista
        class_names: Nomes das classes ['dogs', 'cats', 'lions', 'horses']
        model_name: Nome base para logs
        k_folds: Número de folds

    Returns:
        Matriz (n_amostras, n_especialistas) com probabilidades normalizadas
    """
    from joblib import parallel_backend

    extracted_probabilities = []

    print(f"🚀 Iniciando treinamento de especialistas {model_name}")
    print(f"   📋 {len(specialist_sets)} especialistas para treinar")
    print(f"   🔄 Validação cruzada: {k_folds} folds")
    print("-" * 60)

    # Use threading backend para evitar problemas de memory mapping
    with parallel_backend("threading"):
        for i, dataset in enumerate(specialist_sets):
            class_name = class_names[i]
            specialist_title = f"{model_name}-Specialist-{class_name}"

            print(
                f"\n🎯 Treinando especialista {i+1}/{len(specialist_sets)}: {class_name}"
            )

            # Usa a função de treinamento assistido fornecida
            try:
                specialist_probabilities = extract_func(
                    base_model=base_model,
                    folded_dataset=dataset,
                    title=specialist_title,
                )

                # Adiciona o modelo treinado ao array de especialistas
                extracted_probabilities.append(specialist_probabilities)

            except Exception as e:
                print(
                    f"   ❌ Erro ao extrair probabilidades do especialista {class_name}: {str(e)}"
                )
                raise e

    print(f"\n🎉 Treinamento de especialistas {model_name} concluído!")
    print(f"   ✅ {len(extracted_probabilities)} especialistas treinados com sucesso")
    print(
        "   📦 Array retornado: raw_probabilities[i] = probabilidades do especialista da classe i"
    )
    print("=" * 60)

    images_probabilities = consolidate_model_results(extracted_probabilities)

    return normalize_probabilities(images_probabilities)


def normalize_probabilities(probabilities: ModelResults) -> ModelResults:
    """
    Normaliza probabilidades de especialistas para somarem 1.0 por amostra.
    """

    normalized_probs = {}

    for img, probs in probabilities.items():
        normalized_probs[img] = probs / probs.sum(axis=1, keepdims=True)


    return normalized_probs  # Shape: (n_probabilities, n_specialists)


def shannon_entropy(probabilities: ModelResults) -> ModelResults:
    """
    Calcula H(x_j) por amostra (linha) para uma matriz de probabilidades no formato (n_amostras, n_especialistas).
    H(x_j) = - sum_i P_i(x_j) * log_base(P_i(x_j)), com base = n_especialistas por padrão.

    Args:
        prob_matrix: array (n_amostras, n_especialistas), cada linha soma ~1.0

    Returns:
        entropias das imagens: array (n_amostras,) com H(x_j) para cada amostra (linha)
    """
    entropies = {}
    for img, probs in probabilities.items():
        P = np.asarray(probs, dtype=float)
        if P.ndim != 2:
            raise ValueError("prob_matrix deve ser 2D: (n_amostras, n_especialistas).")
        print(f"Shape: {P.shape}")
        _, base = P.shape

        # log na base desejada
        logP_base = np.log(P) / np.log(base)

        # soma sobre especialistas (colunas), resultando em entropia por amostra (linhas)
        H = -np.sum(P * logP_base, axis=1)
        entropies[img] = H

    return entropies


def shannon_entropy_manual(probabilities: ModelResults) -> ModelResults:
    """
    Calcula H(x_j) por amostra (linha) para uma matriz de probabilidades no formato (n_amostras, n_especialistas).
    H(x_j) = - sum_i P_i(x_j) * log_base(P_i(x_j)), com base = n_especialistas por padrão.

    Args:
        prob_matrix: array (n_amostras, n_especialistas), cada linha soma ~1.0

    Returns:
        entropias: array (n_amostras,) com H(x_j) para cada amostra (linha)
    """
    entropies = {}
    for img, probs in probabilities.items():
        P = np.asarray(probs, dtype=float)
        if P.ndim != 2:
            raise ValueError("prob_matrix deve ser 2D: (n_amostras, n_especialistas).")
        print(f"Shape: {P.shape}")
        _, base = P.shape

        H = []

        for probs in P:
            s = 0  # sum
            for p_i in probs:
                log_p_i = np.log(p_i) / np.log(base)
                x = p_i * log_p_i
                s += x
            H.append(-s)
        entropies[img] = np.array(H)

    return entropies


def calculate_relevance(entropies: np.array) -> np.array:
    """
    Calcula R(x_j) para um vetor de entropias H(x_j)
    R(x_j) = 1 - H(x_j).

    Args:
        entropies: array (n_amostras,) com H(x_j) para cada amostra

    Returns:
        relevancias: array (n_amostras,) com R(x_j) para cada amostra
    """
    return 1.0 - entropies


def calculate_max_relevance(relevances: np.array, prob_matrix: np.ndarray) -> np.array:
    """
    Calcula R_max(x_j) para um vetor de relevâncias R(x_j)
    R_max(x_j) = R(x_j) * max(P(x_j)).

    Args:
        relevances: array (n_amostras,) com R(x_j) para cada amostra
        prob_matrix: array (n_amostras, n_especialistas) com P(x_j) para cada amostra

    Returns:
        relevances: array (n_amostras,) com R_max(x_j) para cada amostra
    """
    return relevances * prob_matrix.max(axis=1)
