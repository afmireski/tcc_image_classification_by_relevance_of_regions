import importlib
import numpy as np
from typing import List, Tuple
from sklearn.base import BaseEstimator, clone

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)

import mytypes as mtp

importlib.reload(mtp)

from mytypes import (
    PreparedSetsForClassification,
    ClassificationDataset,
    ModelResults,
    PredictResults,
    ModelMetrics,
    ModelLabels,
    RelevanceResults
)


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
    all_images = set()
    for specialist_results in specialists_results:
        all_images.update(specialist_results.keys())

    consolidated = {}

    for img in all_images:
        img_probs = []

        for results in specialists_results:
            probs = results.get(img)
            img_probs.append(probs if probs is not None else [])

        consolidated[img] = np.array(
            img_probs
        ).T  # Transpõe para ter shape (n_probabilities, n_specialists)

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
        probs = np.asarray(probs, dtype=float)
        if probs.ndim != 2:
            raise ValueError("prob_matrix deve ser 2D: (n_amostras, n_especialistas).")
        # Normaliza para somar 1.0 por amostra (linha)

        norm_probs = probs / probs.sum(axis=1, keepdims=True)

        normalized_probs[img] = norm_probs

    return normalized_probs  # Shape: (n_probabilities, n_specialists)


def shannon_entropy(probabilities: ModelResults, use_clip=False, eps=1e-12) -> ModelResults:
    """
    Calcula H(x_j) por amostra (linha) para uma matriz de probabilidades no formato (n_amostras, n_especialistas).
    H(x_j) = - sum_i P_i(x_j) * log_base(P_i(x_j)), com base = n_especialistas por padrão.

    Args:
        probabilities: dicionário {img_id: [[specialist0_prob_segment_0, specialist0_prob_segment_1, ...], [specialist1_prob_segment_0, ...], ...]}
        use_clip: Se True, usa np.clip para evitar log(0); se False, ignora zeros
        eps: Valor pequeno para np.clip

    Returns:
        entropias: dicionário {img_id: array (n_amostras,) com H(x_j) para cada amostra (linha)}
    """

    entropies = {}

    for img, probs in probabilities.items():
        P = np.asarray(probs, dtype=float)
        _, base = P.shape

        if use_clip:
            # versão "robusta", aproximação com eps
            P_safe = np.clip(P, eps, 1.0)
            logP_base = np.log(P_safe) / np.log(base)
        else:
            # versão "matemática pura", ignora zeros
            logP_base = np.zeros_like(P)
            mask = P > 0
            logP_base[mask] = np.log(P[mask]) / np.log(base)

        H = -np.sum(P * logP_base, axis=1)
        H = np.where(
            np.isclose(H, 0), 0.0, H
        )  # Remove -0.0 que apareceram em alguns casos
        entropies[img] = H

    return entropies


def shannon_entropy_manual(probabilities: ModelResults) -> ModelResults:
    """
    Calcula H(x_j) por amostra (linha) para uma matriz de probabilidades no formato (n_amostras, n_especialistas).
    H(x_j) = - sum_i P_i(x_j) * log_base(P_i(x_j)), com base = n_especialistas por padrão.

    Args:
        probabilities: dicionário {img_id: [[specialist0_prob_segment_0, specialist0_prob_segment_1, ...], [specialist1_prob_segment_0, ...], ...]}

    Returns:
        entropias: array (n_amostras,) com H(x_j) para cada amostra (linha)
    """
    entropies = {}
    for img, probs in probabilities.items():
        P = np.asarray(probs, dtype=float)
        if P.ndim != 2:
            raise ValueError("prob_matrix deve ser 2D: (n_amostras, n_especialistas).")
        _, base = P.shape

        H = []

        for probs in P:
            s = 0  # sum
            for p_i in probs:
                log_p_i = np.log(p_i) / np.log(base) if p_i > 0 else 0.0
                x = p_i * log_p_i
                s += x
            H.append(-s)
        entropies[img] = np.array(H)

    return entropies


def calculate_relevance(entropies: ModelResults) -> ModelResults:
    """
    Calcula R(x_j) para cada segmento de uma imagem a partir de suas entropias H(x_j).
    R(x_j) = 1 - H(x_j).

    Args:
        entropies: dicionário {img_id: [H(x_0), H(x_1), ..., H(x_n)]} com H(x_j) para cada amostra

    Returns:
        relevancias: dicionário {img_id: [R(x_0), R(x_1), ..., R(x_n)]} com R(x_j) para cada amostra
    """
    return {img: 1.0 - H for img, H in entropies.items()}


def calculate_max_relevance(
    relevances: ModelResults, probabilities: ModelResults
) -> ModelResults:
    """
    Calcula R_max(x_j) para as relevâncias R(x_j) de cada imagem, ponderando pela maior probabilidade entre especialistas.
    R_max(x_j) = R(x_j) * max(P(x_j)).

    Args:
        relevances: dicionário {img_id: [R(x_0), R(x_1), ..., R(x_n)]} com R(x_j) para cada segmento
        probabilities: dicionário {img_id: [[P_0(x_0), P_0(x_1), ...], [P_1(x_0), P_1(x_1), ...], ...]} com P(x_j) para cada segmento

    Returns:
        relevances: dicionário {img_id: [R_max(x_0), R_max(x_1), ..., R_max(x_n)]} com R_max(x_j) para cada amostra
    """
    result = {}
    for img, R in relevances.items():
        P = probabilities.get(img)

        # Converte para arrays numpy com dtype float para garantir compatibilidade
        R_array = np.asarray(R, dtype=float)
        P_array = np.asarray(P, dtype=float)

        # Calcula R_max = R * max(P) ao longo do eixo dos especialistas
        max_probs = P_array.max(axis=1)

        R_max = R_array * max_probs

        result[img] = R_max

    return result


def calculate_ponderate_votes(
    probabilities: ModelResults, max_relevances: ModelResults
) -> ModelResults:
    """
    Calcula votos ponderados para cada segmento de uma imagem.
    Voto ponderado = P(x_j) * R_max(x_j).

    Args:
        probabilities: dicionário {img_id: [[P_0(x_0), P_0(x_1), ...], [P_1(x_0), P_1(x_1), ...], ...]} com P(x_j) para cada amostra
        max_relevances: dicionário {img_id: [R_max(x_0), R_max(x_1), ..., R_max(x_n)]} com R_max(x_j) para cada amostra

    Returns:
        votos_ponderados: dicionário {img_id: [[V_0(x_0), V_0(x_1), ...], [V_1(x_0), V_1(x_1), ...], ...]} com votos ponderados para cada especialista
    """
    weighted_votes = {}
    for img, P in probabilities.items():
        R_max = max_relevances.get(img)

        # Converte para arrays numpy com dtype float para garantir compatibilidade
        P_array = np.asarray(P, dtype=float)
        R_max_array = np.asarray(R_max, dtype=float)

        # Calcula votos ponderados
        votes = (
            P_array * R_max_array[:, np.newaxis]
        )  # Broadcasting para multiplicar cada linha por R_max correspondente

        weighted_votes[img] = votes

    return weighted_votes


def calculate_accumulated_votes(ponderated_votes: ModelResults) -> ModelResults:
    """
    Calcula votos acumulados somando o valor do voto de todos os segmentos de um especialista.
    Voto acumulado = sum(P(x_j) * R_max(x_j)) ao longo dos especialistas.

    Args:
        ponderated_votes: dicionário {img_id: [[V_0(x_0), V_0(x_1), ...], [V_1(x_0), V_1(x_1), ...], ...]} com votos ponderados para cada especialista

    Returns:
        votos_acumulados: dicionário {img_id: [S_0, S_1, ..., S_n]} com votos acumulados para especialista
    """
    accumulated_votes = {}
    for img, votes in ponderated_votes.items():
        votes_specialists = np.asarray(votes, dtype=float)
        votes_by_pieces = (
            votes_specialists.T
        )  # Transpõe para shape (n_amostras, n_especialistas)
        accumulated = votes_by_pieces.sum(axis=1)  # Soma ao longo dos especialistas
        accumulated_votes[img] = accumulated

    return accumulated_votes


def predict_labels(accumulated_votes: ModelResults) -> PredictResults:
    """
    Determina o rótulo final de cada imagem com base nos votos acumulados.
    Rótulo = índice do especialista com maior voto acumulado.

    Args:
        accumulated_votes: dicionário {img_id: [S_0, S_1, ..., S_n]} com votos acumulados para especialista

    Returns:
        image_labels: dicionário {img_id: label} com rótulo final da imagem
    """
    image_labels = {}
    for img, votes in accumulated_votes.items():
        votes_array = np.asarray(votes, dtype=float)
        label = int(
            np.argmax(votes_array)
        )  # Índice do especialista com maior voto acumulado
        image_labels[img] = label

    return image_labels


def compute_metrics(
    true_labels: PredictResults, predicted_labels: PredictResults
) -> Tuple[ModelLabels, ModelMetrics]:
    true_y = []
    predicted_y = []

    for img, true_label in true_labels.items():
        pred_label = predicted_labels.get(img)
        true_y.append(true_label)
        predicted_y.append(pred_label)

    accuracy = accuracy_score(true_y, predicted_y)
    f1 = f1_score(true_y, predicted_y, average="macro")
    recall = recall_score(true_y, predicted_y, average="macro")
    precision = precision_score(true_y, predicted_y, average="macro")

    return (true_y, predicted_y), (accuracy, f1, recall, precision)


def relevance_technique(
    base_model: BaseEstimator,
    specialist_sets: PreparedSetsForClassification,
    class_names: List[str],
    true_labels: PredictResults,
    model_name: str = "Specialist",
    k_folds: int = 5,
) -> RelevanceResults:
    """
    Aplica a técnica de relevância para classificar imagens usando um modelo base e conjuntos de especialistas.

    Args:
        base_model: modelo base a ser utilizado para classificação
        specialist_sets: conjuntos de especialistas preparados para classificação
        class_names: nomes das classes para a tarefa de classificação
        model_name: nome do modelo a ser utilizado (padrão: "Specialist")
        k_folds: número de dobras para validação cruzada (padrão: 5)

    Returns:
        resultados: resultados da classificação
    """

    probabilities = extract_specialists_probabilities(
        base_model=base_model,
        extract_func=extract_model_results,
        specialist_sets=specialist_sets,
        class_names=class_names,
        model_name=model_name,
        k_folds=k_folds,
    )

    entropies = shannon_entropy(probabilities)

    relevances = calculate_relevance(entropies)

    max_relevances = calculate_max_relevance(relevances, probabilities)

    ponderated_votes = calculate_ponderate_votes(probabilities, max_relevances)

    accumulated_votes = calculate_accumulated_votes(ponderated_votes)

    predicted_labels = predict_labels(accumulated_votes)

    labels_list, model_metrics = compute_metrics(
        true_labels, predicted_labels
    )

    return (
        probabilities,
        entropies,
        relevances,
        max_relevances,
        ponderated_votes,
        accumulated_votes,
        predicted_labels,
        labels_list,
        model_metrics,
    )
