import numpy as np
from typing import List, Tuple
from sklearn.base import BaseEstimator


def extract_model_results(
    base_model: BaseEstimator, 
    folded_dataset: List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]], 
    title: str = ""
) -> np.ndarray:
    """
    Extrai probabilidades positivas de um modelo usando validação cruzada.
    
    Args:
        base_model: Modelo sklearn configurado (ex: GridSearchCV)
        folded_dataset: Lista de tuplas ((X_train, y_train), (X_test, y_test))
        title: Nome do modelo para logs
        
    Returns:
        Array 1D com probabilidades da classe positiva (0) para todas as amostras
    """    
    from sklearn.base import clone
    
    probabilities = np.array([])
    
    # Validação cruzada para avaliação
    print(f"=== Iniciando treinamento assistido: {title} ===")
    print(f"Validação cruzada: {len(folded_dataset)} folds")
    print("-" * 50)
        
    
    # Validação cruzada para avaliação
    for (X_train, y_train), (X_test, y_test) in folded_dataset:

        # Clona o modelo base para cada fold
        fold_model = clone(base_model)
        fold_model.fit(X_train, y_train)

        print("Melhores parâmetros:", fold_model.best_params_)

        predict = fold_model.predict_proba(X_test)

        positive_probabilities = [p[0] for p in predict]
        probabilities = np.append(probabilities, positive_probabilities)

    print("-" * 50)
    return probabilities


def extract_specialists_probabilities(
    base_model: BaseEstimator,
    extract_func: callable,
    specialist_sets: List[List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]],
    class_names: List[str],
    model_name: str = "Specialist",
    k_folds: int = 5
) -> np.ndarray:
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
    
    raw_probabilities = []
    
    print(f"🚀 Iniciando treinamento de especialistas {model_name}")
    print(f"   📋 {len(specialist_sets)} especialistas para treinar")
    print(f"   🔄 Validação cruzada: {k_folds} folds")
    print("-" * 60)
    
    # Use threading backend para evitar problemas de memory mapping
    with parallel_backend('threading'):
        for i, dataset in enumerate(specialist_sets):
            class_name = class_names[i]
            specialist_title = f"{model_name}-Specialist-{class_name}"
            
            print(f"\n🎯 Treinando especialista {i+1}/{len(specialist_sets)}: {class_name}")
                    
            # Usa a função de treinamento assistido fornecida
            try:
                specialist_probabilities = extract_func(
                    base_model=base_model,
                    folded_dataset=dataset,
                    title=specialist_title
                )
                
                # Adiciona o modelo treinado ao array de especialistas
                raw_probabilities.append(specialist_probabilities)

                
            except Exception as e:
                print(f"   ❌ Erro ao extrair probabilidades do especialista {class_name}: {str(e)}")
                raise e
    
    print(f"\n🎉 Treinamento de especialistas {model_name} concluído!")
    print(f"   ✅ {len(raw_probabilities)} especialistas treinados com sucesso")
    print("   📦 Array retornado: raw_probabilities[i] = probabilidades do especialista da classe i")
    print("=" * 60)
    
    return normalize_probabilities(raw_probabilities)

def normalize_probabilities(raw_probabilities: List[np.ndarray]) -> np.ndarray:
    """
    Normaliza probabilidades de especialistas para somarem 1.0 por amostra.
    
    Args:
        raw_probabilities: Lista de arrays 1D, um por especialista
        
    Returns:
        Matriz (n_amostras, n_especialistas) onde cada linha soma 1.0
        
    Example:
        Input: [[0.1, 0.2], [0.4, 0.5]]  # 2 especialistas, 2 amostras
        Output: [[0.2, 0.8], [0.29, 0.71]]  # normalizado por linha
    """
    # Converte a lista de arrays em uma matriz 2D
    prob_matrix = np.array(raw_probabilities)  # Shape: (n_specialists, n_probabilities)

    # Transpõe para ter shape (n_probabilities, n_specialists)
    prob_matrix = prob_matrix.T  # Shape: (n_probabilities, n_specialists)
    
    # Normaliza as probabilidades para cada amostra
    normalized_probs = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
    
    return normalized_probs  # Shape: (n_probabilities, n_specialists)

def shannon_entropy(prob_matrix: np.ndarray) -> np.ndarray:
    """
    Calcula H(x_j) por amostra (linha) para uma matriz de probabilidades no formato (n_amostras, n_especialistas).
    H(x_j) = - sum_i P_i(x_j) * log_base(P_i(x_j)), com base = n_especialistas por padrão.

    Args:
        prob_matrix: array (n_amostras, n_especialistas), cada linha soma ~1.0

    Returns:
        entropias: array (n_amostras,) com H(x_j) para cada amostra (linha)
    """
    P = np.asarray(prob_matrix, dtype=float)
    if P.ndim != 2:
        raise ValueError("prob_matrix deve ser 2D: (n_amostras, n_especialistas).")
    print(f'Shape: {P.shape}')
    _, base = P.shape

    # log na base desejada
    logP_base = np.log(P) / np.log(base)

    # soma sobre especialistas (colunas), resultando em entropia por amostra (linhas)
    H = -np.sum(P * logP_base, axis=1)
    return H


def shannon_entropy_manual(prob_matrix: np.ndarray) -> np.ndarray:
    """
    Calcula H(x_j) por amostra (linha) para uma matriz de probabilidades no formato (n_amostras, n_especialistas).
    H(x_j) = - sum_i P_i(x_j) * log_base(P_i(x_j)), com base = n_especialistas por padrão.

    Args:
        prob_matrix: array (n_amostras, n_especialistas), cada linha soma ~1.0

    Returns:
        entropias: array (n_amostras,) com H(x_j) para cada amostra (linha)
    """
    P = np.asarray(prob_matrix, dtype=float)
    if P.ndim != 2:
        raise ValueError("prob_matrix deve ser 2D: (n_amostras, n_especialistas).")
    print(f'Shape: {P.shape}')
    _, base = P.shape

    H = []

    for probs in P:
        s = 0 # sum
        for p_i in probs:
            log_p_i = np.log(p_i) / np.log(base)
            x = (p_i * log_p_i)
            s += x
        H.append(-s)

    return np.array(H)

def relevance(entropies: np.array) -> np.array:
    """
    Calcula R(x_j) para um vetor de entropias H(x_j)
    R(x_j) = 1 - H(x_j).

    Args:
        entropies: array (n_amostras,) com H(x_j) para cada amostra

    Returns:
        relevancias: array (n_amostras,) com R(x_j) para cada amostra
    """
    return 1.0 - entropies

def max_relevance(relevances: np.array, prob_matrix: np.ndarray) -> np.array:
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