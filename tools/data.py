import numpy as np
import random

from typing import Dict, List, Tuple, TypedDict


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


def merge_categories_dicts(
    categories: List[str], textures_dict: Dict[str, Dict[str, np.ndarray]]
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    merged_dict = {}
    labels = []
    for category in categories:
        category_dict = textures_dict[category]
        n_images = 0
        for img, features in category_dict.items():
            merged_dict[img] = features
            labels.append(category)
            n_images += 1
        print(f"Categoria: {category}, Número de elementos: {n_images}")

    return (merged_dict, labels)


def combine_sets(sets: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    """
    Combina conjuntos de features em todas as combinações possíveis.

    Args:
        sets: Lista de dicionários {image_name: features_array}
        labels: Lista de nomes dos conjuntos (ex: ["LBP", "GLCM", "LPQ"])

    Returns:
        Lista com todos os conjuntos originais + todas as combinações possíveis

    Example:
        Para 3 conjuntos [LBP, GLCM, LPQ], retorna 7 conjuntos:
        [LBP, GLCM, LPQ, LBP+GLCM, LBP+LPQ, GLCM+LPQ, LBP+GLCM+LPQ]
    """
    from itertools import combinations

    output_sets = []
    n_sets = len(sets)

    # Gera todas as combinações possíveis (de 1 até n elementos)
    for r in range(1, n_sets + 1):
        for combo_indices in combinations(range(n_sets), r):
            # Combina os conjuntos selecionados
            combined_dict = {}

            # Para cada imagem, combina as features dos conjuntos selecionados
            image_names = sets[
                0
            ].keys()  # Assume que todos os conjuntos têm as mesmas imagens

            for img_name in image_names:
                combined_features = []

                # Concatena features dos conjuntos selecionados
                for idx in combo_indices:
                    features = sets[idx][img_name]
                    combined_features.append(features)

                # Concatena todas as features
                if len(combined_features) == 1:
                    # Apenas um conjunto, não precisa concatenar
                    combined_dict[img_name] = combined_features[0]
                else:
                    # Múltiplos conjuntos - concatena ao longo do eixo das features (axis=1)
                    # Mantém a estrutura de segmentos: (n_segments, n_features_combined)
                    combined_dict[img_name] = np.concatenate(combined_features, axis=1)

            output_sets.append(combined_dict)

    return output_sets


def generate_texture_dicts(
    categories: List[str],
    lbp_dict: Dict[str, Dict[str, np.ndarray]],
    glcm_dict: Dict[str, Dict[str, np.ndarray]],
    lpq_dict: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[List[Dict[str, np.ndarray]], List[str]]:
    """
    Gera dicionários de texturas mesclados e suas combinações.

    Args:
        categories: Lista de categorias
        lbp_dict: Dicionário LBP por categoria
        glcm_dict: Dicionário GLCM por categoria
        lpq_dict: Dicionário LPQ por categoria

    Returns:
        Tupla com (lista_de_conjuntos_combinados, labels)
    """
    (lbp_set, labels) = merge_categories_dicts(categories, lbp_dict)
    (glcm_set, _) = merge_categories_dicts(categories, glcm_dict)
    (lpq_set, _) = merge_categories_dicts(categories, lpq_dict)

    # Combina todos os conjuntos
    sets = [lbp_set, glcm_set, lpq_set]

    combined_sets = combine_sets(sets)

    return (combined_sets, labels)


def show_features_summary(
    combined_sets: List[Dict[str, np.ndarray]], labels: List[str]
):
    """
    Exibe um resumo das features dos conjuntos combinados.

    Args:
        combined_sets: Lista de dicionários com conjuntos combinados
        labels: Lista de labels das imagens
    """
    print("=" * 50)
    print("RESUMO DOS CONJUNTOS DE FEATURES")
    print("=" * 50)

    # Informações gerais
    total_images = len(labels)
    n_sets = len(combined_sets)

    print(f"📊 Total de imagens: {total_images}")
    print(f"🔧 Total de conjuntos gerados: {n_sets}")
    print(f"📋 Labels únicos: {set(labels)}")
    print()

    # Nomes dos conjuntos baseados na combinação
    set_names = [
        "LBP",
        "GLCM",
        "LPQ",
        "LBP+GLCM",
        "LBP+LPQ",
        "GLCM+LPQ",
        "LBP+GLCM+LPQ",
    ]

    # Para cada conjunto combinado
    for i, feature_set in enumerate(combined_sets):
        set_name = set_names[i] if i < len(set_names) else f"Conjunto_{i+1}"

        # Pega a primeira imagem para analisar a estrutura
        first_image_name = next(iter(feature_set.keys()))
        first_features = feature_set[first_image_name]

        n_images_in_set = len(feature_set)

        if len(first_features.shape) == 1:
            # Features simples (1D)
            n_features = first_features.shape[0]
            print(f"🔹 {set_name}:")
            print(f"   📁 Imagens: {n_images_in_set}")
            print(f"   🎯 Features por imagem: {n_features}")
        else:
            # Features segmentadas (2D)
            n_segments, n_features = first_features.shape
            total_features = n_segments * n_features
            total_segments = n_images_in_set * n_segments
            print(f"🔹 {set_name}:")
            print(f"   📁 Imagens: {n_images_in_set}")
            print(f"   🧩 Segmentos por imagem: {n_segments}")
            print(f"   🧱 Total de segmentos do conjunto: {total_segments}")
            print(f"   🎯 Features por segmento: {n_features}")
            print(f"   📈 Total features por imagem: {total_features}")

        print()

    print("=" * 50)


def split_data_in_folds(
    data: Tuple[
        Tuple[Dict[str, np.ndarray], int],
        Tuple[Dict[str, np.ndarray], int],
        Dict[str, int],
    ],
    k_folds=5,
    random_state=42,
) -> List[FoldData]:
    """
    Constrói dados de treino com k-folds a partir do conjunto de especialistas.

    Args:
        data: Tupla com (class_features, no_class_features, true_map)
        k_folds: Número de folds para validação cruzada
        train_factor: Fator de treino (não usado nesta implementação, mantido para compatibilidade)
        random_state: Seed para reprodutibilidade

    Returns:
        Dados organizados em variáveis separadas para posterior organização
    """

    (
        (class_features, len_class_features),
        (no_class_features, len_no_class_features),
        true_map,
    ) = data

    # Configurar seed para reprodutibilidade
    random.seed(random_state)
    np.random.seed(random_state)

    # 1. Separar chaves por classe e embaralhar
    class_images = list(class_features.keys())
    no_class_images = list(no_class_features.keys())

    # Embaralhar cada classe separadamente
    random.shuffle(class_images)
    random.shuffle(no_class_images)

    print(f"📊 Total imagens classe: {len_class_features}")
    print(f"📊 Total imagens não-classe: {len_no_class_features}")
    print(f"🔄 Dividindo em {k_folds} folds...")

    # 2. Dividir cada classe em k_folds partes aproximadamente iguais
    def dividir_em_k_partes(lista, k):
        """Divide uma lista em k partes aproximadamente iguais"""
        n = len(lista)
        tamanho_base = n // k
        resto = n % k

        partes = []
        inicio = 0

        for i in range(k):
            # Distribui o resto nas primeiras partições
            tamanho = tamanho_base + (1 if i < resto else 0)
            fim = inicio + tamanho
            partes.append(lista[inicio:fim])
            inicio = fim

        return partes

    class_folds = dividir_em_k_partes(class_images, k_folds)
    no_class_folds = dividir_em_k_partes(no_class_images, k_folds)

    # Verificar distribuição
    for i in range(k_folds):
        print(
            f"  Fold {i}: {len(class_folds[i])} classe, {len(no_class_folds[i])} não-classe"
        )

    # 3. Construir cada fold
    folds_data = []

    # print("=" * 20)
    # print(class_images)

    for k in range(k_folds):
        print(f"\n🔧 Construindo fold {k}...")

        # Teste = parte k de cada classe
        test_class_images = class_folds[k]
        test_no_class_images = no_class_folds[k]

        # Treino = todas as outras partes
        train_class_images = []
        train_no_class_images = []

        for i in range(k_folds):
            if i != k:  # Excluir a parte usada para teste
                train_class_images.extend(class_folds[i])
                train_no_class_images.extend(no_class_folds[i])

        # 4. Construir dicionários para este fold
        # Dados de treino
        train_class_dict = {img: class_features[img] for img in train_class_images}
        train_no_class_dict = {
            img: no_class_features[img] for img in train_no_class_images
        }
        train_true_map = {}

        # Adicionar ao mapeamento verdadeiro
        for img in train_class_images:
            train_true_map[img] = true_map[img]
        for img in train_no_class_images:
            train_true_map[img] = true_map[img]

        # Dados de teste
        test_class_dict = {img: class_features[img] for img in test_class_images}
        test_no_class_dict = {
            img: no_class_features[img] for img in test_no_class_images
        }
        test_true_map = {}

        # Adicionar ao mapeamento verdadeiro
        for img in test_class_images:
            test_true_map[img] = true_map[img]
        for img in test_no_class_images:
            test_true_map[img] = true_map[img]

        # print("+" * 20)
        # print(train_class_dict.keys())
        # print("-" * 20)
        # print(test_class_dict.keys())
        # print("+" * 20)

        # Armazenar dados do fold em variáveis separadas
        fold_data = {
            "fold_id": k,
            "train_class_features": train_class_dict,
            "train_no_class_features": train_no_class_dict,
            "train_true_map": train_true_map,
            "test_class_features": test_class_dict,
            "test_no_class_features": test_no_class_dict,
            "test_true_map": test_true_map,
            "train_class_count": len(train_class_images),
            "train_no_class_count": len(train_no_class_images),
            "test_class_count": len(test_class_images),
            "test_no_class_count": len(test_no_class_images),
            "train_total": len(train_class_images) + len(train_no_class_images),
            "test_total": len(test_class_images) + len(test_no_class_images),
        }

        folds_data.append(fold_data)

        print(
            f"  ✅ Treino: {len(train_class_images)} classe + {len(train_no_class_images)} não-classe = {fold_data['train_total']}"
        )
        print(
            f"  ✅ Teste: {len(test_class_images)} classe + {len(test_no_class_images)} não-classe = {fold_data['test_total']}"
        )

    print(f"\n🎉 {k_folds} folds construídos com sucesso!")
    # print("=" * 20)

    # Retornar dados em variáveis separadas para fase 2
    return folds_data


def _extract_features_and_labels(
    class_features: Dict[str, np.ndarray],
    no_class_features: Dict[str, np.ndarray],
    true_map: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]]:
    """
    Extrai features e labels de dicionários de classe e não-classe.

    Args:
        class_features: Dicionário com features da classe
        no_class_features: Dicionário com features não-classe
        true_map: Mapeamento de imagem para label verdadeiro

    Returns:
        ClassificationData com X, y e features_map
    """
    # Combinar todos os dicionários de features
    all_features = {**class_features, **no_class_features}

    # Listas para acumular features e labels
    X_list = []
    y_list = []
    features_map = {}

    current_position = 0

    # Processar cada imagem
    for img_name, features in all_features.items():
        # Obter o label verdadeiro da imagem
        label = true_map[img_name]

        # Features pode ser 1D (features simples) ou 2D (features segmentadas)
        if len(features.shape) == 1:
            # Features simples - uma linha por imagem
            n_segments = 1
            features_2d = features.reshape(1, -1)
        else:
            # Features segmentadas - múltiplas linhas por imagem
            n_segments = features.shape[0]
            features_2d = features

        # Adicionar features ao array X
        X_list.append(features_2d)

        # Adicionar labels para todos os segmentos desta imagem
        y_list.extend([label] * n_segments)

        # Registrar posições no mapa
        start_pos = current_position
        end_pos = current_position + n_segments - 1

        features_map[img_name] = (start_pos, end_pos)

        current_position = end_pos + 1

    # Concatenar todas as features
    X = np.vstack(X_list)
    y = np.array(y_list)

    return (X, y, features_map)


def build_classification_data(
    folded_data: List[FoldData],
) -> List[
    Tuple[
        Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]],
        Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]],
    ]
]:
    """
    Converte dados de folds em formato adequado para classificação.

    Para cada fold:
    1. Extrai features de cada imagem para um array sequencial (X)
    2. Em paralelo, popula um array de rótulos (y) com a classe da imagem
    3. Salva em um dicionário o mapa de posições de cada imagem no array
    4. Faz isso para treino e teste

    Args:
        folded_data: Lista de folds com dados brutos

    Returns:
        Lista de folds processados com dados formatados para classificação
    """
    processed_folds = []

    print("🔄 Convertendo folds para formato de classificação...")

    for fold_data in folded_data:
        fold_id = fold_data["fold_id"]
        print(f"\n📂 Processando fold {fold_id}...")

        # Processar dados de treino
        print("  🏋️ Processando dados de treino...")
        train_data = _extract_features_and_labels(
            fold_data["train_class_features"],
            fold_data["train_no_class_features"],
            fold_data["train_true_map"],
        )

        # Processar dados de teste
        print("  🧪 Processando dados de teste...")
        test_data = _extract_features_and_labels(
            fold_data["test_class_features"],
            fold_data["test_no_class_features"],
            fold_data["test_true_map"],
        )

        # Criar fold processado
        processed_fold = (train_data, test_data)

        processed_folds.append(processed_fold)

        # Log informativo
        train_X_shape = train_data[0].shape
        test_X_shape = test_data[0].shape
        train_images = len(train_data[2])
        test_images = len(test_data[2])

        print(
            f"  ✅ Treino: {train_images} imagens → X{train_X_shape}, y{train_data[1].shape}"
        )
        print(
            f"  ✅ Teste: {test_images} imagens → X{test_X_shape}, y{test_data[1].shape}"
        )

    print(f"\n🎉 {len(processed_folds)} folds processados com sucesso!")

    return processed_folds


def prepare_sets_for_classification(
    sets: List[
        Tuple[
            Tuple[Dict[str, np.ndarray], int],
            Tuple[Dict[str, np.ndarray], int],
            Dict[str, int],
        ]
    ],
    k_folds=5,
    random_state=42,
) -> List[
    List[
        Tuple[
            Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]],
            Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]],
        ]
    ]
]:
    """
    Prepara os conjuntos de dados para classificação, dividindo-os em folds e extraindo as características.

    Args:
        sets: Lista de conjuntos de dados a serem preparados.
        k_folds: Número de folds para a validação cruzada.
        random_state: Semente para reprodutibilidade.

    Returns:
        Lista de conjuntos de dados preparados para classificação.
    """

    data = []
    for dataset in sets:
        folds = split_data_in_folds(dataset, k_folds=k_folds, random_state=random_state)
        classification_data = build_classification_data(folds)
        data.append(classification_data)

    return data