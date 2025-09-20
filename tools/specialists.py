import numpy as np
from typing import Dict, List, Tuple

# (class_tuple), (no_class_tuple), (true_map)
SpecialistSet = Tuple[
    Tuple[Dict[str, np.ndarray], int], Tuple[Dict[str, np.ndarray], int], Dict[str, int]
]

def build_specialist_set(
    features_set: Dict[str, np.ndarray],
    labels_set: List[str],
    specialist_class: str,
) -> SpecialistSet:
    class_features = {}
    true_map = {}
    no_class_features = {}

    for i, (img, features) in enumerate(features_set.items()):
        label = labels_set[i]
        if label == specialist_class:
            class_features[img] = features
            true_map[img] = 0
        else:
            no_class_features[img] = features
            true_map[img] = 1
    # Verifica se alguma lista está vazia e trata adequadamente
    len_class_features = len(class_features.values())
    len_no_class_features = len(no_class_features.values())
    print(
        f"Class {specialist_class}: {len_class_features} positive samples, {len_no_class_features} negative samples"
    )
    if len_class_features == 0 or len_no_class_features == 0:
        raise ValueError(
            f"Pelo menos um exemplar positivo e negativo para a classe {specialist_class} são necessários"
        )

    class_features_shape = next(iter(class_features.values())).shape
    no_class_features_shape = next(iter(no_class_features.values())).shape
    print("=" * 15)
    print(f"Specialist summary for class {specialist_class}")
    print(f"Class features shape: {class_features_shape}")
    print(f"Class labels shape: {len_class_features}")
    print(f"No class features shape: {no_class_features_shape}")
    print(f"No class labels shape: {len_no_class_features}")
    print("=" * 15)

    return (
        (class_features, len_class_features),
        (no_class_features, len_no_class_features),
        true_map,
    )


def build_specialist_set_for_many_classes(
    features_set: Dict[str, np.ndarray],
    labels_set: List[str],
    specialist_classes: List[str],
) -> List[SpecialistSet]:
    """
    Constrói conjuntos de especialistas para múltiplas classes.
    """
    specialist_sets = []
    for specialist_class in specialist_classes:
        specialist_set = build_specialist_set(
            features_set, labels_set, specialist_class
        )
        specialist_sets.append(specialist_set)

    return specialist_sets


def train_specialists(
    base_model,
    train_func,
    specialist_sets,
    class_names,
    model_name="Specialist",
    k_folds=5,
    verbose=False,
):
    """
    Treina modelos especialistas para cada classe usando conjuntos já preparados.

    Esta função utiliza uma abordagem de "treinamento assistido" onde cada especialista
    é avaliado usando validação cruzada e depois treinado no dataset completo da classe.

    Args:
        base_model: Modelo base já tunado para treinar (ex: GridSearchCV configurado)
        train_func: Função de treinamento assistido que aceita:
                   - base_model: modelo para treinar
                   - X, y: dados de treino
                   - title: nome do modelo
                   - k_folds: número de folds
                   - verbose: mostrar detalhes
                   E retorna: (modelo_treinado, métricas)
        specialist_sets: Lista com conjuntos (features, labels) para cada classe
                        specialist_sets[i] = (X_i, y_i) onde i é o índice da classe
        class_names: Lista com nomes das classes (ex: ["dogs", "cats", "lions"])
        model_name: Nome base do modelo para logs (ex: "KNN-LBP")
        k_folds: Número de folds para validação cruzada (default: 5)
        verbose: Se True, mostra detalhes do treinamento. Se False, apenas resumo (default: False)

    Returns:
        list: Array de modelos especialistas treinados onde specialists[i] é o
              especialista para a classe i (class_names[i])

    Example:
        >>> specialists = train_specialists(
        ...     base_model=tuned_knn,
        ...     train_func=train_model_knn,
        ...     specialist_sets=sp_lbp_sets,
        ...     class_names=["dogs", "cats", "lions"],
        ...     model_name="KNN-LBP",
        ...     k_folds=5,
        ...     verbose=False
        ... )
        >>> # specialists[0] = especialista para "dogs"
        >>> # specialists[1] = especialista para "cats"
        >>> # specialists[2] = especialista para "lions"
    """
    from joblib import parallel_backend

    specialists = []

    print(f"🚀 Iniciando treinamento de especialistas {model_name}")
    print(f"   📋 {len(specialist_sets)} especialistas para treinar")
    print(f"   🔄 Validação cruzada: {k_folds} folds")
    print(f"   📊 Modo detalhado: {'Ativado' if verbose else 'Resumo apenas'}")
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
                metrics = train_func(
                    base_model=base_model,
                    folded_dataset=dataset,
                    title=specialist_title,
                    verbose=verbose,
                )

                # Adiciona o modelo treinado ao array de especialistas
                specialists.append(metrics)

                if not verbose:
                    # Mostra resumo compacto se verbose=False
                    f1_mean, f1_std = metrics["f1"]
                    acc_mean, acc_std = metrics["accuracy"]
                    print(f"   ✅ Especialista {class_name} treinado!")
                    print(
                        f"      📈 F1: {f1_mean:.3f} ± {f1_std:.3f} | Acc: {acc_mean:.3f} ± {acc_std:.3f}"
                    )

            except Exception as e:
                print(f"   ❌ Erro ao treinar especialista {class_name}: {str(e)}")
                raise e

    print(f"\n🎉 Treinamento de especialistas {model_name} concluído!")
    print(f"   ✅ {len(specialists)} especialistas treinados com sucesso")
    print("   📦 Array retornado: specialists[i][0] = especialista para classe i")
    print(
        "   📊 Métricas de avaliação: specialists[i][1] = métricas do especialista para classe i"
    )
    print("=" * 60)

    return specialists
