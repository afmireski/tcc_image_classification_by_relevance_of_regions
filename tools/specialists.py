import numpy as np

def build_specialist_set(features_set, labels_set, specialist_class):
    class_features = []
    no_class_features = []

    for i, label in enumerate(labels_set):
        if label == specialist_class:
            class_features.append(features_set[i])
        else:
            no_class_features.append(features_set[i])

    # Verifica se alguma lista está vazia e trata adequadamente
    if len(class_features) == 0 and len(no_class_features) == 0:
        raise ValueError(f"Nenhum dado encontrado para a classe {specialist_class}")
    
    if len(class_features) == 0:
        # Se não há exemplares da classe, retorna apenas os dados de "não-classe"
        no_class_features = np.array(no_class_features)
        no_class_labels = np.array([1] * len(no_class_features))
        print("=" * 15)
        print(f"AVISO: Classe {specialist_class} não possui exemplares!")
        print("Retornando apenas dados de não-classe")
        print(f"No class features shape: {no_class_features.shape}")
        print(f"No class labels shape: {no_class_labels.shape}")
        print("=" * 15)
        return (no_class_features, no_class_labels)
    
    if len(no_class_features) == 0:
        # Se não há exemplares de outras classes, retorna apenas os dados da classe
        class_features = np.array(class_features)
        class_labels = np.array([0] * len(class_features))
        print("=" * 15)
        print(f"AVISO: Apenas classe {specialist_class} possui exemplares!")
        print(f"Class features shape: {class_features.shape}")
        print(f"Class labels shape: {class_labels.shape}")
        print("=" * 15)
        return (class_features, class_labels)

    # Caso normal: ambas as listas têm dados
    class_features = np.array(class_features)
    no_class_features = np.array(no_class_features)

    class_labels = np.array([0] * len(class_features))
    no_class_labels = np.array([1] * len(no_class_features))

    # Concatena features e labels ao longo do eixo das amostras (axis=0)
    features = np.concatenate((class_features, no_class_features), axis=0)
    labels = np.concatenate((class_labels, no_class_labels), axis=0)

    print("=" * 15)
    print(f"Specialist summary for class {specialist_class}")
    print(f"Class features shape: {class_features.shape}")
    print(f"Class labels shape: {class_labels.shape}")
    print(f"No class features shape: {no_class_features.shape}")
    print(f"No class labels shape: {no_class_labels.shape}")
    print(f"Final features shape: {features.shape}")
    print(f"Final labels shape: {labels.shape}")
    print("=" * 15)

    return (features, labels)

def build_specialist_set_for_many_classes(features_set, labels_set, specialist_classes):
    specialist_sets = []
    for specialist_class in specialist_classes:
        specialist_set = build_specialist_set(features_set, labels_set, specialist_class)
        specialist_sets.append(specialist_set)

    return specialist_sets