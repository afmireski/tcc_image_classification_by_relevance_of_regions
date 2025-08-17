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

def train_specialists(base_model, train_func, specialist_sets, class_names, model_name="Specialist", k_folds=5, verbose=False):
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
    specialists = []
    
    print(f"🚀 Iniciando treinamento de especialistas {model_name}")
    print(f"   📋 {len(specialist_sets)} especialistas para treinar")
    print(f"   🔄 Validação cruzada: {k_folds} folds")
    print(f"   📊 Modo detalhado: {'Ativado' if verbose else 'Resumo apenas'}")
    print("-" * 60)
    
    for i, (features, labels) in enumerate(specialist_sets):
        class_name = class_names[i]
        specialist_title = f"{model_name}-Specialist-{class_name}"
        
        print(f"\n🎯 Treinando especialista {i+1}/{len(specialist_sets)}: {class_name}")
        
        if verbose:
            print(f"   📐 Features shape: {features.shape}")
            print(f"   🏷️  Labels shape: {labels.shape}")
            print(f"   📊 Distribuição de classes: {np.bincount(labels)}")
        
        # Usa a função de treinamento assistido fornecida
        try:
            specialist_model, metrics = train_func(
                base_model=base_model,
                X=features, 
                y=labels,
                title=specialist_title,
                k_folds=k_folds,
                verbose=verbose
            )
            
            # Adiciona o modelo treinado ao array de especialistas
            specialists.append((specialist_model, metrics))

            if not verbose:
                # Mostra resumo compacto se verbose=False
                f1_mean, f1_std = metrics['f1']
                acc_mean, acc_std = metrics['accuracy']
                print(f"   ✅ Especialista {class_name} treinado!")
                print(f"      📈 F1: {f1_mean:.3f} ± {f1_std:.3f} | Acc: {acc_mean:.3f} ± {acc_std:.3f}")
            
        except Exception as e:
            print(f"   ❌ Erro ao treinar especialista {class_name}: {str(e)}")
            raise e
    
    print(f"\n🎉 Treinamento de especialistas {model_name} concluído!")
    print(f"   ✅ {len(specialists)} especialistas treinados com sucesso")
    print("   📦 Array retornado: specialists[i][0] = especialista para classe i")
    print("   📊 Métricas de avaliação: specialists[i][1] = métricas do especialista para classe i")
    print("=" * 60)
    
    return specialists