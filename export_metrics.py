"""
Script para exportar métricas do modelo BERT antes e depois do fine-tuning
Gera gráficos comparativos lado a lado
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# Métricas do Modelo (Resultados do Notebook)
# ============================================================

# Métricas ANTES do fine-tuning (baseline - modelo sem treinamento)
metrics_before = {
    "accuracy": 0.316,
    "precision": 0.366,
    "recall": 0.316,
    "f1": 0.294
}

# Métricas DEPOIS do fine-tuning (modelo treinado)
metrics_after = {
    "accuracy": 0.873,
    "precision": 0.873,
    "recall": 0.873,
    "f1": 0.873
}

# ============================================================
# Configurações de Estilo
# ============================================================

plt.style.use('seaborn-v0_8-whitegrid')
colors_before = '#e74c3c'  # Vermelho
colors_after = '#27ae60'   # Verde

# ============================================================
# Gráfico 1: Barras Lado a Lado (Comparação Direta)
# ============================================================

def plot_comparison_bars():
    """Gráfico de barras lado a lado - comparação direta"""
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics_names = list(metrics_before.keys())
    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, list(metrics_before.values()), width,
                   label='Antes (Baseline)', color=colors_before, edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, list(metrics_after.values()), width,
                   label='Depois (Fine-tuned)', color=colors_after, edgecolor='black', alpha=0.8)

    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title('Comparação de Métricas: Antes vs Depois do Fine-tuning', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_names], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', fontsize=10, color=colors_before, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', fontsize=10, color=colors_after, fontweight='bold')

    plt.tight_layout()
    plt.savefig('metricas_comparacao_barras.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico 1 salvo: metricas_comparacao_barras.png")


# ============================================================
# Gráfico 2: Evolução com Linhas e Markers
# ============================================================

def plot_evolution_lines():
    """Gráfico de linhas mostrando evolução das métricas"""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_names = list(metrics_before.keys())
    x_pos = np.arange(len(metrics_names))

    ax.plot(x_pos, list(metrics_before.values()), 'o-',
            label='Antes (Baseline)', color=colors_before, linewidth=3, markersize=12)
    ax.plot(x_pos, list(metrics_after.values()), 's-',
            label='Depois (Fine-tuned)', color=colors_after, linewidth=3, markersize=12)

    for i, (before, after) in enumerate(zip(metrics_before.values(), metrics_after.values())):
        improvement = ((after - before) / before) * 100
        ax.annotate(f'+{improvement:.0f}%', 
                    xy=(i, after + 0.05),
                    fontsize=10, ha='center', color='#2c3e50', fontweight='bold')

    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title('Evolução das Métricas Após Fine-tuning', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.upper() for m in metrics_names], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('metricas_evolucao_linhas.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico 2 salvo: metricas_evolucao_linhas.png")


# ============================================================
# Gráfico 3: Área Preenchida (Antes → Depois)
# ============================================================

def plot_area_filled():
    """Gráfico de área preenchida mostrando transição antes → depois"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    metrics_names = list(metrics_before.keys())

    for idx, (metric, ax) in enumerate(zip(metrics_names, axes)):
        before_val = metrics_before[metric]
        after_val = metrics_after[metric]

        x = [0, 1]
        y = [before_val, after_val]

        ax.fill_between(x, y, alpha=0.3, color=colors_after)
        ax.plot(x, y, 'o-', color=colors_after, linewidth=3, markersize=10)

        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Antes', 'Depois'], fontsize=10)
        ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Valor', fontsize=10)

        improvement = ((after_val - before_val) / before_val) * 100
        ax.text(0.5, 0.5, f'+{improvement:.0f}%',
                transform=ax.transAxes, ha='center', fontsize=14,
                fontweight='bold', color=colors_after)

    plt.suptitle('Evolução por Métrica: Antes → Depois', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('metricas_area_preenchida.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico 3 salvo: metricas_area_preenchida.png")


# ============================================================
# Gráfico 4: Tabela com Heatmap
# ============================================================

def plot_heatmap_table():
    """Tabela de métricas com heatmap de melhoria"""
    fig, ax = plt.subplots(figsize=(10, 6))

    data = {
        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Antes': [metrics_before['accuracy'], metrics_before['precision'],
                  metrics_before['recall'], metrics_before['f1']],
        'Depois': [metrics_after['accuracy'], metrics_after['precision'],
                   metrics_after['recall'], metrics_after['f1']],
    }

    df = pd.DataFrame(data)
    df['Melhoria (%)'] = ((df['Depois'] - df['Antes']) / df['Antes'] * 100).round(1)
    df['Δ (Depois - Antes)'] = (df['Depois'] - df['Antes']).round(3)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#3498db'] * len(df.columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    for i in range(len(df.columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if j == 0:
                table[(i, j)].set_text_props(fontweight='bold')

    ax.axis('off')
    ax.set_title('Tabela de Métricas: Comparação Completa', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('metricas_tabela_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico 4 salvo: metricas_tabela_heatmap.png")


# ============================================================
# Gráfico 5: Resumo Final (Dashboard Style)
# ============================================================

def plot_dashboard_summary():
    """Dashboard estilo com todos os gráficos combinados"""
    fig = plt.figure(figsize=(16, 10))

    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Gráfico 1: Barras comparativas (topo esquerdo)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_names = list(metrics_before.keys())
    x = np.arange(len(metrics_names))
    width = 0.35

    ax1.bar(x - width/2, list(metrics_before.values()), width,
            label='Antes', color=colors_before, edgecolor='black')
    ax1.bar(x + width/2, list(metrics_after.values()), width,
            label='Depois', color=colors_after, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in metrics_names])
    ax1.set_ylim(0, 1.1)
    ax1.set_title('Barras Comparativas', fontweight='bold')
    ax1.legend()
    ax1.set_ylabel('Valor')

    # Gráfico 2: Evolução de linhas (topo direito)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, list(metrics_before.values()), 'o-', label='Antes', color=colors_before, linewidth=2, markersize=8)
    ax2.plot(x, list(metrics_after.values()), 's-', label='Depois', color=colors_after, linewidth=2, markersize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in metrics_names])
    ax2.set_ylim(0, 1.1)
    ax2.set_title('Evolução de Linhas', fontweight='bold')
    ax2.legend()
    ax2.set_ylabel('Valor')

    # Gráfico 3: Barras de melhoria em % (baixo esquerdo)
    ax3 = fig.add_subplot(gs[1, 0])
    improvements = [(metrics_after[m] - metrics_before[m]) / metrics_before[m] * 100 
                     for m in metrics_before.keys()]
    bars = ax3.barh(metrics_names, improvements, color=colors_after, edgecolor='black')
    ax3.set_xlabel('Melhoria (%)')
    ax3.set_title('Melhoria por Métrica (%)', fontweight='bold')
    for bar, val in zip(bars, improvements):
        ax3.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                 f'+{val:.0f}%', va='center', fontsize=10, fontweight='bold')
    ax3.set_xlim(0, max(improvements) * 1.2)

    # Gráfico 4: Resumo numérico (baixo direito)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    summary_text = """
    RESUMO DOS RESULTADOS
    ═══════════════════════════════

    ► Accuracy:    31.6% → 87.3%  (+176%)
    ► Precision:   36.6% → 87.3%  (+139%)
    ► Recall:      31.6% → 87.3%  (+176%)
    ► F1-Score:    29.4% → 87.3%  (+196%)

    ═══════════════════════════════
    FINE-TUNING BEM SUCEDIDO! 🎉
    """

    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#3498db'))

    plt.suptitle('Dashboard: Métricas Antes vs Depois do Fine-tuning',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig('metricas_dashboard_completo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico 5 salvo: metricas_dashboard_completo.png")


# ============================================================
# Executar Todos os Gráficos
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EXPORTANDO MÉTRICAS - BERT FINE-TUNING")
    print("=" * 60)
    print()

    print("Métricas antes:", metrics_before)
    print("Métricas depois:", metrics_after)
    print()

    plot_comparison_bars()
    plot_evolution_lines()
    plot_area_filled()
    plot_heatmap_table()
    plot_dashboard_summary()

    print()
    print("=" * 60)
    print("TODOS OS GRÁFICOS FORAM SALVOS COM SUCESSO!")
    print("Arquivos gerados:")
    print("  - metricas_comparacao_barras.png")
    print("  - metricas_evolucao_linhas.png")
    print("  - metricas_area_preenchida.png")
    print("  - metricas_tabela_heatmap.png")
    print("  - metricas_dashboard_completo.png")
    print("=" * 60)