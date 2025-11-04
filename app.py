from flask import Flask, render_template, request, jsonify
import numpy as np
import networkx as nx
import matplotlib

matplotlib.use('Agg')  # Para evitar problemas con el backend gráfico
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)


def validate_matrix(matrix):
    for row in matrix:
        if not np.isclose(sum(row), 1.0):
            return False
    return True


def calculate_markov(matrix, initial_state, steps):
    P = np.array(matrix, dtype=float)

    if not validate_matrix(P.tolist()):
        return None, "Cada fila debe sumar 1.0"

    P_n = np.linalg.matrix_power(P, steps)

    initial_vector = np.zeros(len(P))
    initial_vector[initial_state] = 1.0

    final_probs = initial_vector @ P_n

    return {
        'P_n': P_n.tolist(),
        'probabilities': final_probs.tolist()
    }, None


def create_markov_graph(matrix, states):
    """Crea un grafo de Markov y lo convierte a imagen base64"""
    G = nx.DiGraph()

    # Agregar nodos
    for i, state in enumerate(states):
        G.add_node(i, label=state)

    # Agregar aristas con probabilidades
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            prob = matrix[i][j]
            if prob > 0:  # Solo agregar aristas con probabilidad > 0
                G.add_edge(i, j, weight=prob, label=f"{prob:.2f}")

    # Crear figura más grande para mejor visualización
    plt.figure(figsize=(12, 10))

    # Usar un layout circular para mejor visualización
    pos = nx.circular_layout(G)

    # Dibujar nodos más grandes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue',
                           alpha=0.9, edgecolors='black', linewidths=2)

    # Dibujar etiquetas de nodos con mejor formato
    labels = {i: states[i] for i in range(len(states))}
    nx.draw_networkx_labels(G, pos, labels, font_size=11, font_weight='bold',
                            font_family='sans-serif', bbox=dict(boxstyle="round,pad=0.3",
                                                                facecolor="white", edgecolor="black", alpha=0.8))

    # Dibujar aristas
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    nx.draw_networkx_edges(G, pos, edgelist=edges,
                           width=3, alpha=0.7, edge_color='darkblue',
                           arrows=True, arrowsize=25,
                           connectionstyle="arc3,rad=0.1")

    # Dibujar etiquetas de probabilidades
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                 font_size=9, font_weight='bold',
                                 bbox=dict(boxstyle="round,pad=0.2",
                                           facecolor="white", edgecolor="none", alpha=0.7))

    plt.title("Grafo de la Cadena de Markov", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()

    # Convertir a base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{image_base64}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    matrix = data.get('matrix')
    initial_state = int(data.get('initial_state', 0))
    steps = int(data.get('steps', 5))
    states = data.get('states', ['Estado 1', 'Estado 2', 'Estado 3'])

    result, error = calculate_markov(matrix, initial_state, steps)

    if error:
        return jsonify({'success': False, 'error': error})

    # Generar el grafo de Markov
    graph_image = create_markov_graph(matrix, states)

    return jsonify({
        'success': True,
        'P_n': result['P_n'],
        'probabilities': result['probabilities'],
        'states': states,
        'markov_graph': graph_image
    })


if __name__ == '__main__':
    app.run(debug=True)