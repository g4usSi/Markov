from flask import Flask, render_template, request, jsonify
import numpy as np

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
    
    return jsonify({
        'success': True,
        'P_n': result['P_n'],
        'probabilities': result['probabilities'],
        'states': states
    })

if __name__ == '__main__':
    app.run(debug=True)