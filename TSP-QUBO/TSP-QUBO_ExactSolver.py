import csv

import dimod
import numpy as np
from dimod import AdjDictBQM
from dwave.cloud import Client

NUM_NODES = 5
alpha = 1.0
beta = 0.01

nodesDistancesMatrix = np.zeros(shape=(NUM_NODES, NUM_NODES), dtype='float')
with open('NodesDistances.csv', 'r') as ficheroCSV:
    distancesReader = csv.reader(ficheroCSV, delimiter=';')

    for rowIdx, row in enumerate(distancesReader):
        for column in range(NUM_NODES):
            if rowIdx < NUM_NODES:
                nodesDistancesMatrix[rowIdx][column] = float(row[column])

print('Successfully filled the edges weights matrix with the file data')
# print(nodesDistancesMatrix)

_QUBOdictionary = AdjDictBQM('BINARY')

with open('QUBO_conf_matrix.csv', 'w') as file_CSV:
    writer = csv.writer(file_CSV, delimiter='\t')

    _QUBO_configuration_matrix = np.zeros(shape=(NUM_NODES ** 2, NUM_NODES ** 2), dtype='float')
    for row in range(NUM_NODES ** 2):
        rowData = []
        for column in range(NUM_NODES ** 2):

            # if row > column:
            #     rowData.append(0.0)
            #     continue

            node_X = row // NUM_NODES
            position_X = row % NUM_NODES
            node_Y = column // NUM_NODES
            position_Y = column % NUM_NODES

            if row == column:
                _QUBO_configuration_matrix[row][column] = alpha * -2
                rowData.append(_QUBO_configuration_matrix[row][column])
                _QUBOdictionary.add_variable("X{0},{1}".format(node_X, position_X),
                                             _QUBO_configuration_matrix[row][column])

            elif node_X == node_Y and position_X != position_Y:  # Second check is unnecessary because it is already
                # discarded since row != column and nodes are the same
                # one. Positions cannot be equal too at this point.
                _QUBO_configuration_matrix[row][column] = alpha * 2  # Penalty for the condition: just one position for
                # each node.
                rowData.append(_QUBO_configuration_matrix[row][column])
                _QUBOdictionary.set_quadratic("X{0},{1}".format(node_X, position_X),
                                              "X{0},{1}".format(node_Y, position_Y),
                                              _QUBO_configuration_matrix[row][column])

            elif position_X == position_Y and node_X != node_Y:  # Second check is unnecessary because it is already
                # discarded since row != column and positions are the
                # same one. Nodes cannot be equal too at this point.
                _QUBO_configuration_matrix[row][column] = alpha * 2  # Penalty for the condition: just one position for
                # each node
                rowData.append(_QUBO_configuration_matrix[row][column])
                _QUBOdictionary.set_quadratic("X{0},{1}".format(node_X, position_X),
                                              "X{0},{1}".format(node_Y, position_Y),
                                              _QUBO_configuration_matrix[row][column])

            elif abs(position_X - position_Y) % (
                    NUM_NODES - 2) == 1:  # If nodes are adjacent in the path...  The module
                # operation consider the returning edge from the last position to the first one
                _QUBO_configuration_matrix[row][column] = beta * nodesDistancesMatrix[node_X][node_Y]  # Penalty for the
                # situation in which to nodes are adjacent in the path.
                # In that case, edge weight between the two nodes needs
                # to be added as an energy penalty.
                rowData.append(_QUBO_configuration_matrix[row][column])
                _QUBOdictionary.set_quadratic("X{0},{1}".format(node_X, position_X),
                                              "X{0},{1}".format(node_Y, position_Y),
                                              _QUBO_configuration_matrix[row][column])
            else:
                rowData.append(0.0)
                _QUBOdictionary.set_quadratic("X{0},{1}".format(node_X, position_X),
                                              "X{0},{1}".format(node_Y, position_Y), 0.0)

        writer.writerow(rowData)

# print(_QUBO_configuration_matrix)
print('Successfully filled the matrix with the problem QUBO modeling')
print('Successfully filled the dictionary with the problem QUBO modeling')

quboDict = _QUBOdictionary.to_qubo()[0]  # print(quboDict)

# Connect using the default or environment connection information
with Client.from_config() as client:
    # Load the default solver
    solver = client.get_solver()

    print(quboDict)

    import time
    start_time = time.time()
    computation = dimod.ExactSolver().sample_qubo(quboDict)
    print('Execution time for {0} nodes: {1} milliseconds'.format(NUM_NODES, (time.time() - start_time)*1000))

    # Print results
    # print("Solution: sample={.samples.first}".format(final_state))
    # print(computation)
    print(computation.first.energy)
    with open('less-energyExactSolver.csv', 'w') as file_CSV:
        writer = csv.writer(file_CSV)
        rowData = ['Energy: {0}'.format(computation.first.energy),
                   'Lowest energy solution: {0}'.format(computation.first)]
        writer.writerow(rowData)

# # Print the first sample out of a hundred
# show(final_state)
