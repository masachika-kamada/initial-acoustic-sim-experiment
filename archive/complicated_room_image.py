import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Room A
A_vertices = np.array([[0, 0, 0],
                       [10, 0, 0],
                       [10, 10, 0],
                       [0, 10, 0],
                       [0, 0, 10],
                       [10, 0, 10],
                       [10, 10, 10],
                       [0, 10, 10]])

A_faces = [[A_vertices[j] for j in face] for face in [[0, 1, 5, 4],
                                                      [1, 2, 6, 5],
                                                      [2, 3, 7, 6],
                                                      [3, 0, 4, 7],
                                                      [0, 1, 2, 3],
                                                      [4, 5, 6, 7]]]

# Room B
B_vertices = np.array([[5, -1, 0],
                       [6, -1, 0],
                       [6, 0, 0],
                       [5, 0, 0],
                       [5, -1, 1],
                       [6, -1, 1],
                       [6, 0, 1],
                       [5, 0, 1]])

B_faces = [[B_vertices[j] for j in face] for face in [[0, 1, 5, 4],
                                                      [1, 2, 6, 5],
                                                      [2, 3, 7, 6],
                                                      [3, 0, 4, 7],
                                                      [0, 1, 2, 3],
                                                      [4, 5, 6, 7]]]

# Passage
P_vertices = np.array([[5, -0.1, 0.4],
                       [5, -0.1, 0.6],
                       [5, 0.1, 0.6],
                       [5, 0.1, 0.4],
                       [5.1, -0.1, 0.4],
                       [5.1, -0.1, 0.6],
                       [5.1, 0.1, 0.6],
                       [5.1, 0.1, 0.4]])

P_faces = [[P_vertices[j] for j in face] for face in [[0, 1, 5, 4],
                                                      [1, 2, 6, 5],
                                                      [2, 3, 7, 6],
                                                      [3, 0, 4, 7],
                                                      [0, 1, 2, 3],
                                                      [4, 5, 6, 7]]]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.add_collection3d(Poly3DCollection(A_faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
ax.add_collection3d(Poly3DCollection(B_faces, facecolors='green', linewidths=1, edgecolors='r', alpha=.25))
ax.add_collection3d(Poly3DCollection(P_faces, facecolors='red', linewidths=1, edgecolors='r', alpha=.25))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([0, 14])
ax.set_ylim([-2, 12])
ax.set_zlim([0, 14])

ax.view_init(90, -90)

plt.show()
