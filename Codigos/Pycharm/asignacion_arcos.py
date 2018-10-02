import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import math


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


input_path = '../Daniel_inputs/block_model.csv'
MB = pd.read_csv(input_path)
ejex_MB = MB.sort_values(by=['xcentre'], ascending=False)[['xcentre']]
ejey_MB = MB.sort_values(by=['ycentre'], ascending=False)[['ycentre']]
ejez_MB = MB.sort_values(by=['zcentre'], ascending=False)[['zcentre']]
pos_x_MB = np.unique(ejex_MB.values)
pos_y_MB = np.unique(ejey_MB.values)
pos_z_MB = np.unique(ejez_MB.values)

# recuperar las dimensiones del modelo de bloques
# se utilizan las dimensiones en todas las direcciones
# para determinar el conjunto de aracos. Considerar
# primero una cantidad mas pequeña de bloques, i.e. de niveles
# en el eje z
N = 4  # numero de ptos pos dimension
MB_sorted = MB.sort_values(by=['zcentre'], ascending=False)
MB_sorted = MB_sorted.loc[MB_sorted['zcentre'] < pos_z_MB[N]]
MB_sorted = MB_sorted.loc[MB_sorted['xcentre'] < pos_x_MB[N]]
MB_sorted = MB_sorted.loc[MB_sorted['ycentre'] < pos_y_MB[N]]

pos_z = MB_sorted[['zcentre']].values
pos_x = MB_sorted[['xcentre']].values
pos_y = MB_sorted[['ycentre']].values

# eliminar repeticiones de las posiciones
# para determinar dimensiones del modelo
# de bloques
pos_z_unique = np.unique(pos_z)
pos_y_unique = np.unique(pos_y)
pos_x_unique = np.unique(pos_x)

# usar la notacion de lerchs-grossmann algorithm with
# variables slopes angles.
numx = np.size(pos_x_unique)
numy = np.size(pos_y_unique)
numz = np.size(pos_z_unique)

print('numx: ', numx)
print('numy: ', numy)
print('numz: ', numz)

# determinar dimensiones del modelo de bloques
zdim = pos_z_unique[1]-pos_z_unique[0]
xdim = pos_x_unique[1]-pos_x_unique[0]
ydim = pos_y_unique[1]-pos_y_unique[0]

print('xdim: ', xdim)
print('ydim: ', ydim)
print('zdim: ', zdim)

# determianr conjunto de arcos
# dados los angulos theta en las
# direcciones principales. Como primer
# ejercicio se consideraron todos iguales
theta_n = np.pi / 6
theta_s = np.pi / 6
theta_w = np.pi / 6
theta_e = np.pi / 6
arcos = []
dicc_semiejes = {}
for k in range(1, numz + 1):
    for i in range(1, numx + 1):
        for j in range(1, numy + 1):
            for t in range(1, k):

                dx1 = t * zdim / np.tan(theta_w)
                dy1 = t * zdim / np.tan(theta_s)
                dx2 = t * zdim / np.tan(theta_e)
                dy2 = t * zdim / np.tan(theta_n)

                m1 = dx1 / xdim
                n1 = dy1 / ydim
                m2 = dx2 / xdim
                n2 = dy2 / ydim
                print('###################################')
                print('nivel t=', t, '.bloque base:', (i, j, k),
                      '.semi-ejes: dx1 = %.2f, dy1 = %.2f, dx2 = %.2f, dy2 = %.2f' % (dx1, dy1, dx2, dy2))
                print('Parametros de las cantidades de bloques: m1=%.2f, n1=%.2f, m2=%.2f, n2=%.2f' % (m1, n1, m2, n2))
                for m in range(max(math.floor(i - m2), 1), min(numx, math.ceil(i + m1)) + 1):
                    for n in range(max(math.floor(j - n2), 1), min(numy, math.ceil(j + n1)) + 1):
                        print()
                        a = xdim * (i - m)
                        b = ydim * (j - n)
                        print('*********** Distancia del bloque X_{%.2f,%.2f,%.f} ***********' % (m, n, k - t))
                        print('a=%.2f, b=%.2f' % (a, b))
                        a2 = a ** 2
                        b2 = b ** 2
                        if m >= i and n >= j:
                            Value = a2 / (dx1 ** 2) + b2 / (dy1 ** 2)
                        elif m >= i and n <= j:
                            Value = a2 / (dx1 ** 2) + b2 / (dy2 ** 2)
                        elif m <= i and n <= j:
                            Value = a2 / (dx2 ** 2) + b2 / (dy2 ** 2)
                        else:
                            Value = a2 / (dx2 ** 2) + b2 / (dy1 ** 2)
                        if Value <= 1:
                            arcos.append([(i, j, k), (m, n, k - t)])

# Lo mismo de antes, pero solo predeccesor inmediato
S = {}
# inicializamos S para el primer nivel
for i in range(1, numx + 1):
    for j in range(1, numy + 1):
        S[(i, j, 1)] = []
for k in range(1, numz + 1):
    for i in range(1, numx + 1):
        for j in range(1, numy + 1):
            lista_predecesores = []
            for t in range(1, k):
                # dx1 = (k-t)*zdim/np.tan(theta_w)
                # dy1 = (k-t)*zdim/np.tan(theta_s)
                # dx2 = (k-t)*zdim/np.tan(theta_e)
                # dy2 = (k-t)*zdim/np.tan(theta_n)

                dx1 = t * zdim / np.tan(theta_w)
                dy1 = t * zdim / np.tan(theta_s)
                dx2 = t * zdim / np.tan(theta_e)
                dy2 = t * zdim / np.tan(theta_n)

                m1 = dx1 / xdim
                n1 = dy1 / ydim
                m2 = dx2 / xdim
                n2 = dy2 / ydim
                print('###################################')
                print('nivel t=', t, '.bloque base:', (i, j, k),
                      '.semi-ejes: dx1 = %.2f, dy1 = %.2f, dx2 = %.2f, dy2 = %.2f' % (dx1, dy1, dx2, dy2))
                print('Parametros de las cantidades de bloques: m1=%.2f, n1=%.2f, m2=%.2f, n2=%.2f' % (m1, n1, m2, n2))
                for m in range(max(math.floor(i - m2), 1), min(numx, math.ceil(i + m1)) + 1):
                    for n in range(max(math.floor(j - n2), 1), min(numy, math.ceil(j + n1)) + 1):
                        if m == i and n == j and t > 1:
                            continue
                        a = xdim * (i - m)
                        b = ydim * (j - n)
                        print('\n*********** Distancia del bloque X_{%.2f,%.2f,%.f} ***********' % (m, n, k - t))
                        print('a=%.2f, b=%.2f' % (a, b))
                        a2 = a ** 2
                        b2 = b ** 2
                        if m >= i and n >= j:
                            Value = a2 / (dx1 ** 2) + b2 / (dy1 ** 2)
                        elif m >= i and n <= j:
                            Value = a2 / (dx1 ** 2) + b2 / (dy2 ** 2)
                        elif m <= i and n <= j:
                            Value = a2 / (dx2 ** 2) + b2 / (dy2 ** 2)
                        else:
                            Value = a2 / (dx2 ** 2) + b2 / (dy1 ** 2)
                        if Value <= 1:
                            lista_predecesores.append((m, n, k - t))
            S[(i, j, k)] = lista_predecesores

bloque_base_1 = (2, 3, 2)  # bloque base para plotear
sacar = []
for arco in arcos:
    bloque_tail = arco[0]
    bloque_head = arco[1]
    if bloque_base_1 == bloque_tail:
        sacar.append(bloque_head)
print(sacar)

# graficar resultados de la asignacion
MB_grafico = MB_sorted[['xcentre', 'ycentre', 'zcentre']]-(min(MB_sorted[['xcentre']].values),
                                                           min(MB_sorted[['ycentre']].values),
                                                           min(MB_sorted[['zcentre']].values))
ejex = MB_grafico[['xcentre']]
ejey = MB_grafico[['ycentre']]
ejez = MB_grafico[['zcentre']]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.gca().invert_zaxis()

for v in sacar:
    a = Arrow3D([(bloque_base_1[0]-1)*20, (v[0]-1)*20], [(bloque_base_1[1]-1)*20, (v[1]-1)*20],
                [(bloque_base_1[2]-1)*15, (v[2]-1)*15], mutation_scale=20,
                lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)
ax.scatter(ejex, ejey, ejez, s=80)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
plt.gca().invert_zaxis()

# sacar_pred: lista que tiene sacar predecesores
bloque_base_2 = (3, 3, 3)
sacar_pred = [bloque for bloque in S[bloque_base_2]]
for v in sacar_pred:
    a = Arrow3D([(bloque_base_2[0]-1)*20, (v[0]-1)*20], [(bloque_base_2[1]-1)*20, (v[1]-1)*20],
                [(bloque_base_2[2]-1)*15, (v[2]-1)*15], mutation_scale=20,
                lw=3, arrowstyle="-|>", color="b")
    ax2.add_artist(a)
ax2.scatter(ejex, ejey, ejez, s=80)

# LG para MB_sorted (i.e. que tiene bastantes menos nodos)

plt.show()
