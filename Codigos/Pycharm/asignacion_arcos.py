import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import math
import time
import copy as cp


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
#input_path = '../Blasor_inputs_fases/BINPHA_OA20.csv'
#MB = pd.read_csv(input_path, sep=';', usecols=['xcentre', 'ycentre', 'zcentre'], dtype=float, decimal=',')
MB = pd.read_csv(input_path)
print(list(MB))
ejex_MB = MB.sort_values(by=['xcentre'], ascending=False)[['xcentre']]
ejey_MB = MB.sort_values(by=['ycentre'], ascending=False)[['ycentre']]
ejez_MB = MB.sort_values(by=['zcentre'], ascending=False)[['zcentre']]
pos_x_MB = np.unique(ejex_MB.values)
pos_y_MB = np.unique(ejey_MB.values)
pos_z_MB = np.unique(ejez_MB.values)

# recuperar las dimensiones del modelo de bloques
# se utilizan las dimensiones en todas las direcciones
# para determinar el conjunto de arcos. Considerar
# primero una cantidad mas pequeña de bloques, i.e. de niveles
# en el eje z
#N = 10
# Ejemplo 1
# Nx = 4  # numero de ptos pos dimension
# Ny = 1
# Nz = 2
# Ejemplo 2
Nx = 6  # numero de ptos pos dimension
Ny = 1
Nz = 3
MB_sorted = MB.sort_values(by=['zcentre'], ascending=False)
# MB_sorted = MB_sorted.loc[MB_sorted['zcentre'] < pos_z_MB[N]]
# MB_sorted = MB_sorted.loc[MB_sorted['xcentre'] < pos_x_MB[N]]
# MB_sorted = MB_sorted.loc[MB_sorted['ycentre'] < pos_y_MB[N]]

MB_sorted = MB_sorted.loc[MB_sorted['zcentre'] < pos_z_MB[Nz]]
MB_sorted = MB_sorted.loc[MB_sorted['xcentre'] < pos_x_MB[Nx]]
MB_sorted = MB_sorted.loc[MB_sorted['ycentre'] < pos_y_MB[Ny]]

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
#ydim = pos_y_unique[1]-pos_y_unique[0]
ydim = 20
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
# for k in range(1, numz + 1):
#     for i in range(1, numx + 1):
#         for j in range(1, numy + 1):
#             for t in range(1, k):
#
#                 dx1 = t * zdim / np.tan(theta_w)
#                 dy1 = t * zdim / np.tan(theta_s)
#                 dx2 = t * zdim / np.tan(theta_e)
#                 dy2 = t * zdim / np.tan(theta_n)
#
#                 m1 = dx1 / xdim
#                 n1 = dy1 / ydim
#                 m2 = dx2 / xdim
#                 n2 = dy2 / ydim
#                 for m in range(max(math.floor(i - m2), 1), min(numx, math.ceil(i + m1)) + 1):
#                     for n in range(max(math.floor(j - n2), 1), min(numy, math.ceil(j + n1)) + 1):
#                         a = xdim * (i - m)
#                         b = ydim * (j - n)
#                         a2 = a ** 2
#                         b2 = b ** 2
#                         if m >= i and n >= j:
#                             Value = a2 / (dx1 ** 2) + b2 / (dy1 ** 2)
#                         elif m >= i and n <= j:
#                             Value = a2 / (dx1 ** 2) + b2 / (dy2 ** 2)
#                         elif m <= i and n <= j:
#                             Value = a2 / (dx2 ** 2) + b2 / (dy2 ** 2)
#                         else:
#                             Value = a2 / (dx2 ** 2) + b2 / (dy1 ** 2)
#                         if Value <= 1:
#                             arcos.append([(i, j, k), (m, n, k - t)])

# Lo mismo de antes, pero solo predeccesor inmediato
S = {}
# inicializamos S para el primer nivel
for i in range(1, numx + 1):
    for j in range(1, numy + 1):
        S[(i, j, 1)] = []
init_time = time.time()
for k in range(1, numz + 1):
    #for k in range(numz, 0, -1):
    for i in range(1, numx + 1):
        for j in range(1, numy + 1):
            lista_predecesores = []
            for t in range(1, k):

                dx1 = t * zdim / np.tan(theta_w)
                dy1 = t * zdim / np.tan(theta_s)
                dx2 = t * zdim / np.tan(theta_e)
                dy2 = t * zdim / np.tan(theta_n)

                m1 = dx1 / xdim
                n1 = dy1 / ydim
                m2 = dx2 / xdim
                n2 = dy2 / ydim
                for m in range(max(math.floor(i - m2), 1), min(numx, math.ceil(i + m1)) + 1):
                    for n in range(max(math.floor(j - n2), 1), min(numy, math.ceil(j + n1)) + 1):
                        if k > 2 and t > 1:
                            if n == j and (m+1, j, k-t+1) in S and (m, j, k-t) in S[(m+1, j, k-t+1)]:
                                continue
                            elif n == j and (m-1, j, k-t+1) in S and (m, j, k-t) in S[(m-1, j, k-t+1)]:
                                continue
                            elif m == i and (i, n+1, k-t+1) in S and (i, n, k-t) in S[(i, n+1, k-t+1)]:
                                continue
                            elif m == i and (i, n-1, k-t+1) in S and (i, n, k-t) in S[(i, n-1, k-t+1)]:
                                continue
                        a = xdim * (i - m)
                        b = ydim * (j - n)
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

print('\n############################################\n')
print('     Tiempo determiar vecinos: %.2f' % (time.time()-init_time))
print('\n############################################\n')

# sacar = []
# for arco in arcos:
#     bloque_tail = arco[0]
#     bloque_head = arco[1]
#     if bloque_base_1 == bloque_tail:
#         sacar.append(bloque_head)
# print(sacar)

# graficar resultados de la asignacion
# prueba: retirar bloques repetidos
# for (p,q,z) in S[bloque_base_1]:
#     if (p,q,z) in S[bloque_base_2]:
#         S[bloque_base_2].remove((p,q,z))
# ahora para todos los bloques, se sacan los arcos redundantes
#-----------------------------------------------------
#-------- Dejar solo los vecinos inmediatos ----------
#-----------------------------------------------------

init_time = time.time()
for i in range(1, numx+1):
    for j in range(1, numy + 1):
        for k in range(1, numz + 1):
            for (p, q, z) in [(i+1, j, k-1), (i-1, j, k-1), (i, j+1, k-1), (i, j-1, k-1), (i, j, k-1)]:
                if (p, q, z) in S:
                    for (w, t, h) in S[(p, q, z)]:
                        if (w, t, h) in S[(i, j, k)]:
                            S[(i, j, k)].remove((w, t, h))
print('\n############################################\n')
print('     Tiempo determiar vecino inmediato: %.2f' % (time.time()-init_time))
print('\n############################################\n')
#-------------------------------------------------------
#------------ Escribir arcos de precedencia ------------
#-------------------------------------------------------
# precedencias = pd.DataFrame(columns=['tail_i', 'tail_j', 'tail_k','head_i', 'head_j', 'head_k'])
# index = 0
# init_time = time.time()
# for k in range(1, numz+1):
#     for i in range(1, numx + 1):
#         for j in range(1, numy + 1):
#             if (i, j, k) in S:
#                 print('Escribiendo precedencia para: (%.1d,%.1d,%.1d)' % (i, j, k))
#                 for (p, q, z) in S[(i, j, k)]:
#                     precedencias.loc[index] = [i, j, k, p, q, z]
#                     index += 1
# precedencias.to_csv(path_or_buf='../Blasor_inputs_fases/presedencias.csv', index=False)
# print('\n############################################\n')
# print('     Tiempo escritura: %.2f' % (time.time()-init_time))
# print('\n############################################\n')

#-------------------------------------------------------
#------------ Graficar arcos de precedencia ------------
#-------------------------------------------------------

print('\n############################################\n')
print('   Ejecutando Algoritmo de Lerchs & Grossmann  ')
print('\n############################################\n')
print('Inicialización...')
S['vo'] = list()
for j in range(1, numy+1):
    for k in range(1, numz + 1):
        for i in range(1, numx + 1):
            S['vo'].append((i, j, k))
pesos_ejemplo_2 = [-1, 1, 2, -1, 1, -1, -2, -1, 3, 4, -1, -2, -3, -2, 1, 2, -1, -2]
pesos_ejemplo_1 = [1, 1, -1, -1, -100, 2, -1, -100]
w = dict()
for j in range(1, numy+1):
    for k in range(1, numz + 1):
        for i in range(1, numx + 1):
            w[(i, j, k)] = pesos_ejemplo_2[i + (k-1)*numx-1]

etiquetas = dict()
for (p, q, z) in S['vo']:
    etiquetas[('vo', (p, q, z))] = ['+', w[(p, q, z)]]

Y = list()
for j in range(1, numy+1):
    for k in range(1, numz + 1):
        for i in range(1, numx + 1):
            if w[(i, j, k)] > 0:
                Y.append((i, j, k))
T = [('vo', (p, q, z)) for (p, q, z) in S['vo']]


def test_opt(vertices):
    for vertice in vertices:
        for vecino in S[vertice]:
            if vecino not in vertices:
                return vertice, vecino
    return True


def encontrar_camino(origen, destino, arcosArbol, visitados, antecesor):
    neighbors_origen = []
    for tail, head in arcosArbol:
        if origen in (tail, head):
            if head not in visitados and not head == 'vo':
                neighbors_origen.append(head)
            elif tail not in visitados and not tail == 'vo':
                neighbors_origen.append(tail)
    visitados.extend(neighbors_origen)
    for vertice in neighbors_origen: # recuperar vo-vk camino almacenado en pi
        antecesor[vertice] = origen
    if destino in visitados:
        hoja_fuerte = cp.copy(destino)
        padre_fuerte = antecesor[hoja_fuerte]
        aristas_camino = [(padre_fuerte, hoja_fuerte)]
        while not padre_fuerte == 'vo':
            hoja_fuerte = padre_fuerte
            padre_fuerte = antecesor[hoja_fuerte]
            aristas_camino.append((padre_fuerte, hoja_fuerte))
        return aristas_camino
    else:
        vecino = visitados.pop()
        encontrar_camino(vecino, destino, arcosArbol, visitados, antecesor)


def encontrar_camino_alt(origen, destino, arcosArbol):
    antecesor = {}
    visitados = [origen]
    cola = [origen]
    while cola:
        neighbors_nodo = []
        nodo = cola.pop()
        for tail, head in arcosArbol:
            if nodo in (tail, head):
                if not head == nodo:
                    neighbors_nodo.append(head)
                else:
                    neighbors_nodo.append(tail)
        cola.extend([vecino for vecino in neighbors_nodo if w not in visitados])
        for vecino in [vecino for vecino in neighbors_nodo if vecino not in visitados]:
            antecesor[vecino] = nodo
            visitados.append(vecino)
        if destino in visitados:
            break
    hoja_fuerte = cp.copy(destino)
    padre_fuerte = antecesor[hoja_fuerte]
    aristas_camino = [(padre_fuerte, hoja_fuerte)]
    while not padre_fuerte == 'vo':
        hoja_fuerte = padre_fuerte
        padre_fuerte = antecesor[hoja_fuerte]
        aristas_camino.append((padre_fuerte, hoja_fuerte))
    return aristas_camino


def encontrar_camino_alt2(origen, destino, arcosArbol):  # ahora guarda la orientación de los arcos!!!
    antecesor = {}
    visitados = [origen]
    cola = [origen]
    while cola:
        neighbors_nodo = []
        nodo = cola.pop()
        for tail, head in arcosArbol:
            if nodo in (tail, head):
                if not head == nodo:
                    neighbors_nodo.append((head, 'head'))
                else:
                    label = 'tail'
                    neighbors_nodo.append((tail, 'tail'))
        cola.extend([w for w, label in neighbors_nodo if w not in visitados])
        for vecino, label in neighbors_nodo:
            if vecino not in visitados:
                antecesor[vecino] = (nodo, label)
                visitados.append(vecino)
        if destino in visitados:
            break
    hoja = cp.copy(destino)
    padre, label_hoja = antecesor[hoja]
    if label_hoja == 'head':
        aristas_camino = [(padre, hoja)]
    else:
        aristas_camino = [(hoja, padre)]
    while not padre == 'vo':
        hoja = padre
        padre, label_hoja = antecesor[hoja]
        if label_hoja == 'head':
            aristas_camino.append((padre, hoja))
        else:
            aristas_camino.append((hoja, padre))
    return aristas_camino


def obtener_rama(raiz, arcosArbol, nodos_fuertes):
    vecinos = []
    for tail, head in arcosArbol:
        if raiz in (tail, head):
            if (not head == 'vo') and head not in nodos_fuertes:
                vecinos.append(head)
            elif (not tail == 'vo') and tail not in nodos_fuertes:
                vecinos.append(tail)
    if vecinos:
        for vecino in vecinos:
            if vecino not in nodos_fuertes:
                nodos_fuertes.append(vecino)
            obtener_rama(vecino, arcosArbol, nodos_fuertes)


print('Arbol de la inicializacion', T)
print('Yo de la iniciailizacion', Y)
init_time = time.time()
while True:
    if test_opt(Y) is True:  # verificar optimalidad
        break
    (vk, vl) = test_opt(Y)  # si no se tiene, actualizar arbol: STEP 3
    # vo_vk_camino = encontrar_camino('vo', vk, T, vistos, pi)  # retorna un diccionario para el camino
    # hoja_fuerte = cp.copy(vk)
    # padre_fuerte = vo_vk_camino[hoja_fuerte]
    # aristas_vo_vk_camino = [(padre_fuerte, hoja_fuerte)]
    # while not padre_fuerte == 'vo':  # recuperar vo-vk camino almacenado en pi
    #     hoja_fuerte = padre_fuerte
    #     padre_fuerte = vo_vk_camino[hoja_fuerte]
    #     aristas_vo_vk_camino.extend([(padre_fuerte, hoja_fuerte)])
    #aristas_vo_vk_camino = encontrar_camino('vo', vk, T, vistos, pi)
    aristas_vo_vk_camino = encontrar_camino_alt2('vo', vk, T)#  encontrar caminos a extremos de ñas aristas
    vm = cp.copy(aristas_vo_vk_camino[0][1])
    aristas_vo_vk_camino.reverse()  # reverse para que el nombre tenga sentido
    #aristas_vl_vo_camino = encontrar_camino('vo', vl, T, vistos, pi)
    aristas_vl_vo_camino = encontrar_camino_alt2('vo', vl, T)
    # vo_vl_camino = encontrar_camino('vo', vl, T, vistos, pi)
    # hoja_debil = cp.copy(vl)
    # padre_debil = vo_vl_camino[hoja_debil]
    # aristas_vl_vo_camino = [(padre_debil, hoja_debil)]
    # while not padre_debil == 'vo':  # recuperar vo-vl camino almacenado en pi
    #     hoja_debil = padre_fuerte
    #     padre_debil = vo_vl_camino[hoja_debil]
    #     aristas_vl_vo_camino.extend([(padre_debil, hoja_debil)])
    # vn = cp.copy(hoja_debil)
    vn = cp.copy(aristas_vl_vo_camino[0][1])
    aristas_vm_vo_camino = []
    aristas_vm_vo_camino.extend(aristas_vo_vk_camino[1:])
    aristas_vm_vo_camino.extend(aristas_vl_vo_camino)
    T.remove(('vo', vm))
    T.extend([(vk, vl)])
    T_prima = T.copy()
    etiquetas[(vk, vl)] = ('-', etiquetas[('vo', vm)][1])
    for arista in aristas_vo_vk_camino[1:]:  # actualizar etiquetas de ambos caminos
        etiqueta = etiquetas[arista]
        if etiqueta[0] == '+':
            etiquetas[arista] = ('-', etiquetas[('vo', vm)][1]-etiqueta[1])
        else:
            etiquetas[arista] = ('+', etiquetas[('vo', vm)][1]-etiqueta[1])
    for arista in aristas_vl_vo_camino:
        etiqueta = etiquetas[arista]
        if etiqueta[0] == '+':
            etiquetas[arista] = ('+', etiquetas[('vo', vm)][1]+etiqueta[1])
        else:
            etiquetas[arista] = ('-', etiquetas[('vo', vm)][1]+etiqueta[1])
    # normalizacion del arbol despues de actualizar etiquetas: STEP 4
    for inidice, arista in enumerate(aristas_vm_vo_camino):
        etiqueta = etiquetas[arista]
        if etiqueta[0] == '+' and etiqueta[1] > 0 and 'vo' not in arista:  # preguntar si hay arco fuerte en el camino
            T_prima.remove(arista)
            T_prima.append(('vo', arista[1]))
            for edge in aristas_vm_vo_camino[inidice+1:]:  # actualizar etiquetas del camino
                w_actualizado = etiquetas[edge][1]-etiquetas[arista][1]
                if etiquetas[edge][0] == '+':
                    etiquetas[edge] = ('+', w_actualizado)
                else:
                    etiquetas[edge] = ('-', w_actualizado)
    T = cp.copy(T_prima)  # asignar a T el arbol normalizado
    Y = list()  # actualizar Y
    arcos_fuertes = [arco for arco in T if arco[0] == 'vo' and etiquetas[arco][0] == '+' and etiquetas[arco][1] > 0]
    for arco in arcos_fuertes:
        obtener_rama(arco[1], T, Y)
print('los nodos a extraer son:', Y)
print('LG tomo:', time.time()-init_time)

print('\n############################################\n')
print('   Ejecutando Algoritmo de Ford-Fulkerson  ')
print('\n############################################\n')



#############################################
# Graficar resultados
#############################################

bloque_base_1 = (4, 1, 3)  # bloque base para plotear
bloque_base_2 = (5, 1, 2)

MB_grafico = MB_sorted[['xcentre', 'ycentre', 'zcentre']]-(min(MB_sorted[['xcentre']].values),
                                                           min(MB_sorted[['ycentre']].values),
                                                           min(MB_sorted[['zcentre']].values))
ejex = MB_grafico[['xcentre']]
ejey = MB_grafico[['ycentre']]
ejez = MB_grafico[['zcentre']]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.gca().invert_zaxis()
#
# #for v in sacar:
# for v in S[bloque_base_1]:
#     a = Arrow3D([(bloque_base_1[0]-1)*20, (v[0]-1)*20], [(bloque_base_1[1]-1)*20, (v[1]-1)*20],
#                 [(bloque_base_1[2]-1)*15, (v[2]-1)*15], mutation_scale=20,
#                 lw=3, arrowstyle="-|>", color="r")
#     ax.add_artist(a)
# ax.scatter(ejex, ejey, ejez, s=80)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
plt.gca().invert_zaxis()

# sacar_pred: lista que tiene sacar predecesores
sacar_pred = [bloque for bloque in S[bloque_base_2]]
for v in sacar_pred:
    a = Arrow3D([(bloque_base_2[0]-1)*20, (v[0]-1)*20], [(bloque_base_2[1]-1)*20, (v[1]-1)*20],
                [(bloque_base_2[2]-1)*15, (v[2]-1)*15], mutation_scale=20,
                lw=3, arrowstyle="-|>", color="b")
    ax2.add_artist(a)
for v in S[bloque_base_1]:
    a = Arrow3D([(bloque_base_1[0]-1)*20, (v[0]-1)*20], [(bloque_base_1[1]-1)*20, (v[1]-1)*20],
                [(bloque_base_1[2]-1)*15, (v[2]-1)*15], mutation_scale=20,
                lw=3, arrowstyle="-|>", color="r")
    ax2.add_artist(a)
ax2.scatter(ejex, ejey, ejez, s=80)

# LG para MB_sorted (i.e. que tiene bastantes menos nodos)
plt.show()
