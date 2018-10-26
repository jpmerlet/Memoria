import time
import copy as cp


def test_opt(vertices):
    for vertice in vertices:
        for vecino in S[vertice]:
            if vecino not in vertices:
                return vertice, vecino
    return True


def encontrar_camino(origen, destino, arcosArbol):
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
                    neighbors_nodo.append((tail, 'tail'))
        cola.extend([vecino for vecino, label in neighbors_nodo if vecino not in visitados])
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


upit_path = '../minelib_inputs/newman1.upit'
prec_path = '../minelib_inputs/newman1.prec'

lineas = []
w = {}  # diccionario de peses
# recuperar los pesos de cada bloque
with open(upit_path, 'r') as file:
    for line in file:
        line_list = line.split()
        lineas.append(line_list)
lineas = cp.copy(lineas[4:])
for nodo_id, linea in enumerate(lineas):
    w[nodo_id] = float(linea[1])
nblocks = len(w)

# recuperar las precedencias para cada bloque
S = {}  # lista de precedencias inmediatas
with open(prec_path) as file:
    for nodo_id, line in enumerate(file):
        S[nodo_id] = []
        line_list = line.split()
        if not line_list[1] == '0':
            S[nodo_id].extend(list(map(int, cp.copy(line_list[2:]))))

# agregar nodo vo artificial, e inicializar arbol y nodos fuertes
S['vo'] = []
for nodo_id in range(nblocks):
    S['vo'].append(nodo_id)
T = [('vo', nodo) for nodo in S['vo']]  # incializar arbol
etiquetas = {}  # crear diccionario de etiquetas
for nodo in S['vo']:
    etiquetas[('vo', nodo)] = ['+', w[nodo]]
Y = []  # crear lista de nodos fuertes
for nodo_id in range(nblocks):
    if w[nodo_id] > 0:
        Y.append(nodo_id)

print('Arbol de la inicializacion', T)
print('Yo de la iniciailizacion', Y)
init_time = time.time()
while True:
    if test_opt(Y) is True:  # verificar optimalidad
        break
    (vk, vl) = test_opt(Y)  # si no se tiene, actualizar arbol: STEP 3
    aristas_vo_vk_camino = encontrar_camino('vo', vk, T)  # encontrar caminos a extremos de ñas aristas
    vm = cp.copy(aristas_vo_vk_camino[0][1])
    aristas_vo_vk_camino.reverse()  # reverse para que el nombre tenga sentido
    aristas_vl_vo_camino = encontrar_camino('vo', vl, T)
    vn = cp.copy(aristas_vl_vo_camino[0][1])
    aristas_vm_vo_camino = []
    aristas_vm_vo_camino.extend(aristas_vo_vk_camino[1:])
    aristas_vm_vo_camino.extend(aristas_vl_vo_camino)
    print('Se remueve el arco del ciclo: ', ('vo', vm))
    T.remove(('vo', aristas_vo_vk_camino[0][1]))
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
print('El valor optimo es:', sum([pesos for key, pesos in w.items() if key in Y]))
