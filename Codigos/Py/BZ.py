import pandas as pd
import numpy as np
from gurobipy import *
from itertools import combinations
import time
from math import isclose

# leer datos de minelib
# obtener funcion objetivo del .pcpsp, para comparar
# el resultado del modelo gurobi con el modelo AMPL
data_name = 'zuck_small'
pcpsp_path = '../minelib_inputs/' + data_name + '.pcpsp'
prec_path = '../minelib_inputs/' + data_name + '.prec'
upit_path = '../minelib_inputs/'+ data_name + '.upit'
objective_function_pcpsp = {}
resource_constraint_ub_limits = {}
resource_constraint_lb_limits = {}
resource_constraint_coefficients = {}
with open(pcpsp_path, 'r') as f:
    for linea in f:
        linea_lista = linea.split()
        print(linea_lista)
        if linea_lista[0] == 'NAME:':
            dato = linea_lista[1].strip('\n')
            name = dato
        elif linea_lista[0] == 'NBLOCKS:':
            dato = linea_lista[1].strip('\n')
            nblocks = int(dato)
        elif linea_lista[0] == 'NPERIODS:':
            dato = linea_lista[1].strip('\n')
            nperiods = int(dato)
        elif linea_lista[0] == 'NDESTINATIONS:':
            dato = linea_lista[1].strip('\n')
            ndestinations = int(dato)
        elif linea_lista[0] == 'NRESOURCE_SIDE_CONSTRAINTS:':
            dato = linea_lista[1].strip('\n')
            nresource_side_constraints = int(dato)
        elif linea_lista[0] == 'NGENERAL_SIDE_CONSTRAINTS:':
            dato = linea_lista[1].strip('\n')
            ngeneral_side_constraints = int(dato)
        elif linea_lista[0] == 'DISCOUNT_RATE:':
            dato = linea_lista[1].strip('\n')
            discount_rate = float(dato)
        elif linea_lista[0] == 'RESOURCE_CONSTRAINT_LIMITS:':
            for r in range(nresource_side_constraints):
                for t in range(nperiods):
                    linea = f.readline()
                    lista = linea.split()
                    if lista[2] == 'L':
                        resource_constraint_ub_limits[r,t] = int(lista[-1])
                        resource_constraint_lb_limits[r,t] = '-Infinity' # falta hacer para el caso general
                    else:
                        print('Este problema tiene cotas inferiores.')
                        break
    
        elif linea_lista[0] == 'OBJECTIVE_FUNCTION:':
            for b in range(nblocks):
                linea = f.readline()
                lista= linea.split()
                for d in range(ndestinations):
                    objective_function_pcpsp[b,d] = float(lista[d+1])

        elif linea_lista[0] == 'RESOURCE_CONSTRAINT_COEFFICIENTS:':
                for linea in f:
                    if linea == 'EOF\n':
                        break
                    lista = linea.split()
                    b = int(lista[0])
                    d = int(lista[1])
                    r = int(lista[2])
                    resource_constraint_coefficients[b,r,d] = float(lista[3])

# llenar con ceros las entradas de resource_constraint_coefficients
# que no está definidas
for b,r,d in itertools.product(range(nblocks), range(nresource_side_constraints),range(ndestinations)):
    if not (b,r,d) in resource_constraint_coefficients:
        resource_constraint_coefficients[b,r,d] = 0

# block value list para upit
bv_list = list() 
with open(upit_path, 'r') as f:
    for i in range(4):
        f.readline()
    for line in f:
        if not line == 'EOF\n':
            lista = line.split()
            bv_list.append(float(lista[1]))

print('Data base name: %s' % (name))
print('NBLOCKS: %d' % nblocks)
print('NPERIODS: %d' % nperiods)
print('NDESTINATIONS: %d' % ndestinations)
print('NRESOURCE_SIDE_CONSTRAINTS: %d' % nresource_side_constraints)
print('NGENERAL_SIDE_CONSTRAINTS: %d' % ngeneral_side_constraints)
print('DISCOUNT_RATE: %.2f\n' % discount_rate)
resourceTimesPeriod = list(itertools.product(list(range(nresource_side_constraints)),
                                             list(range(nperiods))))
# funcion de reformulacion
# para PCP_at a PCP_by
def at_by_key(b,d,t):
    if d == 0 and t > 0:
        return b,ndestinations-1,t-1
    elif d > 0:
        return b,d-1,t
    elif d == 0 and t == 0:
        print('at_by_key no esta difinida para los valores d = %d, t = %d' % (d,t))

def check_optimality(sol,particion,ell,iteracion):
    for h in range(1,ell+1):
        for (b1,d1,t1),(b2,d2,t2) in combinations(particion[h,iteracion-1],2):
            if not sol[iteracion][b1,d1,t1] == sol[iteracion][b2,d2,t2]:
                return False
    return True

def check_opt_vect(sol, particion, ell, iteracion):
    for h in range(1,ell[iteracion-1]+1):
        arreglo = np.array([sol[iteracion][b].x for b in particion[h,iteracion-1]])
        summ = np.matmul(arreglo,np.ones(arreglo.shape))
        if not (np.isclose(summ,0) or np.isclose(summ,len(particion[h,iteracion-1]))):
            return False
    return True

if __name__ == "__main__":
	# resolver upit
	upit = Model()
	# variable de desicion para el modelo
	x = {}
	for i in range(nblocks):
	    x[i] = upit.addVar(vtype=GRB.BINARY, name = "x%d" % i)
	upit.update()
	# definir objetivo
	upit.setObjective(LinExpr([bv_list[i]for i in range(nblocks)], [x[i] for i in range(nblocks)]), GRB.MAXIMIZE)
	# definir restricciones
	with open(prec_path, 'r') as f:
	    for linea in f:
	        linea_lista = linea.split()
	        nvecinos = int(linea_lista[1])
	        u = int(linea_lista[0])
	        for j in range(nvecinos):
	            v = int(linea_lista[j+2])
	            upit.addConstr(x[u] <= x[v])
	upit.optimize()
	# recuperar upit
	blocks_id_upit = [i for i in range(nblocks) if x[i].x==1] # recuperar upit
	print('UPIT BLOCKS: %d' % len(blocks_id_upit))
	###################################################################################
	# inicializar variables para BZ
	blocks = blocks_id_upit # se parte de los bloques ya encontrados por upit
	blocks_prime = list(itertools.product(blocks, list(range(ndestinations)),list(range(nperiods))))
	blockTimesDest = list(itertools.product(blocks,range(ndestinations)))
	mu = {}
	C = {}
	mu[0] = {}
	for r,t in resourceTimesPeriod:
	    mu[0][r,t] = 0
	C[1,0] = set(blocks_prime)
	k = 1
	l = 1
	# calculamos la forma equivalente del problema
	# PCP_at, llamada PCP_by.
	# funcion objetivo
	c = {}
	for (b,d,t) in blocks_prime:
	    c[b,d,t] = (1.0/(1.0 + discount_rate))**t * objective_function_pcpsp[b,d]

	# recalculamos c para PCP_by, c_hat
	c_hat = {}
	for (b,d,t) in blocks_prime:
	    if d > 0 or t >0:
	        c_hat[b,d,t] = c[b,d,t]
	        c_hat[at_by_key(b,d,t)] = -c[b,d,t]
	    else:
	        c_hat[b,d,t] = c[b,d,t]

	# modelo L(PCP_by,mu[k-1])
	problema_aux = Model()
	# definir variable de PCP_by
	x = {}
	for (b,d,t) in blocks_prime:
	    x[b,d,t] = problema_aux.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name = "x(%d,%d,%d)" % (b,d,t))
	problema_aux.update()
	# agregar restricciones de precedencias
	# precedencia temporal
	for t in range(nperiods-1):
	    for b in blocks:
	        problema_aux.addConstr(x[b,ndestinations-1,t] <= x[b,0,t+1])

	# precedencia en los destinos
	for d in range(ndestinations-1):
	    for t in range(nperiods):
	        for b in blocks:
	            problema_aux.addConstr(x[b,d,t] <= x[b,d+1,t])

	# precedencia espacial
	with open(prec_path, 'r') as f:
	    for linea in f:
	        linea_lista = linea.split()
	        nvecinos = int(linea_lista[1])
	        a = int(linea_lista[0])
	        if a in blocks:
	            for j in range(nvecinos):
	                b = int(linea_lista[j+2])
	                for t in range(nperiods):
	                    problema_aux.addConstr(x[a,ndestinations-1,t] <= x[b,ndestinations-1,t])

	# Funcion objetivo de L(PCP_by,mu[k-1])
	cx_direct = quicksum([c[b,d,t]*(x[b,d,t]-x[b,d-1,t]) for b,d,t in blocks_prime if d>0 and t>0])\
	            +quicksum([c[b,0,t]*(x[b,0,t]-x[b,ndestinations-1,t-1]) for (b,d,t) in blocks_prime if d==0 and t>0])\
	            +quicksum([c[b,0,0]*x[b,0,0] for b in blocks])
	q = resource_constraint_coefficients
	d_rhs = resource_constraint_ub_limits
	w = {}
	z = {}
	eles = {}
	eles[0] = 1
	time_cm = []
	problema_aux.setParam( 'OutputFlag', False ) # El modelo no imprime output en pantalla
	while True:
	    #if k==10:
	    #    break
	    # STEP 1: resolver L(PCPby,mu[k-1])
	    suma = {}
	    side_const = LinExpr()
	    LHS = {}
	    for r in range(nresource_side_constraints):
	        for t in range(nperiods):
	            sumando_1 = quicksum([q[b,r,d]*(x[b,d,t]-x[b,d-1,t]) for b,d in blockTimesDest if d>0])
	            if t > 0:
	                LHS[r,t] = sumando_1+quicksum([q[b,r,0]*(x[b,0,t]-x[b,ndestinations-1,t-1]) for b in blocks])
	            else:
	                LHS[r,0] = sumando_1+quicksum([q[b,r,0]*x[b,0,0] for b in blocks])
	    
	    fn_objetivo = cx_direct - quicksum([mu[k-1][r,t]*(LHS[r,t]-d_rhs[r,t]) for r,t in resourceTimesPeriod])
	    problema_aux.setObjective(fn_objetivo, GRB.MAXIMIZE)
	    
	    print('\nResolviendo problema auxiliar: k = %d' % k)
	    problema_aux.Params.presolve = 0
	    problema_aux.optimize()
	    w[k] = x
	    # verificar optimalidad de z[k-1]
	    init_time = time.time()
	    if k>=2 and check_opt_vect(w,C,eles,k):
	        print('Algoritmo termino: H^%d w[%d] = 0' % (k-1,k))
	        print('Valor optimo de BZ: %.2f' % (model_p2k.ObjVal*(1/obj_scale)))
	        break
	    print('\nchequear optimalidad tomo: %.2f[s]' % (time.time()-init_time))
	    print('\n Encontrar particion')
	    init_time = time.time()
	    #STEP 3: encontrar particion de blocks_prime 
	    I = [ block for block in blocks_prime if w[k][block].x == 1]
	    O = [ block for block in blocks_prime if w[k][block].x == 0]
	    count = 0
	    for h in range(1,eles[k-1]+1):
	        if C[h,k-1].intersection(I):
	            count += 1
	            C[count,k] = C[h,k-1].intersection(I)
	        if C[h,k-1].intersection(O):
	            count += 1
	            C[count,k] = C[h,k-1].intersection(O)
	    
	    print('Determinar particion para k = %d tomo: %.2f' % (k,time.time()-init_time))
	    eles[k] = count
	    # STEP 4: resolver P2k
	    print('\nConstruccion del modelo P2^k')
	    init_time = time.time()
	    model_p2k = Model()
	    lmbda = {}
	    init_dv = time.time()
	    for i in range(1,eles[k]+1):
	        lmbda[i] = model_p2k.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name = "lambda%d" % i)
	    
	    model_p2k.update()
	    print('Definir lmbda y actualizar: %.2f[s]' % (time.time()-init_dv))
	    # fn objetivo del problema auxiliar P2^k
	    init_sev = time.time()
	    x_lmbda = {}
	    for b in blocks_prime:
	        #x_lmbda[b] = quicksum([lmbda[h]*(b in C[h,k])for h in range(1,eles[k]+1)])
	        x_lmbda[b] = LinExpr([(b in C[h,k]) for h in range(1,eles[k]+1)],[lmbda[h] for h in range(1,eles[k]+1)])
	    print('Proyectar variable en H^kx = 0 tomo: %.2f[s]' % (time.time()-init_sev))
	    init_obj = time.time()
	    cx_lmbda = quicksum([c[b,d,t]*(x_lmbda[b,d,t]-x_lmbda[b,d-1,t]) for b,d,t in blocks_prime if d>0 and t>0])\
	                +quicksum([c[b,0,t]*(x_lmbda[b,0,t]-x_lmbda[b,ndestinations-1,t-1]) for (b,d,t) in blocks_prime if d==0 and t>0])\
	                +quicksum([c[b,0,0]*x_lmbda[b,0,0] for b in blocks])
	    obj_scale = 1e0
	    model_p2k.setObjective(obj_scale*cx_lmbda, GRB.MAXIMIZE)
	    print('Calcular fn obj y actualizarla: %.2f[s]' % (time.time()-init_obj))
	    # agregamos las restricciones
	    # para P2k con el cambio de variable
	    # agregar restricciones de
	    # precedencias.
	    init_const = time.time()
	    init_prec_1 = time.time()
	    for t in range(nperiods-1):
	        for b in blocks:
	            model_p2k.addConstr(x_lmbda[b,ndestinations-1,t] <= x_lmbda[b,0,t+1])
	    
	    print('Agregar precedencias 1 : %.2f[s]' % (time.time()-init_prec_1))
	    init_prec_2 = time.time()
	    for d in range(ndestinations-1):
	        for t in range(nperiods):
	            for b in blocks:
	                model_p2k.addConstr(x_lmbda[b,d,t] <= x_lmbda[b,d+1,t])

	    print('Agregar precedencias 2: %.2f[s]' % (time.time()-init_prec_2))
	    init_prec_3 = time.time()
	    with open(prec_path, 'r') as f:
	        for linea in f:
	            linea_lista = linea.split()
	            nvecinos = int(linea_lista[1])
	            a = int(linea_lista[0])
	            if a in blocks:
	                for j in range(nvecinos):
	                    b = int(linea_lista[j+2])
	                    for t in range(nperiods):
	                        model_p2k.addConstr(x_lmbda[a,ndestinations-1,t] <= x_lmbda[b,ndestinations-1,t])
	    
	    print('Agregar precedencias 3: %.2f[s]' % (time.time()-init_prec_3))
	    # agregar side constraints: si at_by_key:
	    LHS_lmbda = {}
	    #for r in range(nresource_side_constraints):
	    #    for t in range(nperiods):
	    #        LHS_lmbda[r,t] = quicksum([q[b,r,d]*(quicksum([lmbda[j]*((b,d,t) in C[j,k]) for j in range(1,eles[k]+1)])-quicksum([lmbda[j]*((b,d-1,t) in C[j,k]) for j in range(1,eles[k]+1)])) for (b,d) in blockTimesDest])
	    #        if t>0:
	    #            LHS_lmbda[r,t] = LHS_lmbda[r,t] + quicksum([q[b,r,0]*(quicksum([lmbda[j]*((b,0,t) in C[j,k]) for j in range(1,eles[k]+1)])-quicksum([lmbda[j]*((b,ndestinations-1,t-1) in C[j,k]) for j in range(1,eles[k]+1)])) for b in blocks])
	    #        else:
	    #            LHS_lmbda[r,0] = LHS_lmbda[r,0] + quicksum([q[b,r,0]*(quicksum([lmbda[j]*((b,0,0) in C[j,k]) for j in range(1,eles[k]+1)])) for b in blocks])
	    init_sd_qs = time.time()
	    for r in range(nresource_side_constraints):
	        for t in range(nperiods):
	            sumando_1 = quicksum([q[b,r,d]*(x_lmbda[b,d,t]-x_lmbda[b,d-1,t]) for b,d in blockTimesDest if d>0])
	            if t > 0:
	                LHS_lmbda[r,t] = sumando_1+quicksum([q[b,r,0]*(x_lmbda[b,0,t]-x_lmbda[b,ndestinations-1,t-1]) for b in blocks])
	            else:
	                LHS_lmbda[r,0] = sumando_1+quicksum([q[b,r,0]*x_lmbda[b,0,0] for b in blocks])
	    
	    print('Construir LinExpr para side const.: %.2f[s]' % (time.time()-init_sd_qs))
	    # agregar side constraints Dx <= d
	    side_const = {}
	    sc_scale = 1e0 # ponderador para side constraints (Dx <=d)
	    init_sd_add = time.time()
	    for r in range(nresource_side_constraints):
	        for t in range(nperiods):
	            side_const[r,t] = model_p2k.addConstr(sc_scale*LHS_lmbda[r,t] <= sc_scale*d_rhs[r,t], name='side_const[%d,%d]' % (r,t))
	        
	    print('Agrergar side const.: %.2f[s]' % (time.time()-init_sd_add))
	    print('\n\nAgregar restricciones: %.2f[s]' % (time.time()-init_const))
	    time_cm.append([time.time()-init_time])
	    print('\nConstruccion del modelo P2^k tomo:%.2f' % (time.time()-init_time))
	    print('\nResolviendo master problem: k=%d' % k)
	    model_p2k.Params.presolve = 0
	    model_p2k.setParam( 'OutputFlag', False )
	    model_p2k.optimize()
	    #break # para chequear si tuvo warnings en p2^k
	    # recuperar variables duales mu[k]
	    mu[k] = {}
	    for r in range(nresource_side_constraints):
	        for t in range(nperiods):
	            mu[k][r,t] = side_const[r,t].pi
	    if all([np.isclose(mu[k][r,t], mu[k-1][r,t]) for r,t in resourceTimesPeriod]): # Cambiar por una tolerancia!!!
	        # recuperar z[k] la solucion del optimo
	        print('Algoritmo termino: mu[%d] = mu[%d]' % (k,k-1))
	        print('Valor optimo de BZ: %.2f' % (model_p2k.ObjVal*(1/obj_scale)))
	        break
	    k += 1