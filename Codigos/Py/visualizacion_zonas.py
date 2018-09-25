from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
# input_path = '../Blasor_inputs/MB_LoA20_v2.csv'
input_path = '../Daniel_inputs/block_model.csv'
MB = pd.read_csv(input_path)
# MB_zona_1= MB.loc[MB['zone'] == 1]
N = 200
# datos_x = MB_zona_1[['xcentre']][:N]
# datos_y = MB_zona_1[['ycentre']][:N]
# datos_z = MB_zona_1[['zcentre']][:N]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(datos_x,datos_y,datos_z)

# MB_zona_2= MB.loc[MB['zone'] == 2]
# datos_x = MB_zona_2[['xcentre']][:N]
# datos_y = MB_zona_2[['ycentre']][:N]
# datos_z = MB_zona_2[['zcentre']][:N]
# fig2 = plt.figure()
# ax = fig2.add_subplot(111, projection='3d')
# ax.scatter(datos_x,datos_y,datos_z)

datos_x = MB[['xcentre']]
datos_y = MB[['ycentre']]
datos_z = MB[['zcentre']]
fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
ax.scatter(datos_x,datos_y,datos_z)

# plotear ordenado por eje z
MB_sorted = MB.sort_values(by=['zcentre'],ascending=False)

# datos_x = MB_sorted[['xcentre']][:N]
# datos_y = MB_sorted[['ycentre']][:N]
# datos_z = MB_sorted[['zcentre']][:N]
# fig3 = plt.figure()
# ax = fig3.add_subplot(111, projection='3d')
# ax.scatter(datos_x,datos_y,datos_z,s=1.3)

# alturas plot
# fig4 = plt.figure()
# alturas = MB_sorted[['zcentre']]
# ax = fig4.add_subplot(111)
# ax.plot(range(len(alturas)),alturas)
plt.show()

######################################################
# determinar el conjunto de arcos del grafo de la mina
# Lerchs-Grossmann algorithm with variable slope angles
# R. Khalohahaie, P. A. Dowd and R. J. Fowell
######################################################

# alturas_ndarray = MB_sorted[['zcentre']].as_matrix()
# alturas_unicas = np.unique(alturas_ndarray)
# altura_final = alturas_unicas[-1]
# E = [] #conjunto de aristas
# k = len(alturas_unicas) # numero de niveles del modelo de bloque
# xdim = 20; ydim = 20; zdim = 15



