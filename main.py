#Importamos las librerías pertinentes
import numpy as np
from functions import *

#Tomamos los valores iniciales y los guardamos según el tipo de barra
init_values = initial_values()
if init_values[-1] == 0:
    bar = init_values[2]
    si_loc = init_values[0]
    sf_loc = init_values[1]
else:
    bar = init_values[1]
    si_loc = init_values[0]
    sf_loc = -1 #Codificarle a las funciones que la barra está empotrada

#Le preguntamos al usuario si desea considerar el perfil de la barra e ingresa los datos adicionales requeridos (Para calcular TaoMax y Sigma Max)
aditional_values = aditional_info()

#Generamos la discretización (Nótese que esta se mantiene fija en 0.01, es decir 100 puntos por metro)
discret = bar/(bar*100)

#EL usuario ingresa los momentos puros
p_moments = pure_moments(bar, discret)

#El usuario ingresa las fuerzas distribuidas y las fuerzas puntuales. Generamos matriz de fuerzas
func = process_forces(bar, discret)

force_mat = func[0] #Guardamos matriz de fuerzas

#Calculamos las reacciones
calc_data = calc_reactions_sup(force_mat, si_loc, sf_loc, p_moments, discret)

#Guardamos los datos de las reacciones para incluirlos en los gráficos
reac_1 = calc_data[2]
reac_2 = calc_data[3]

#Guardamos la matriz de fuerzas teniendo en cuenta las reacciones
force_with_reactions = calc_data[0]

#Guardamos los momentos puros teniendo en cuenta la reacción angular
moments = calc_data[1]

#Calculamos las cortanes
sheer_forces = calc_sheer_forces(force_with_reactions, bar, discret)

#Integramos para obtener las flexionantes y hallamos esfuerzos máximos en la viga
integral_list = riemann_sum(discret, sheer_forces[1,:])
if np.sum(moments[1]) != 0:
    integral_list = add_pure_moments(moments, integral_list, bar, discret)
if aditional_values[5] == "y":
    sigma_max = max_sigma(integral_list, float(aditional_values[0]))
    tao_max = max_tao(sheer_forces[1], float(aditional_values[1]), float(aditional_values[2]), float(aditional_values[3]), aditional_values[6], float(aditional_values[4]))
elif aditional_values[5] == "n":
    sigma_max = 0
    tao_max = 0

#Extraemos todos los datos adicionales necesarios para graficar
if sf_loc == -1:
    bar_typ = init_values[2]
else:
    bar_typ = init_values[3]
locs_sup = [si_loc, -3]
loc_der_sup, loc_iz_sup = find_point(locs_sup)
locs_sup2 = [sf_loc, -3]
loc_der_sup2, loc_iz_sup2 = find_point(locs_sup2)
force_m = func[1]
distri = func[2]

#Dibujamos la barra con las reacciones y todas las fuerzas ingresadas por el usuario
gen_graph(bar, bar_typ, locs_sup, loc_der_sup, loc_iz_sup, locs_sup2,loc_der_sup2, loc_iz_sup2,
          force_m, distri, p_moments, reac_1, reac_2)

#Dibujamos las gráficas de cortantes y flexionantes teniendo en cuenta
#Cortante máximo, flexionante máximo, Tao Máximo (Esfuerzo por cortante) y Sigma Máximo (Esfuerzo por flexionante)
cool_graphs(sheer_forces,moments,discret,bar, sigma_max, tao_max)