from math import pi
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

#Se ingresan los inputs de tipo,posición de soportes,discretización ylongitud de barra

#Aprpxima números al diferencial m en caso de que el usuario ingrese una ubicación que no se encuntra en la discretiación
def aprox_diff(num, diff_x):
    n_diff = num//diff_x
    residual = num % diff_x
    diff_mean = diff_x/2
    #Decide hacia que dirección aproximar el valor ingresado por el usario
    if residual >= diff_mean:
        return (n_diff*diff_x) + diff_x
    else:
        return n_diff * diff_x

#Convierte un string en una función anónima y le agrega una multiplicación por el diferencial 
#Escribir las funciones con np.func
#Convertimos la función de Newtons/metro a Newtons/cm (=> Factor de conversión 0.01)
def str_to_function(math_expr):
    if "x" in math_expr:
        return lambda x: 0.01 * (eval(math_expr))
    else:
        return lambda x: (x*0) + (eval(math_expr) * 0.01)

#Convierte un string en función anónima conservando Newtons/metro
def str_to_function_wo_dis(math_expr):
    if "x" in math_expr:
        return lambda x:  (eval(math_expr))
    else:
        return lambda x: (x*0) + (eval(math_expr))

#Retorna en index de una distancia dada la distancia y el diferencial 
def get_idx(xi, diff):
    return int(round(xi/diff))

#Limpia los valores infinitos y las discontinuidades 
def remove_discon(mat):
    for i in range(len(mat)):
        if np.isinf(mat[i]):
            if i == 0:
                mat[i] = mat[i+1]
            elif i == len(mat) - 1: #Último elemento
                mat[i] = mat[i-1]
            else: 
                mat[i] = (mat[i-1] + mat[i+1])/2

    return mat

#Genera una matriz con las posiciones en la primera fila y ceros en la segunda, dada la longitud de la barra y el diferencial
def gen_beam_mat(b_len, x_diff):
    int_len = b_len/x_diff
    f = lambda x: x*x_diff
    row_diff = np.arange(0, int_len + 1, 1) #Se suma uno a la longitud entera para garantizar que termine en la longitud final y no un diferencial antes
    row_diff = f(row_diff)
    empty_row = np.zeros_like(row_diff)
    return np.stack([row_diff, empty_row])

#Centra una matriz en la posición de un soporte dado
def shift_mat(mat, s_value):
    func = lambda x: x - s_value
    shifted_mat = func(mat) #Restamos la posición del soporte a todas las otras posiciones para definirlo como el nuevo 0
    return shifted_mat

#Hace sumatoria de momentos para encontrar una de las reacciones en los soportes
#La matriz que utiliza tiene la fila 1 shifteada con el centro en el primer soporte
def moment_sum(mat, react_indx, mat_pure_moments = np.zeros(0)): #Editar para que funcione con empotradas
    #Rf = (sumatoria(momentos)+sumatoria(momentos_puros))/dist_rf
    dist_rf = mat[0, react_indx]
    #Se hace negativo antes de sumar, análogo a pasar al otro lado de la ecuación
    neg_mat_pure_moments = mat_pure_moments[1] * (-1)
    pm_sum = np.sum(neg_mat_pure_moments)
    m_sum = 0
    #Multiplicamos el valor de la fuerza puntual al 0 de posición en la matriz
    for i in range(0, len(mat[0])):
        m_sum += (mat[0, i]) * (mat[1, i]) 

    #Previene que el resultado sea indefinido
    if dist_rf == 0:
        return m_sum + pm_sum #Reacción angular
    else:
        return (m_sum + pm_sum)/dist_rf

#Hace sumatoria de fuerzas en y para encontrar la reacción en el segundo soporte
def force_sum_y(force_arr):
    force_sum = 0
    for i in range(0, len(force_arr)):
        force_sum += force_arr[i]

    return force_sum

#Cálcula las reacciones para empotradas y simplemente apoyadas
def calc_reactions_sup(beam_mat, sup_i, sup_f, pure_moments, diff):
    #EMPOTRADAS, el -1 en el soporte final indica que está empotrada
    if sup_f == -1:
        sup_i_idx = -1
        # Véase que en este caso no importa el orden en que se calculen las reacciones
        react_y = force_sum_y(beam_mat[1])
        react_ang = moment_sum(beam_mat, sup_i_idx, pure_moments)
        #pure_moments[1, -1] += react_ang  
        # Añadimos la reacción en y a la matriz de fuerzas en la última posición
        beam_mat[1, -1] -= react_y


        return [beam_mat, pure_moments, react_y, react_ang]

    else:
        sup_i_idx = get_idx(sup_i, diff)
        sup_f_idx = get_idx(sup_f, diff)
        # Centramos la matriz de fuerzas dónde vamos a calcular los momentos
        # Esto genera correspondencia entre la distancia al eje donde se calculan los momentos y la primera fila de la matriz de fuerzas
        dist_mat = shift_mat(beam_mat[0], sup_i)
        sh_beam_mat = np.stack((dist_mat, beam_mat[1]))
        react_sf = moment_sum(sh_beam_mat, sup_f_idx, pure_moments)
        # Añadimos la reacción en el soporte final antes de calcular suma de fuerzas en y
        beam_mat[1, sup_f_idx] -= react_sf 
        # Calculamos la segunda reacción utilizand0 sumatoria de fuerzas en y
        react_si = force_sum_y(beam_mat[1]) 
        # Añadimos la segunda reacción el el primer soporte previamente calculada
        beam_mat[1, sup_i_idx] -= react_si

        return [beam_mat, pure_moments, react_si, react_sf]

#Recibe los inputs inciales del usuario => Tipo de barra, longitud de la barra y posición de los soportes
def initial_values():
    bar_len = float(input("Ingrese la longitud de la viga en metros: "))
    diff = bar_len/(bar_len*100)
    bar_type = int(input("Ingrese el tipo de viga \n-0 => Con dos soportes \n-1 => Empotrada \nSu selección: "))
    if bar_type == 0:
        #Nota: Las posiciones se aproximan utilizando aprox diff en caso de que no existan actualmente en la discretización
        #Se le añade un pequeño valor a la longitud total de la barra a la hora de evaluar si la posición del soporte es válida para evitar floating point errors
        sup_1 = aprox_diff(float(input("Donde quiere localizar su soporte 1: ")), diff)
        while sup_1<0 or sup_1>bar_len+0.0001:
            print("Su localización de soporte está por fuerza de la barra")
            sup_1 = aprox_diff(float(input("Donde quiere localizar su soporte 1: ")), diff)

        sup_2 = aprox_diff(float(input("Donde quiere localizar su soporte 2: ")), diff)
        while sup_2<0 or sup_2>bar_len+0.0001:
            print("Su localización de soporte está por fuerza de la barra")
            sup_2 = aprox_diff(float(input("Donde quiere localizar su soporte 2: ")), diff)

        # Se garantiza que el soporte 2 sea el mayor de ambos ingresados por el usuario
        if sup_1 > sup_2:
            temp = sup_1
            sup_1 = sup_2
            sup_2 = temp

        return np.array([sup_1,sup_2,bar_len, bar_type]) 


    #Si la barra está empotrada, siempre queda empotrada a la derecha. Algunos de los valores de las posiciones de los soportes pasan a ser simbólicos
    if bar_type == 1:
        print("El análisis se realizará automáticamente con la barra empotrada a la derecha")
        sup_1 = bar_len

        return np.array([sup_1, bar_len, bar_type])

#Recibe los momentos puros que quiera ingresar el usuario y retorna una matriz con las ubicaciones y magnitud de los momentos
def pure_moments(bar_len, diff_x):
    moment_mat = gen_beam_mat(bar_len, diff_x)
    flag = input("¿Desea ingresar momentos puros? \n-y => Si \n-n => No \nSu selección: ")
    if flag == 'n':
        return moment_mat
    n = int(input("¿Cuántos momentos puros desea ingresar?: "))

    c = 1
    while n:
        n -= 1
        loc = aprox_diff(float(input(f"Ingresa la localización del momento {c} : ")), diff_x)
        moment_idx = get_idx(loc, diff_x)
        value = float(input(f"Ingrese la magnitud del momento {c} (N.m): "))
        moment_mat[1, moment_idx] += value
        c += 1

    return moment_mat

#Procesa las fuerzas que el usuario desea añadir y retorna la matriz que definiremos como la función fuerza F(x)
def process_forces(bar_len, diff_x):
    f_mat = gen_beam_mat(bar_len, diff_x) #Matriz de fuerzas
    punt_num = int(input("Ingrese la cantidad de cargas puntuales: "))
    inf_punt = np.zeros(shape=(2, punt_num), dtype=float) #Matriz de información acerca de las cargas puntuales
    #Registramos tanats cargas puntuales como el usuario quiera
    i = 1
    while punt_num:
        punt_num -= 1   
        force_loc = aprox_diff(float(input(f"Ingrese en donde quiere localizar su fuerza puntual {i}: ")), diff_x)
        force_value = float(input(f"Ingrese la fuerza puntual {i} en Newtons: "))

        #Añadimos la fuerza en la posición indicada por el usuario
        index = get_idx(force_loc, diff_x) 
        f_mat[1, index] += np.array([force_value]) 

        #Añadimos la fuerza a la matriz de información que será utilzada para graficar 
        inf_punt[0, i-1] = force_loc
        inf_punt[1, i-1] = force_value

        i += 1
     

    distr_num = int(input("¿Cuántas cargas distribuidas desea poner?: "))
    inf_dist = [[], [], [], [], []] #Matriz de información cargas distribuidas
    k = 1
    #Recibimos tantas funciones distribuidas como el usuario desee
    while distr_num:
        distr_num -= 1  

        #Recibimos la función como un string 
        math_expr_str = input(f"Ingrese la función la carga distribuida {k} (en N/m) , con la debida notación de Python: ")

        #Guardamos la función como una expresión lambda con la multiplicación por el diferencial (Para los cálculos)
        math_expr = str_to_function(math_expr_str)
        #Guardamos la función como expresión lambda sin tener en cuenta el diferencial para graficar
        math_expr_wo_dis = str_to_function_wo_dis(math_expr_str)
        begin_fun = aprox_diff(float(input(f"Ingrese la posicion inicial de la carga distribuida {k}: ")), diff_x)
        end_fun = aprox_diff(float(input(f"Ingrese la posición final de la carga distribuida {k}: ")), diff_x)
        dom_f = aprox_diff(float(input(f"Ingrese el valor inicial del dominio de la función {k}: ")), diff_x)
        dist_diff = end_fun - begin_fun #Distancia del principio de la función
        #int_len = bar_len/diff_x
        int_dom = dom_f/diff_x #Dominio en enteros
        int_dist = dist_diff/diff_x #Distancia en enteros
        f = lambda x: x*diff_x #Función para devolver el diferencial

        #Creamos un arange con todos los puntos del dominio donde se quiera evaluar decuerdo a nuestra discretización 
        row_diff = np.arange(int_dom, int_dom + int_dist + 1, 1)
        row_diff = f(row_diff)
        #row_diff = np.arange(dom_f ,dom_f + dist_diff + diff_x, diff_x)

        #Evaluamos la función en todos los puntos del dominio donde ha indicado el usuario
        y_images = math_expr(row_diff)

        #Quitamos las discontinuidades en caso de que se den
        y_images = remove_discon(y_images)

        ini_idx = get_idx(begin_fun, diff_x)

        #Añadimos los resultados de la función evaluada en la discretización (Esto implica tomar cada punto evaluado como una carga puntual)
        j=0
        for i in range(ini_idx, ini_idx+ len(y_images)):
            f_mat[1, i] += y_images[j]
            j+=1

        #Guardamos la información relevante en la matriz de información para graficar
        inf_dist[0].append(math_expr_wo_dis)
        inf_dist[1].append(begin_fun)
        inf_dist[2].append(end_fun)
        inf_dist[3].append(dom_f)
        inf_dist[4].append(math_expr_str)
        k += 1

    #Retornamos la matriz de fuerzas y las matrices de información
    return f_mat, inf_punt, inf_dist

#Calcula las fuerzas cortantes una vez todas las fuerzas están interpretadas como puntuales
#Mat es la matriz de fuerzas con reacciones
def calc_sheer_forces(mat, bar_len, diff):
    sheer_mat = gen_beam_mat(bar_len, diff)
    #No se multiplica el diferencial al integral puesto que ya está procesado en str_to_function y process_forces
    acum_integral = 0
    for i in range(0, len(mat[0])):
        acum_integral += mat[1, i]
        sheer_mat[1, i] = acum_integral

    #Multiplicamos todo por (-1) por definición {T(x) = -integral(F(x)), T(x)=>Cortantes, F(x)=>Fuerzas}
    sheer_mat[1] *= (-1)
    return sheer_mat

#Suma de riemann tomando elementos a la derecha
def riemann_sum(diff_x, y_images):
    integral = np.zeros(len(y_images))
    Area = 0
    #Multiplicamos los valores de y por el diferencial
    for i in range(0, len(y_images)):
        Area += (diff_x)*(y_images[i])
        integral[i] = Area
    i += 1
    return integral

#Añadimos los momentos puros a la matriz de momentos y la retornamos modifciada
def add_pure_moments(moment_mat, bending_mat, b_len, x_diff):
    int_len = b_len/x_diff
    f = lambda x: x*x_diff
    row_diff = np.arange(0, int_len + 1, 1)
    row_diff = f(row_diff)
    bending_mat = np.stack([row_diff, bending_mat])
    #Se utiliza suma para saber cual es el efecto de los momentos puros en los datos para y
    #Los momentos no solo afectan la posición actual, sino también todas las siguientes
    #Por eso se debe sumar sum a todos los elementos de la matriz
    sum = 0
    for i in range(len(bending_mat[0])):
        sum += moment_mat[1, i] * (-1)
        bending_mat[1, i] += sum
    return bending_mat[1]

#Recibe información adicional de la viga para realizar análisis de esfuerzos
def aditional_info():
    decision = input("Desea considerar el perfil de la viga? \n -y => Si \n -n => No \n Su selección: ")

    if decision == "n":
        my_S = 0
        my_Q = 0 
        my_I = 0
        my_t = 0 
        my_r = 0
        decision2 = 0
    elif decision == "y":
        decision2 = input("Ingrese el perfil que quiere para su viga \n -r => Rectangular \n -w => Tipo H \n -c => Circular \n Su selección:")
        if decision2 == "w":
            print("El perfil a considerar será de tipo W, el cual es una viga tipo H. \n Revise el anexo de excel llamado SpecsTipo_H. ")
            my_I = float(input("Ingrese el momento de inercia respecto a x (I) (Columna DQ): "))
            my_Q = float(input("Ingrese el primer momento respecto al eje centroidal del área de la sección transversal (Q) (Columna EI): "))
            my_t = float(input("Ingrese el espesor del alma de la viga (t) (Columna CU): "))
            my_S = float(input("Ingrese el módulo de sección (S) (Columna DS): "))
            my_r = 0
        if decision2 == "c":
            decision3 = input("El perfil a considerar será de tipo circular\n ¿Quiere calcular usted mismo las especificaciones de la viga? \n -y => Si \n -n => No \n Su selección:")
            if decision3 == "y":
                print("Revise el anexo tipo PDF llamado especificaciones \n Siendo r el radio del círculo. Ingrese:  ")
                my_Q = 1
                my_I = 1
                my_Q = 1
                my_t = 1
                my_r = float(input("Ingrese el radio del círculo: "))
                my_S = float(input("Ingrese el módulo de sección (S): "))
            elif decision3 == "n":
                print("Revise el anexo tipo PDF llamado especificaciones \n Siendo r el radio del círculo. Ingrese:  ")
                my_Q = 1
                my_t = 1
                my_r = float(input("Ingrese el radio del círculo: "))
                #Para una sección transversal de un semicirculo se calcula el momento de inercia para el mismo, estandarizado por el libro.
                my_I = ((1/8)*pi)*(my_r**4)
                #De manera analoga, se saca el módulo de sección para esta gemetría que está estandarizado debido al momento de inercia y el radio.
                my_S = my_I/my_r
        elif decision2 == "r":
            decision3 = input("El perfil a considerar será de tipo rectangular \n ¿Quiere calcular usted mismo las especificaciones de la viga? \n -y => Si \n -n => No \n Su selección:")
            if decision3 == "y":
                print("Revise el anexo tipo PDF llamado especificaciones \n Siendo t el espesor de la viga y h su altura, calcule como se muestra. ")
                my_t = float(input("Ingrese el espesor de la viga (t): "))
                my_h = float(input("Ingrese la altura de la viga (h): "))
                my_I = float(input("Ingrese el momento de inercia respecto a x (I): "))
                my_Q = float(input("Ingrese el primer momento respecto al eje centroidal del área de la sección transversal (Q): "))
                my_S = float(input("Ingrese el módulo de sección de la viga (S): "))
                my_r = 0
            elif decision3 == "n":
                print("Revise el anexo tipo PDF llamado especificaciones \n Siendo t el espesor de la viga y h su altura, ingrese según la tabla. ")
                my_t = float(input("Ingrese el espesor de la viga (t): "))
                my_h = float(input("Ingrese la altura de la viga (h): "))
                #El momento de inercia es calculado según el espesor y la altura de la sección transversal
                my_I = ((my_t)*(my_h**3))/12
                #De manera analoga, se calcula el módulo de sección para esta gemetría que está estandarizado debido al momento de inercia y la altura.
                my_S = my_I/(my_h/2)
                my_r = 0
                #Se calcula el primer momento de área para la sección transversal escogida.
                my_Q = (my_t*(my_h**2))/8
    return np.array([my_S, my_Q, my_I, my_t, my_r, decision, decision2]) 

#Calcula el esfuerzo debido a flexión máximo
def max_sigma(moments, section):
    moment = np.zeros(2)
    moment[0] = abs(max(moments))
    moment[1] = abs(min(moments))
    max_moment = max(moment)
    #Teniendo el módulo de sección hallado en la función anterior y el momento máximo, se calcula el esfuerzo debido a flexión máximo y se representa e punto en la barra.
    sigma_max = (max_moment)/(section)
    return sigma_max

#Calcula el esfuerzo cortante máximo
def max_tao(sheers, Q, I, t, type, r):
    sheer = np.zeros(2)
    sheer[0] = abs(max(sheers))
    sheer[1] = abs(min(sheers))
    max_sheer = max(sheer)
    tao_max = 0
    if type == "r" or "w":
        #El tao maximo se calcula gracias al cortante maximo, el primer momento de área de la sección transversal, su momento de inercia y el espesor de la viga.
        tao_max = (max_sheer*Q)/(I*t)
    elif type == "c":
        #Para una sección circular, se calcula según su área y el cortante máximo.
        tao_max = (4*max_sheer)/(3*(pi*(r**2)))
    return tao_max

# Función para hallar los puntos inferiores de los soportes, para graficar
def find_point(z):
    bajo_der = [z[0] + 1, z[1] - 3]
    bajo_iz = [z[0] - 1, z[1] - 3]
    return bajo_der, bajo_iz

# Función para la gráfica ilustrativa de fuerzas en una viga
def gen_graph(bar_l, bar_type, x_sup, der_sup, iz_sup, x2_sup, der_sup2, iz_sup2, forc_m, distrib, moments,
              rea_1, rea_2):

    #Se establece el estilo de la gráfica
    with plt.style.context('ggplot'):

        #Diseño de la gráfica (ejes y títulos)
        plt.xlim([-2, bar_l + 2])
        plt.ylim([-8, 30])
        plt.suptitle("Gráfica de cargas sobre una barra apoyada", fontsize=12, fontweight='bold',
                         color="darkslateblue")
        plt.xlabel("Distancia (m)")

        # Se crea un condicional para barras con soportes, con patches de triánglulos
        if bar_type == 0:
            #Soporte 1, con un polígono de tres lados
            list_trian = [x_sup, der_sup, iz_sup]
            trian = mpatches.Polygon(list_trian, zorder=2, color="darkseagreen")
            plt.gca().add_patch(trian) #Call para añadir el patch

            # soporte2
            list_trian2 = [x2_sup, der_sup2, iz_sup2]
            trian2 = mpatches.Polygon(list_trian2, zorder=2, color="darkseagreen")
            plt.gca().add_patch(trian2)

            #Si es una barra empotrada, se añade un patch de pared
        else:
            wall = mpatches.Rectangle((bar_l, -10), bar_l, 25, zorder=2, color="teal")
            plt.gca().add_patch(wall)


        # Se dibuja la barra, módulo patches rectangle
        # Dar coordenada de inicio, largo en y, longitud en x y grosor. zorder pone adelante la figura
        rect = mpatches.Rectangle((0, -3), bar_l, 3, zorder=2, color="lightslategrey")
        plt.gca().add_patch(rect)

        #Se definen las variables de las cargas, para poder graficarlas ilustrativamente
        # Variables fuerzas puntuales
        loc_f = forc_m[0]
        value_f = forc_m[1]

        # Variables distribuidas
        math_exp = distrib[0]
        begin_f = distrib[1]
        end_f = distrib[2]
        domain = distrib[3]
        str_m = distrib[4]

        # Variables momento
        mom = moments[0]
        values_m_mat = moments[1]
        value = np.nonzero(values_m_mat)  #tupla
        value_m = value[0] #tiene lista de posiciones

        #Se extrae el valor de posición del momento, de la fila de magnitudes de momentos
        values = []
        for t in value_m:
            values_magnitude = values_m_mat[t].astype(int)
            values.append(values_magnitude)

        #saber las posiciones del eje x, con los indexes anteriores. Para poder graficar x y y
        positions_moments = []
        for p in values:
            position = mom[p]
            mul = position*100 #*100 porque previamente se le había dividido el diferencial
            positions_moments.append(mul)

        # Dibujar fuerzas puntuales como flechas
        for i in range(0, len(loc_f)):

            col = (np.random.random(), np.random.random(), np.random.random())  # genera colores random

            # dibuja una flecha con coordenadas iniciales a finales
            plt.arrow(loc_f[i], 5, 0, -3.5, zorder=2, color=col, width=0.3, label=f"{value_f[i]} N")

        # Dibujar distribuidas, como funciones
        for c in range(0, len(distrib[0])):

            f = math_exp[c]
            begin = begin_f[c]
            end = end_f[c]
            dom = domain[c]
            shift = begin - dom #dISTANCIA DEL DOM AL BEGIN FUN
            dif_f = end - begin

            xpts = np.linspace(dom, dom + dif_f, 50)
            evalu = f(xpts)
            xpts += shift

            col = (np.random.random(), np.random.random(), np.random.random())  # genera colores random

            plt.plot(xpts, evalu, zorder=2, color=col, label=str_m[c])
            plt.fill_between(xpts, 0, evalu, color=col, alpha=0.2) #Rellena con color las funciones

        # Dibujar momentos, como flechas curvadas
        for z in range(0, len(positions_moments)):
            col = (np.random.random(), np.random.random(), np.random.random())
            mom_n = mpatches.FancyArrowPatch((positions_moments[z] + 2, 0), (positions_moments[z] - 2, 0),
                                            connectionstyle="arc3,rad=0.7", zorder=2,
                                            mutation_scale=11, color=col, linewidth=1, label=f" {values[z]} Nm")
            plt.gca().add_patch(mom_n)

        # Se añade un marcador para indicar la deflexión máxima de la barra

        if bar_type == 0:

            plt.plot(-100, -100, marker="2", color="firebrick", zorder=0, markersize=11, label=f'Reac. sup 1:\n{round(rea_1,2)} N',alpha=1)
            plt.plot(-100, -100, marker="2", color="blue", zorder=0, markersize=11, label=f'Reac. sup 2:\n{round(rea_2,2)} N', alpha=1)

        else:
            plt.plot(-100, -100, marker="2", color="firebrick", zorder=0, markersize=11, label=f'Reac. en y:\n{round(rea_1,2)} N',alpha=1)
            plt.plot(-100, -100, marker="2", color="blue", zorder=0, markersize=11, label=f'Reac. ang:\n{round(rea_2,2)} Nm', alpha=1)


        plt.legend(fancybox=True, loc='upper right', shadow=True)
        plt.subplots_adjust(left=0.135,
                                    bottom=0.125,
                                    right=0.905,
                                    top=0.9,
                                    wspace=0.2,

                                    hspace=0.205)
        plt.minorticks_on()  # se activan rayas en los ejes para dar presición a las medidas
        plt.tick_params(which='major',  # Options for both major and minor ticks
                            bottom='off')  # turn off bottom ticks

        plt.grid(color='white')

        plt.show()

def cool_graphs(mat, mat_m, dis, bar_l, sigma, tao):
    x = mat[0]
    y = mat[1]

    # Encontrar coordenadas xy y del cortante max
    max_ref = np.max(y)
    abs_cort = np.absolute(y)
    max_cort = np.max(abs_cort)

    if max_ref < max_cort:
        max_cort *= -1

    punto_xc = np.where(y == max_cort)
    punto_x = x[punto_xc[0]][0]
    integral_list = riemann_sum(dis, y)

    if np.sum(mat_m[1]) != 0:
        integral_list = add_pure_moments(mat_m, integral_list, bar_l, dis)

    # Encontrar coordenadas x y del flexionante máximo
    max_refc = np.max(integral_list)
    abs_flex = np.absolute(integral_list)
    max_flex = np.max(abs_flex)

    if max_refc < max_flex:
        max_flex *= -1

    punto_xf = np.where(integral_list == max_flex)
    punto_xflex = x[punto_xf[0]][0]


    with plt.style.context('ggplot'):
        # se crea la ventana de subplots
        fig, axs = plt.subplots(2)
        fig.suptitle('Gráficas de análisis de viga', fontsize=12, fontweight='bold', color="mediumslateblue")

        # grafica cortantes
        func = axs[0].plot(np.concatenate([[0], x]), np.concatenate([[0], y]), color="red", label="función cortante")
        axs[0].set_xlabel('Puntos de la viga (m)')
        axs[0].set_ylabel('Cortantes (N)')
        axs[0].grid(which='major', linestyle='-', linewidth='1.2', color='white')
        axs[0].set_title("Gráfica de fuerzas cortantes", fontsize=9, fontweight='bold', color="darkslategrey")
        axs[0].minorticks_on()
        axs[0].tick_params(which='major',  # Options for both major and minor ticks
                           left='on',  # turn off left ticks
                           bottom='off')  # turn off bottom ticks
        # Dibujar punto máximo
        maxi = round(max_cort, 1)
        axs[0].annotate((round(punto_x, 2), round(max_cort, 2)), xy=(punto_x, max_cort), xytext=(round(punto_x, 2), round(max_cort, 2)))
        axs[0].plot(punto_x, max_cort, marker="D", color="green", label=f'Cortante máximo\n{maxi}\nTao máx.\n{tao}')
        axs[0].fill_between(x, 0, y, color="red", alpha=0.2)  # colorear entre función y eje x
        axs[0].legend(func, [punto_x, max_cort])
        axs[0].legend(fancybox=True, loc='center left', shadow=True, bbox_to_anchor=((1,0.5)))

        # grafica flexionantes
        axs[1].plot(x, integral_list, color="navy", label="función flex.")
        axs[1].set_xlabel('Puntos de la viga (m)')
        axs[1].set_ylabel('Flexionantes (N)')
        axs[1].grid(which='major', linestyle='-', linewidth='1.2', color='white')
        axs[1].set_title("Gráfica de fuerzas flexionantes", fontsize=9, fontweight='bold', color="darkslategrey")
        axs[1].minorticks_on()
        axs[1].tick_params(which='major',  # Options for both major and minor ticks
                           left='on',  # turn off left ticks
                           bottom='off')  # turn off bottom ticks

        axs[1].fill_between(x, 0, integral_list, color="blue", alpha=0.2)

        # punto max flexionante
        maxi_f = round(max_flex, 1)
        axs[1].annotate(( round(punto_xflex, 2), round(max_flex, 2)), xy=(punto_xflex, max_flex),
                        xytext=(round(punto_xflex, 2), round(max_flex, 2)))
        axs[1].plot(punto_xflex, max_flex, marker="D", color="salmon", label=f'flex. máximo\n{maxi_f}\nSigma máx.\n{sigma}')
        axs[1].legend(func, [punto_xflex, max_flex])
        axs[1].legend(fancybox=True, loc='center left', shadow=True, bbox_to_anchor=((1, 0.5)))

        plt.subplots_adjust(left=0.13,
                            bottom=0.15,
                            right=0.7,
                            top=0.87,
                            wspace=0.675,
                            hspace=0.695)

        plt.show(block=False)
        plt.pause(9900)