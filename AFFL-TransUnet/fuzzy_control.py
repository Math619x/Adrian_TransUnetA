import numpy as np

import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Definir antecedentes (entradas) y consecuente (salida)
class_sizes = ctrl.Antecedent(np.linspace(0, 1, num=1001), 'class_sizes')
#focal_fuzzy = ctrl.Antecedent(np.linspace(0, 0.5, num=1001), 'focal_fuzzy')
focal_delta = ctrl.Antecedent(np.linspace(-0.3, 0.3, num=1001), 'focal_delta')

#new_gamma = ctrl.Consequent(np.linspace(-0.4, 0.4, num=1001), 'new_gamma')
#Esto es para sustituir directamente un alpha
new_gamma = ctrl.Consequent(np.linspace(-0.5, 0.5, num=1001), 'new_gamma')



# Definir funciones de membresía para las entradas y la salida
class_sizes['pocos'] = fuzz.trimf(class_sizes.universe, [0.0, 0.0, 0.03])
class_sizes['masomenos'] = fuzz.trimf(class_sizes.universe, [0.0, 0.03, 0.1])
class_sizes['muchos'] = fuzz.trimf(class_sizes.universe, [0.09, 1.0, 1.0])

#class_sizes.automf(3)  
#focal_fuzzy.automf(5)  
#focal_delta.automf(5)
new_gamma.automf(5)

#primer ajuste de las reglas, para ver si no afecta a los resultados los rangos de la perdida focal.
focal_delta['v_negative'] = fuzz.trimf(focal_delta.universe, [-0.3, -0.3, -0.03])
focal_delta['negative'] = fuzz.trimf(focal_delta.universe, [-0.07, -0.03, 0])
focal_delta['neutro'] = fuzz.trimf(focal_delta.universe, [-0.03, 0, 0.0015])
focal_delta['positive'] = fuzz.trimf(focal_delta.universe, [0, 0.0015, 0.005])
focal_delta['v_positive'] = fuzz.trimf(focal_delta.universe, [0.0015, 0.35, 0.35])

focal_delta.view()

#poor, mediocre, average, decent, good

rules = []
# rules.append(ctrl.Rule(error['left'], deviation['positive']))


#Reglas 4 - funcionan chido con la instruccion alpha = alpha_cero + delta_alpha 
#Se hizo el cambio en las ultimas dos reglas de cada grupo de 5, para que tengan sentido
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['v_negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['neutro'], new_gamma['decent']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['positive'], new_gamma['good']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['v_positive'], new_gamma['good']))

rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['v_negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['neutro'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['positive'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['v_positive'], new_gamma['decent']))

rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['v_negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['neutro'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['positive'], new_gamma['mediocre']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['v_positive'], new_gamma['mediocre']))


# Crear el sistema de control utilizando las reglas de inferencia
gamma_ctrl = ctrl.ControlSystem(rules)
gamma_ = ctrl.ControlSystemSimulation(gamma_ctrl)


def act_alpha(class_size, focal_n0, focal_n1):
    
    delta = focal_n1-focal_n0
    
    #print(np.clip(delta, -0.3, 0.3))
    
    gamma_.input['class_sizes'] = np.clip(class_size, 0, 1)
    #gamma_.input['focal_fuzzy'] = np.clip(focal_n0, 0, 0.5)
    gamma_.input['focal_delta'] = np.clip(delta, -0.3, 0.3)
    
    gamma_.compute()
    return gamma_.output['new_gamma']


#################### TODO AQUI DEBAJO ES LAS PRUEBAS (NO SE USAN)
# focal_delta['v_negative'] = fuzz.trimf(focal_delta.universe, [-0.3, -0.3, -0.2])
# focal_delta['negative'] = fuzz.trimf(focal_delta.universe, [-0.3, -0.2, -0.1])
# focal_delta['neutro'] = fuzz.trimf(focal_delta.universe, [-0.2, 0, 0.2])
# focal_delta['positive'] = fuzz.trimf(focal_delta.universe, [0.1, 0.2, 0.3])
# focal_delta['v_positive'] = fuzz.trimf(focal_delta.universe, [0.2, 0.3, 0.3])

"""
#reglas 1
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['v_negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['negative'], new_gamma['decent']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['neutro'], new_gamma['good']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['positive'], new_gamma['mediocre']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['v_positive'], new_gamma['poor']))

rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['v_negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['negative'], new_gamma['decent']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['neutro'], new_gamma['good']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['positive'], new_gamma['mediocre']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['v_positive'], new_gamma['poor']))

rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['v_negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['neutro'], new_gamma['decent']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['positive'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['v_positive'], new_gamma['mediocre']))



#reglas 2
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['v_negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['negative'], new_gamma['decent']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['neutro'], new_gamma['good']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['positive'], new_gamma['mediocre']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['v_positive'], new_gamma['poor']))

rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['v_negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['negative'], new_gamma['decent']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['neutro'], new_gamma['good']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['positive'], new_gamma['mediocre']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['v_positive'], new_gamma['mediocre']))

rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['v_negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['neutro'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['positive'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['v_positive'], new_gamma['poor']))
#rules[0].view()

#Reglas 3 - funcionan chido con la instruccion alpha = alpha_cero + delta_alpha 
#la probare con el signo negativo
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['v_negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['neutro'], new_gamma['poor']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['positive'], new_gamma['mediocre']))
rules.append(ctrl.Rule(class_sizes['pocos'] & focal_delta['v_positive'], new_gamma['mediocre']))

rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['v_negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['neutro'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['positive'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['masomenos'] & focal_delta['v_positive'], new_gamma['decent']))

rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['v_negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['negative'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['neutro'], new_gamma['average']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['positive'], new_gamma['decent']))
rules.append(ctrl.Rule(class_sizes['muchos'] & focal_delta['v_positive'], new_gamma['decent']))
"""