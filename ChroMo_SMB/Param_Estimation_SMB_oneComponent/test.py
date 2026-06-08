import numpy as np
from objectiveSMB_oneComp import objectiveSMB

'''Optimization Results for Component A: 
{'henry_constant': 0.48553981993246786, 
'delta': 26.721043567912183, 
'MSE': 0.2075520057647311}
'''

'''Optimization Results for Component B: 
{'henry_constant': 0.4607646162383974, 
'delta': 30.385638090786898, 
'MSE': 0.1069269516992426}
'''

'''Optimization terminated successfully.
         Current function value: 0.119733
         Iterations: 90
         Function evaluations: 164
Optimization Results for Component Man: {'henry_constant': 0.705995582355348, 'delta': 22.1579395236581, 'porosity': 0.11590377100536634, 'MSE': 0.1197325382475663}'''

'''Two peaks in Raff
Optimization Results for Component Man: {'henry_constant': 0.6100050077679293, 'delta': 382.7138990312855, 'porosity': 0.032958080178846216, 'MSE': 262.2339158079607}'''

objectiveSMB(10000, 'SMB_onePeriond_experiment5_RI.xlsx', name_compA="Man", porosity=0.03295, henry_constantA=0.6100050077679293, deltaA=382.7138990312855,
                  name_compB="Gal", henry_constantB=0.5140, deltaB=17.5510, switch_interval=780,
                  optimize_component='Man', save_plot=True, plotting=True)

'''objectiveSMB(10000, 'SMB_onePeriond_experiment5_RI.xlsx', name_compA="Man", henry_constantA=0.5436, deltaA=14.6589,
                  name_compB="Gal", henry_constantB=6.63, deltaB=13.51, switch_interval=780,
                  optimize_component='Gal', save_plot=True)'''

