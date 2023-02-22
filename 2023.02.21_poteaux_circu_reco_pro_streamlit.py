
"""
Notes : 
    
    - Dimensionnement de poteaux en béton armé selon Recommandations Professionnelles
    - Objectif : donner les sections minimales de poteaux circulaires/rectangulaires selon différentes nuances de béton
    - Ratio : On pourra le déterminer manuellement. En première approche, on part sur 200 kg/m3 soit ratio de 0.025
    - kh / ks = 1.0 - A affiner dans une version ultérieure si besoin
    - Elancement : H0 déterminé comme un poteau bi-articulé : H0 = H
   
    
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ________________________________ IMPORT DONNEES _______________________________________________________

st.title('Poteaux circulaires béton - Recommandations Professionnelles')


# Choix utilisateurs
st.subheader('Hypothèses pour votre poteau')
hauteur=st.slider("Hauteur du poteau (m):", min_value=1.5, max_value=20.0, value=2.5, step=0.2)
effort_elu=st.slider("Effort ELU (kN) :", min_value=500, max_value=10000, value=1000, step=100)

list_fck_alpha=['C16/20','C20/25','C25/30','C30/40','C35/45','C40/50','C45/55','C50/60','C55/65','C60/70']
list_fck_num=[16,20,25,30,35,40,45,50,55,60]
nuance_beton=st.select_slider('Nuance de béton :',options=list_fck_alpha,value='C25/30')
fck=int(list_fck_num[list_fck_alpha.index(nuance_beton)])
ferraillage=st.slider('Ferraillage du poteau (kg/m3):',min_value=0, max_value=310, value=20, step=20)
ratio_ui=ferraillage/7800
st.write("Le ratio de ferraillage du poteau est de {}".format(round(ratio_ui,4)))
list_phi_circ=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2]


#  _______________________________  FONCTIONS  ______________________________________________________

def elancement_circ(L0, phi):
    return L0 * 4 / phi
# def elancement_rect(b,L0):
#     return L0*np.sqrt(12)/b

def alpha_circ(x):
    if x <= 60:
        return 0.84 / (1 + (x / 52) ** 2)
    elif 60 < x <= 120:
        return (27 / x) ** 1.24
    else:
        return 0
vec_alpha_circ = np.vectorize(alpha_circ)

# def alpha_rect(x):
#     if x<=60:
#         return 0.86/(1+(x/62)**2)
#     elif 60<x<=120:
#         return (32/x)**1.3
#     else:
#         return 0
def NRd_circ(alpha, fck, phi,  ratio, kh=1, ks=1):
    return 1000 * (kh * ks * alpha * (np.pi * phi ** 2 / 4 * fck / 1.5 + (ratio) * np.pi * phi ** 2 / 4 * 500 / 1.15))

def Phi_min(N, h, fck, ratio, kh=1, ks=1):
    for i in range(len(list_phi_circ)):
        phi = list_phi_circ[i]
        elanc = elancement_circ(h, phi)
        alpha = alpha_circ(elanc)
        nrd=NRd_circ(alpha=alpha,fck=fck,phi=phi,ratio=ratio)
        if nrd<N:
            i = i + 1
        else:
            #phi_min = np.sqrt(4 * (N / 1000) / (np.pi * (kh * ks * alpha * (fck / 1.5 + (ratio) * 500 / 1.15))))
            return phi

vec_phi_min = np.vectorize(Phi_min)  # Vectorisation de la fonction Phi_min
result_phi_min=Phi_min(N=effort_elu, h=0.7*hauteur, fck=fck, ratio=ratio_ui)
elancement_phi_min=elancement_circ(L0=.7*hauteur,phi=result_phi_min)
alpha_phi_min=alpha_circ(elancement_phi_min)
effort_resistant=NRd_circ(alpha=alpha_phi_min,fck=fck,phi=result_phi_min,ratio=ratio_ui)

# Résultats
st.subheader("Résultat")
st.write("Le diamètre minimum du poteau à considérer avec vos hypothèses ci-dessus est de   {} cm".format(round(result_phi_min*100,0)))
st.write('Effort sollicitant : {} kN'.format(float(effort_elu)))
st.write('Effort résistant : {} kN'.format(round(effort_resistant,0)))

# Graphique
list_nrd=[]
for element in list_phi_circ:
    elancement = elancement_circ(L0=0.7 * hauteur, phi=element)
    alpha = alpha_circ(elancement)
    list_nrd.append(NRd_circ(alpha=alpha,fck=fck,phi=element,ratio=ratio_ui))

if result_phi_min<list_phi_circ[-1]:
    if result_phi_min<0.6:
        fig = plt.figure()
        plt.axhline(y=effort_elu,color='r',label='effort sollicitant (kN)')
        plt.scatter(x=[100*i for i in list_phi_circ[:10]], y=list_nrd[:10], label='effort résistant (kN)')
        plt.xlabel('Diamètres poteau (cm)')
        plt.xticks(ticks=[100*i for i in list_phi_circ[:10]])
        plt.ylabel('Effort (kN)')
        plt.legend()
        plt.title('Efforts sollicitant et résistants du poteau étudié')
        st.pyplot(fig)
    else:
        fig = plt.figure()
        plt.axhline(y=effort_elu,color='r',label='effort sollicitant (kN)')
        plt.scatter(x=[100*i for i in list_phi_circ[10:len(list_phi_circ)]], y=list_nrd[10:len(list_phi_circ)], label='effort résistant (kN)')
        plt.xlabel('Diamètres poteau (cm)')
        plt.xticks(ticks=[100*i for i in list_phi_circ[10:len(list_phi_circ)]])
        plt.ylabel('Effort (kN)')
        plt.legend()
        plt.title('Efforts sollicitant et résistants du poteau étudié')
        st.pyplot(fig)
else:
    pass
