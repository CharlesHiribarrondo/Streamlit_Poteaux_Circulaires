
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


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ________________________________ IMPORT DONNEES _______________________________________________________

st.title('Poteaux circulaires béton - Recommandations Professionnelles')


# Choix utilisateurs
hauteur=st.slider("Hauteur du poteau (m):", min_value=1.5, max_value=20.0, value=2.5, step=0.2)
effort_elu=st.slider("Effort ELU (kN) :", min_value=100, max_value=1500, value=350, step=50)

list_fck_alpha=['C16/20','C20/25','C25/30','C30/40','C35/45','C40/50','C45/55','C50/60','C55/65','C60/70']
list_fck_num=[16,20,25,30,35,40,45,50,55,60]
nuance_beton=st.select_slider('Nuance de béton :',options=list_fck_alpha,value='C25/30')
ferraillage=st.slider('Ferraillage du poteau (kg/m3):',min_value=0, max_value=310, value=20, step=20)



'''

pd_data=pd.read_excel("Desktop/2022.04.14 - Poteaux_data.xlsx")
print(pd_data)
print(pd_data.dtypes)


pot_df=pd.DataFrame(
    {
     "Identifiant":pd_data["Identifiant"],
     "Hauteur":pd_data["Hauteur"],
     "Effort ELU":pd_data["Effort ELU"],
     "Ratio ferraillage":pd.Series([25]*len(pd_data)),
     "kh":pd.Series([1.0]*len(pd_data)),
     "ks": pd.Series([1.0]*len(pd_data)),
     }
    )

print(pot_df)

# ________________________________ HYPOTHESES _______________________________________________________

list_fck=[16,20,25,30,35,40,45,50,55,60]
list_phi_circ=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3]

#  _______________________________  FONCTIONS  ______________________________________________________

def elancement_circ(L0,phi):
    return L0*4/phi

# def elancement_rect(b,L0):
#     return L0*np.sqrt(12)/b

def alpha_circ(x):
    if x<=60:
        return 0.84/(1+(x/52)**2)
    elif 60<x<=120:
        return (27/x)**1.24
    else:
        return 0

vec_alpha_circ=np.vectorize(alpha_circ)

   
# def alpha_rect(x):
#     if x<=60:
#         return 0.86/(1+(x/62)**2)
#     elif 60<x<=120:
#         return (32/x)**1.3
#     else:
#         return 0

def NRd_circ(alpha,fck,phi,kh,ks,ratio):
    return 1000*(kh*ks*alpha*(np.pi*phi**2/4*fck/1.5+(ratio/1000)*np.pi*phi**2/4*500/1.15))

def Phi_min(N,h,fck,kh,ks,ratio):
    for i in range(len(list_phi_circ)):
        phi=list_phi_circ[i]
        elanc=elancement_circ(h, phi)
        alpha=alpha_circ(elanc)
        if alpha==0:
            i=i+1
        else:   
            phi_min=np.sqrt(4*(N/1000)/(np.pi*(kh*ks*alpha*(fck/1.5+(ratio/1000)*500/1.15))))
            if phi_min>phi:
                i=i+1
            else: return list_phi_circ[i]

vec_phi_min=np.vectorize(Phi_min) # Vectorisation de la fonction Phi_min


# ___________________________ DIMENSIONNEMENT _____________________________________________________________

df1=pot_df.copy() # On duplique la Dataframe pour supprimer la 1ere colonne et obtenir que des chiffres
df1.pop("Identifiant") # On supprime la colonne des identifiants, pour ne conserver que des chiffres

df2=df1.to_numpy(dtype=int) # On transforme le Dataframe en Matrice Numpy pour effectuer des opérations mathématiques ensuite

# -----------------------------  C16 --------------------------------------------------------------------------------
fck_16=list_fck[0]
col1_16=np.array([[fck_16]]*df2.shape[0]).transpose() # Ajout colonne Fck
col2_16=vec_phi_min(np.array([df2[:,1]]),np.array([df2[:,0]]),col1_16,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])) # Ajout colonne Diametre mini
col3_16=(np.array(elancement_circ(df2[:,0], col2_16))) # Ajout colonne Elancement
col4_16=vec_alpha_circ(col3_16)
col5_16=np.array(NRd_circ(col4_16,col1_16,col2_16,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])))
ar16=np.hstack((df2,col1_16.transpose(),col2_16.transpose(),col3_16.transpose(),col4_16.transpose(),col5_16.transpose()))

df16=pot_df.copy()
df16["Fck"]=ar16[:,5]
df16["Diametre mini"]=ar16[:,6]
df16["Elancement"]=ar16[:,7]
df16["Alpha"]=ar16[:,8]
df16["NRd"]=ar16[:,9]
df16["Critere validé ?"]=(ar16[:,9]>ar16[:,1])

# -----------------------------  C20 --------------------------------------------------------------------------------
fck_20=list_fck[1]
col1_20=np.array([[fck_20]]*df2.shape[0]).transpose() # Ajout colonne Fck
col2_20=vec_phi_min(np.array([df2[:,1]]),np.array([df2[:,0]]),col1_20,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])) # Ajout colonne Diametre mini
col3_20=(np.array(elancement_circ(df2[:,0], col2_20))) # Ajout colonne Elancement
col4_20=vec_alpha_circ(col3_20)
col5_20=np.array(NRd_circ(col4_20,col1_20,col2_20,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])))
ar20=np.hstack((df2,col1_20.transpose(),col2_20.transpose(),col3_20.transpose(),col4_20.transpose(),col5_20.transpose()))

df20=pot_df.copy()
df20["Fck"]=ar20[:,5]
df20["Diametre mini"]=ar20[:,6]
df20["Elancement"]=ar20[:,7]
df20["Alpha"]=ar20[:,8]
df20["NRd"]=ar20[:,9]
df20["Critere validé ?"]=(ar20[:,9]>ar20[:,1])

# -----------------------------  C25 --------------------------------------------------------------------------------
fck_25=list_fck[2]
col1_25=np.array([[fck_25]]*df2.shape[0]).transpose() # Ajout colonne Fck
col2_25=vec_phi_min(np.array([df2[:,1]]),np.array([df2[:,0]]),col1_25,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])) # Ajout colonne Diametre mini
col3_25=(np.array(elancement_circ(df2[:,0], col2_25))) # Ajout colonne Elancement
col4_25=vec_alpha_circ(col3_25)
col5_25=np.array(NRd_circ(col4_25,col1_25,col2_25,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])))
ar25=np.hstack((df2,col1_25.transpose(),col2_25.transpose(),col3_25.transpose(),col4_25.transpose(),col5_25.transpose()))

df25=pot_df.copy()
df25["Fck"]=ar25[:,5]
df25["Diametre mini"]=ar25[:,6]
df25["Elancement"]=ar25[:,7]
df25["Alpha"]=ar25[:,8]
df25["NRd"]=ar25[:,9]
df25["Critere validé ?"]=(ar25[:,9]>ar25[:,1])

# -----------------------------  C30 --------------------------------------------------------------------------------
fck_30=list_fck[3]
col1_30=np.array([[fck_30]]*df2.shape[0]).transpose() # Ajout colonne Fck
col2_30=vec_phi_min(np.array([df2[:,1]]),np.array([df2[:,0]]),col1_30,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])) # Ajout colonne Diametre mini
col3_30=(np.array(elancement_circ(df2[:,0], col2_30))) # Ajout colonne Elancement
col4_30=vec_alpha_circ(col3_30)
col5_30=np.array(NRd_circ(col4_30,col1_30,col2_30,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])))
ar30=np.hstack((df2,col1_30.transpose(),col2_30.transpose(),col3_30.transpose(),col4_30.transpose(),col5_30.transpose()))

df30=pot_df.copy()
df30["Fck"]=ar30[:,5]
df30["Diametre mini"]=ar30[:,6]
df30["Elancement"]=ar30[:,7]
df30["Alpha"]=ar30[:,8]
df30["NRd"]=ar30[:,9]
df30["Critere validé ?"]=(ar30[:,9]>ar30[:,1])


# -----------------------------  C35 --------------------------------------------------------------------------------
fck_35=list_fck[4]
col1_35=np.array([[fck_35]]*df2.shape[0]).transpose() # Ajout colonne Fck
col2_35=vec_phi_min(np.array([df2[:,1]]),np.array([df2[:,0]]),col1_35,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])) # Ajout colonne Diametre mini
col3_35=(np.array(elancement_circ(df2[:,0], col2_35))) # Ajout colonne Elancement
col4_35=vec_alpha_circ(col3_35)
col5_35=np.array(NRd_circ(col4_35,col1_35,col2_35,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])))
ar35=np.hstack((df2,col1_35.transpose(),col2_35.transpose(),col3_35.transpose(),col4_35.transpose(),col5_35.transpose()))

df35=pot_df.copy()
df35["Fck"]=ar35[:,5]
df35["Diametre mini"]=ar35[:,6]
df35["Elancement"]=ar35[:,7]
df35["Alpha"]=ar35[:,8]
df35["NRd"]=ar35[:,9]
df35["Critere validé ?"]=(ar35[:,9]>ar35[:,1])


# -----------------------------  C40 --------------------------------------------------------------------------------
fck_40=list_fck[5]
col1_40=np.array([[fck_40]]*df2.shape[0]).transpose() # Ajout colonne Fck
col2_40=vec_phi_min(np.array([df2[:,1]]),np.array([df2[:,0]]),col1_40,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])) # Ajout colonne Diametre mini
col3_40=(np.array(elancement_circ(df2[:,0], col2_40))) # Ajout colonne Elancement
col4_40=vec_alpha_circ(col3_40)
col5_40=np.array(NRd_circ(col4_40,col1_40,col2_40,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])))
ar40=np.hstack((df2,col1_40.transpose(),col2_40.transpose(),col3_40.transpose(),col4_40.transpose(),col5_40.transpose()))

df40=pot_df.copy()
df40["Fck"]=ar40[:,5]
df40["Diametre mini"]=ar40[:,6]
df40["Elancement"]=ar40[:,7]
df40["Alpha"]=ar40[:,8]
df40["NRd"]=ar40[:,9]
df40["Critere validé ?"]=(ar40[:,9]>ar40[:,1])


# -----------------------------  C45 --------------------------------------------------------------------------------
fck_45=list_fck[6]
col1_45=np.array([[fck_45]]*df2.shape[0]).transpose() # Ajout colonne Fck
col2_45=vec_phi_min(np.array([df2[:,1]]),np.array([df2[:,0]]),col1_45,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])) # Ajout colonne Diametre mini
col3_45=(np.array(elancement_circ(df2[:,0], col2_45))) # Ajout colonne Elancement
col4_45=vec_alpha_circ(col3_45)
col5_45=np.array(NRd_circ(col4_45,col1_45,col2_45,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])))
ar45=np.hstack((df2,col1_45.transpose(),col2_45.transpose(),col3_45.transpose(),col4_45.transpose(),col5_45.transpose()))

df45=pot_df.copy()
df45["Fck"]=ar45[:,5]
df45["Diametre mini"]=ar45[:,6]
df45["Elancement"]=ar45[:,7]
df45["Alpha"]=ar45[:,8]
df45["NRd"]=ar45[:,9]
df45["Critere validé ?"]=(ar45[:,9]>ar45[:,1])


# -----------------------------  C50 --------------------------------------------------------------------------------
fck_50=list_fck[7]
col1_50=np.array([[fck_50]]*df2.shape[0]).transpose() # Ajout colonne Fck
col2_50=vec_phi_min(np.array([df2[:,1]]),np.array([df2[:,0]]),col1_50,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])) # Ajout colonne Diametre mini
col3_50=(np.array(elancement_circ(df2[:,0], col2_50))) # Ajout colonne Elancement
col4_50=vec_alpha_circ(col3_50)
col5_50=np.array(NRd_circ(col4_50,col1_50,col2_50,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])))
ar50=np.hstack((df2,col1_50.transpose(),col2_50.transpose(),col3_50.transpose(),col4_50.transpose(),col5_50.transpose()))

df50=pot_df.copy()
df50["Fck"]=ar50[:,5]
df50["Diametre mini"]=ar50[:,6]
df50["Elancement"]=ar50[:,7]
df50["Alpha"]=ar50[:,8]
df50["NRd"]=ar50[:,9]
df50["Critere validé ?"]=(ar50[:,9]>ar50[:,1])


# -----------------------------  C55 --------------------------------------------------------------------------------
fck_55=list_fck[8]
col1_55=np.array([[fck_55]]*df2.shape[0]).transpose() # Ajout colonne Fck
col2_55=vec_phi_min(np.array([df2[:,1]]),np.array([df2[:,0]]),col1_55,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])) # Ajout colonne Diametre mini
col3_55=(np.array(elancement_circ(df2[:,0], col2_55))) # Ajout colonne Elancement
col4_55=vec_alpha_circ(col3_55)
col5_55=np.array(NRd_circ(col4_55,col1_55,col2_55,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])))
ar55=np.hstack((df2,col1_55.transpose(),col2_55.transpose(),col3_55.transpose(),col4_55.transpose(),col5_55.transpose()))

df55=pot_df.copy()
df55["Fck"]=ar55[:,5]
df55["Diametre mini"]=ar55[:,6]
df55["Elancement"]=ar55[:,7]
df55["Alpha"]=ar55[:,8]
df55["NRd"]=ar55[:,9]
df55["Critere validé ?"]=(ar55[:,9]>ar55[:,1])


# -----------------------------  C60 --------------------------------------------------------------------------------
fck_60=list_fck[9]
col1_60=np.array([[fck_60]]*df2.shape[0]).transpose() # Ajout colonne Fck
col2_60=vec_phi_min(np.array([df2[:,1]]),np.array([df2[:,0]]),col1_60,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])) # Ajout colonne Diametre mini
col3_60=(np.array(elancement_circ(df2[:,0], col2_60))) # Ajout colonne Elancement
col4_60=vec_alpha_circ(col3_60)
col5_60=np.array(NRd_circ(col4_60,col1_60,col2_60,np.array([df2[:,3]]),np.array([df2[:,4]]),np.array([df2[:,2]])))
ar60=np.hstack((df2,col1_60.transpose(),col2_60.transpose(),col3_60.transpose(),col4_60.transpose(),col5_60.transpose()))

df60=pot_df.copy()
df60["Fck"]=ar60[:,5]
df60["Diametre mini"]=ar60[:,6]
df60["Elancement"]=ar60[:,7]
df60["Alpha"]=ar60[:,8]
df60["NRd"]=ar60[:,9]
df60["Critere validé ?"]=(ar60[:,9]>ar60[:,1])


# -----------------------------  FUSION DATAFRAMES ----------------------------------------------------------------------------

print("")
print("Dimensionnements")
df_final = pd.concat([df16,df20,df25,df30,df35,df40,df45,df50,df55,df60], ignore_index=True)
print(df_final)


# _____________________________ EXPORT RESULTS _________________________________________________________________________________


# Specify the name of the excel file
file_name = 'Poteaux_circulaires_resultats.xlsx'
  
# saving the excelsheet
df_final.to_excel(file_name)
print("")
print('Dataframe successfully exported into Excel File')


'''

    