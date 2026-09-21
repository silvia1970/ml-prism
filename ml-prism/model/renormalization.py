import pandas as pd
import json


if __name__=="__main__":

    features_selected = [
        "meanbp", "resprate", "heartrate", "spo2_pulsoxy", "tempc",
        "cardiacoutput", "o2flow", "fio2", "albumin", "bands",
        "bicarbonate", "bilirubin", "creatinine", "chloride", "glucose",
        "hemoglobin", "lactate", "platelet", "potassium", "ptt",
        "inr", "sodium", "wbc", "creatinekinase", "ck_mb",
        "fibrinogen", "ldh", "magnesium", "calcium_free", "po2_bloodgas",
        "ph_bloodgas", "pco2_bloodgas", "so2_bloodgas", "troponin_t"
    ]

    
    single_patient_df = pd.read_csv("datasets/MIMIC/carry_forward/mean/backend/random_patients/patient_00_id4989.csv")
    with open("datasets/MIMIC/carry_forward/mean/backend/temporal_signature_info_split_0 (1).json") as f:
        mimic_dict = json.load(f)

    renormalize_single_patient_dict = {
        'Index': single_patient_df['Index'].values,
        'timestep': single_patient_df['timestep'].values,
        'sepsis': single_patient_df['sepsis'].values
    }
    
    for i in range(len(mimic_dict['names'])):

        feature_name = mimic_dict['names'][i]
        feature_mean = mimic_dict['mean'][i]
        feature_std = mimic_dict['std'][i]

        
        if feature_name in features_selected:
            renormalize_single_patient_dict[feature_name] = single_patient_df[feature_name]*feature_std+feature_mean

    renormalize_single_patient_df = pd.DataFrame(renormalize_single_patient_dict)
    renormalize_single_patient_df.to_csv("datasets/MIMIC/carry_forward/mean/backend/random_patients/unormalized_patient_00_id4989.csv", index=False)

    print(renormalize_single_patient_df.head())
  