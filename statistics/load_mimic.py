import pandas as pd
import numpy as np
from datetime import datetime
import csv
import sys
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def collect_records(filepath=None, outpath=None):

    print("Collecting extracted database records in single lines per observation time")
    start = time.time() # Get current time

    # -----------------------------------------------------------------------------
    #Step 1. Open the input file:
    try:
        f_in = open(filepath, 'r')
    except IOError:
        print("Cannot read input file %s" % filepath)
        sys.exit(1)


    # -----------------------------------------------------------------------------
    #Step 2. read file line by line

    #first read header by reading first line
    line = f_in.readline()

    #print('First Line: {}'.format(line))

    parts = line.rstrip().split(",") # split the line

    firstrow2write = parts # will be written to the output file as first line

    #print('First parts: {}'.format(parts))

    variable_indices = range(4, len(parts))
    variables = parts[4: len(parts)] #containing the medical variable names
    var_counter = np.zeros(len(variable_indices)) #counting how many observation for certain timepoint available (for averaging over)
    var_sum = np.repeat(np.nan, len(variable_indices)) #summing the value of all variables for certain timepoint

    #process first line of values for initializing:
    line = f_in.readline()
    parts = line.rstrip().split(",")
    current_icustay = parts[0]
    current_time = parts[2]
    header = parts[0:4]

    tmp_values = parts[4:len(parts)] # get list of all medical values as each as string

    current_values = np.repeat(np.nan, len(variable_indices)) # initialize current values to NANs

    # Process each value of the first line:
    for i in range(len(current_values)): # convert only available numbers as integer to current_values array
        if tmp_values[i] != '':
            current_values[i] = float(tmp_values[i])
            var_counter[i] += 1 #for each non-NAN value count it for each variable seperately
            if var_sum[i] != var_sum[i]: #check if it is a nan
                var_sum[i] = current_values[i] #set it to the new value, as np.nans stay nan when adding numbers
            else:
                var_sum[i] += current_values[i] # sum all observations of each variable up (seperately) to later build timepoint-wise average

    # Open the output file
    with open(outpath, 'w') as data_file: 
        data_writer = csv.writer(data_file, delimiter=',')

        data_writer.writerow(firstrow2write) # Write header information to first line of outfile
        
        #process the remaining lines:
        for line in f_in:
            # Strip of the \n and split the line
            parts = line.rstrip().split(",")
            # Get new id
            new_icustay = parts[0]
            new_time = parts[2]
            new_header = parts[0:4]

            # Check if patient or point in time have changed! if yes, compute average of the medical variables and write to outfile
            if (new_icustay != current_icustay) or (new_time != current_time):
                # for each entry of var_sum divide by var_counter iff number available (non-NAN)
                averages = np.repeat(np.nan, len(var_sum))
    
                for i in range(len(var_sum)):
                    if var_sum[i] == var_sum[i]: # if var_sum is NOT a NaN, compute average
                        averages[i] = var_sum[i]/var_counter[i]
                # write this array of averages (and potentially NANs) to a line of output file
                row2write = np.append(header, averages)
                data_writer.writerow(row2write)
                #f_out.write("%s,%s,%s\n" % (header ...))

                # reinitialise icustay_id, time and header such that next timepoint (and or patient) can be processed.
                current_icustay = new_icustay
                current_time = new_time
                header = new_header
                # reinitialise count and sum of variables for computing new averages:
                var_counter = np.zeros(len(variable_indices)) #counting how many observation for certain timepoint available (for averaging over)
                var_sum = np.repeat(np.nan, len(variable_indices)) # summing over the variables for certain timepoint (numerator for average)

        	# Process the current (patient, time) tuple!	
            tmp_values = parts[4:len(parts)]

            new_values = np.repeat(np.nan, len(variable_indices)) # initialize current values to NANs

            # Process each value of the line:
            for i in range(len(new_values)): # convert only available numbers to integer to current_values array
                if tmp_values[i] != '':
                    new_values[i] = float(tmp_values[i])
                    var_counter[i] += 1 #for each non-NAN value count it for each variable seperately
                    if var_sum[i] != var_sum[i]: #check if it is a nan
                        var_sum[i] = new_values[i] #set it to the new value, as np.nans stay nan when adding numbers
                    else:
                        var_sum[i] += new_values[i] # sum all observations of each variable up (seperately) to later build timepoint-wise average
    # Close the files
    f_in.close()

    end = time.time()
    print('Collecting records RUNTIME {} seconds'.format(end - start)) # Print runtime of this process

    return None

def extract_window(data=None, static_data=None, onset_name=None, horizon=0):
    result = pd.DataFrame()
    ids = data['icustay_id'].unique()
    for icuid in tqdm(ids):
        pat = data.query( "icustay_id == @icuid" ) # select only rows where icustay_id matches 
        pat = pat.set_index(pd.DatetimeIndex(pat['chart_time'])) # set chart_time as index, such that window extraction works
        
        start = static_data[static_data['icustay_id']==icuid]['intime'].values[0] #determine start time (icu-intime)
        end = static_data[static_data['icustay_id']==icuid][onset_name].values[0] #determine end time (onset of sepsis or control onset)
        early_end = end - pd.Timedelta(hours=horizon) # define earlier end of extraction window depending on prediction horizon! (in hours)
        pat_window = pat[start:early_end] #select window from in-time up to onset minus horizon padding.
        # pat_window_cp = pat_window.copy()
        # pat_window_cp['chart_time'] = (pat_window['chart_time']-start)/pd.Timedelta(hours=1) # convert chart_time to relative hour
        result=pd.concat([result, pat_window])
    return result

def drop_short_series(data, case_static, control_static, min_length=7, max_length=200):
    #Determine if onset hour earlier than min_length:
    long_cases = case_static[case_static['sepsis_onset_hour']>=min_length]['icustay_id'].values
    long_controls= control_static[control_static['control_onset_hour'] >=min_length]['icustay_id'].values
    selected_patients = np.concatenate([long_cases, long_controls])
    #intialize result
    result = pd.DataFrame()
    ids = data['icustay_id'].unique()

    cases, controls = 0,0
    for icuid in ids:
        #process current icustay_id
        pat = data.query( "icustay_id == @icuid" ) #select only those rows where icustay_id matches current icuid iteration
        # Get end_time for num-/grid_times:
        end_time = pat['chart_time'].iloc[-1] # last used time of this icustay
        # Determine size on grid:
        num_rnn_grid_time = int(np.floor(end_time)+1)
        
        if icuid not in selected_patients: #if onset earlier than min_length, continue to next patient   
            #print('Skipping patient {}, TS too short'.format(icuid))
            continue
        ##if too long, skip this outlier sample and continue to next patient; (those 5 outlier patients (>200) almost quadruple memory usage..)
        elif num_rnn_grid_time > max_length:
            #print('Skipping patient {}, TS too long'.format(icuid))
            continue
        else: #if both exclusion criteria do not apply, add patient to resulting df
            result=pd.concat([result, pat])
            if icuid in long_cases:
                cases+=1 
            elif icuid in long_controls:
                controls+=1
    print('Using {} cases, {} controls.'.format(cases, controls))
    return result

def select_static_vars(df, static_vars): # returns a selection of used static variables with simplified categories.
    start_time = time.time()
    #convert age to bins (>70, >50, <50)
    #convert ethn to summary vars
    df = df[static_vars] # first select list of variables for further processing
        
    df_out = df.copy()
    df_out = df_out.drop(columns=['ethnicity','admission_age'])
    df_out['ethnicity'] = np.nan
    df_out['admission_age'] = np.nan
    df_out = df_out[static_vars]

    #Line-by-Line processing (faster and looping over entire df multiple times..)
    for row in df_out[:10].itertuples(): #start with 10 rows for debugging
        row_ind = row[0]
        old_row = df.iloc[row_ind]
        #Loop over columns, replace with simplified / discretized covariate variables
        for col_ind, item in zip(old_row.index, old_row):
            #process ethnicity column:
            if col_ind == 'ethnicity':
                if item in ['BLACK/AFRICAN AMERICAN','BLACK/CAPE VERDEAN']:
                    new_eth = 'black'
                elif item == 'WHITE':
                    new_eth = 'white'
                elif item in ['UNKNOWN/NOT SPECIFIED','UNABLE TO OBTAIN']:
                    new_eth = 'na'
                else:
                    new_eth = 'other'
                df_out.loc[row_ind, col_ind] = new_eth
            #process if age column:
            if col_ind == 'admission_age':
                if item > 70:
                    new_age = '>70'
                elif item > 50:
                    new_age = '>50'
                elif item <= 50:
                    new_age = '<=50'
                df_out.loc[row_ind, col_ind] = new_age
    
    dummy_vars = static_vars[1:]
    df_dummy = pd.get_dummies(df_out[dummy_vars]) #create one-hot dummy variables for all static variables used for the model (not icustay_id)
    df_dummy.insert(loc=0, column='icustay_id', value=df_out['icustay_id'].values)
    print('Select static variables took {}'.format(time.time() - start_time))

    return df_dummy

#preprocessing script to binned and imputed final data to apply on simple baselines..

def bin_and_impute(data, bin_width=60, variable_start_index=5):
    result = [] #list of patients dataframes

    #set of variables to process:
    variables = np.array(list(data.iloc[:,variable_start_index:]))
    #create resample parameter string:
    bin_width = str(bin_width)+'min'

    #all distinct icustay ids:
    id_s = data['icustay_id'].unique() # get unique ids

    #loop over patients:
    for icustay_id in id_s:
        print(f'Processing ID: {icustay_id} ....')
        pat = data.query( "icustay_id == @icustay_id" ) #select subset of dataframe featuring current icustay_id
        pat_i = pat.set_index('chart_time', inplace=False)     

        #resampling needs datetime or timedelta format, create dummy timestamp from relative hour:
        #pat_i.index = pd.to_datetime(pat_i.index, unit='D') #unit='s'
        pat_i.index = pd.to_timedelta(pat_i.index, unit='h')

        # if first index > 0, add empty point to start with (such that reampling occurs on same grid as TCN/RNN grid )
        start = pd.to_timedelta(0.0, unit='h')
        if pat_i.index[0] > start:
            #take the first row and set the variables to nan, and the index to 0. append first row with pat_i
            first_row = pat_i.iloc[0].copy(deep=True)
            first_row[variables] = np.nan
            first_row = pd.DataFrame(first_row, columns=pat_i.columns, index=[start])
            pat_a = first_row.append(pat_i)
        else:
            pat_a = pat_i.copy(deep=True)
        
        #Patient with no measurements in extracted window can not be resampled -> will be removed later on (when masking_samples)
        #for here, simply replace all vars with 0, as they will be dropped when masking
        n_non_nans = (~pat[variables].isnull()).sum().sum()
        if n_non_nans == 0:
            pat_a[variables] = pat[variables].replace(np.nan, 0)

        #resampling to bins of bin_width size:
        pat_rs = pat_a[variables].resample(bin_width, how='mean') #loffset=pat_i.index[0], label='left'
        
        #forward filling
        pat_ff = pat_rs.ffill()
        #fill remaining NaN with train mean (sample and if sample mean is nan -> train_stats mean!)
        for variable in variables:
            n_nans = pat_ff[variable].isnull().sum()
            if n_nans > 0: #if yes, this variable/channel has still NaNs --> address with mean imputation
                if n_nans == len(pat_ff[variable]):
                    #do zero-imputation (as data is already centered)
                    replacement = 0 # as dataset mean is 0 after centering
                else:
                    #sample mean imputation
                    replacement = pat_ff[variable].mean()

                pat_ff[variable] = pat_ff[variable].replace(np.nan, replacement)
        if pat_ff.isnull().sum().sum() > 0:
            print(f'NaN REMAINING for patient {icustay_id} !!')
        
        result.append(pat_ff)
    return result, id_s

"""def resample_and_propagate(group, measurement_cols, static_cols, freq='1h'):

    static_dict = {col: group[col].iloc[0] for col in static_cols}

    # Impostiamo correttamente l'indice temporale
    temp_series = group.set_index('chart_time')[measurement_cols]

    # # SOLO PER DEBUG
    # if group['icustay_id'].iloc[0] == 299875:
    #     print("=== temp_series ===")
    #     print(temp_series)
    #     print("=== temp_series.dropna() ===")
    #     print(temp_series.dropna())

    # Resampling robusto
    resampled = (
        temp_series
        .resample(freq)
        .first()     # Non usare nearest
        .ffill()
        .bfill()
    )

    # Aggiungiamo le statiche
    for col, value in static_dict.items():
        resampled[col] = value

    return resampled.reset_index()"""

# --- Funzione per split stratificato per label ---
def split_stratified(df, label_col='label', test_size=0.2, val_size=0.1, random_state=42):
    # Calcola label per paziente (ad esempio max della serie)
    patient_labels = df.groupby('icustay_id')[label_col].max().reset_index()
    
    # Stratified split pazienti
    train_val_ids, test_ids = train_test_split(
        patient_labels['icustay_id'], 
        test_size=test_size, 
        stratify=patient_labels[label_col], 
        random_state=random_state
    )
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size/(1-test_size),
        stratify=patient_labels[patient_labels['icustay_id'].isin(train_val_ids)][label_col],
        random_state=random_state
    )
    
    train_df = df[df['icustay_id'].isin(train_ids)].copy()
    val_df   = df[df['icustay_id'].isin(val_ids)].copy()
    test_df  = df[df['icustay_id'].isin(test_ids)].copy()
    
    return train_df, val_df, test_df

# --- Funzione per normalizzazione per fold ---
def normalize_df(train_df, val_df, test_df, feature_cols, save_path='scaler_stats.json'):
    scaler = StandardScaler() #z-score
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Salva mean e std in JSON
    stats = {col: {'mean': float(scaler.mean_[i]), 'std': float(np.sqrt(scaler.var_[i]))} 
             for i, col in enumerate(feature_cols)}
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    return train_df, val_df, test_df

# --- Funzione aggiornata di resampling per paziente ---
def resample_propagate_mask(df, measurement_cols, static_cols, freq='1h', mask_value=-100):
    """
    Resample, imputa valori mancanti e aggiunge feature statiche.
    Adatto per LSTM con masking.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe originale con colonne temporali e statiche.
    measurement_cols : list of str
        Colonne variabili (da resample/imputare).
    static_cols : list of str
        Colonne statiche (ad esempio età, sesso).
    freq : str
        Frequenza di resampling (default '1h').
    mask_value : float
        Valore da usare per indicare timestep da ignorare.
    
    Returns
    -------
    pd.DataFrame
        Dataframe resampled e pronto per LSTM.
    """
    
    all_patients = []
    
    for icustay_id, group in tqdm(df.groupby('icustay_id')):
        # Salva feature statiche
        static_dict = {col: group[col].iloc[0] for col in static_cols}
        
        # Imposta indice temporale
        temp_series = group[measurement_cols].set_index(group['chart_time']).sort_index()
        
        # Se il primo timestamp > freq.min, aggiungi riga iniziale NaN
        start = temp_series.index.min()
        if start > temp_series.index.min():  # solitamente start = primo timestamp disponibile
            first_row = pd.DataFrame({col: [np.nan] for col in measurement_cols}, index=[start])
            temp_series = pd.concat([first_row, temp_series])
        
        # Resampling + Ffill + Bfill
        resampled_data = temp_series.resample(freq).first().ffill().bfill()
        
        # Imputazione dummy per eventuali NaN rimanenti
        for col in measurement_cols:
            if resampled_data[col].isnull().all():
                # tutta la colonna è NaN → metti valore di masking
                resampled_data[col] = mask_value
            else:
                # alcune righe NaN → metti valore di masking
                resampled_data[col] = resampled_data[col].fillna(mask_value)
        
        # Aggiungi feature statiche
        for col, val in static_dict.items():
            resampled_data[col] = val
        
        # Mantieni icustay_id
        resampled_data['icustay_id'] = icustay_id
        
        # Reset index
        resampled_data = resampled_data.reset_index().rename(columns={'index': 'chart_time'})
        
        all_patients.append(resampled_data)
    
    # Concatenazione finale
    return pd.concat(all_patients, ignore_index=True)

def load_data(test_size=0.1, horizon=0, na_thres=500, variable_start_index=5, data_sources=['labs','vitals','covs'], min_length=None, max_length=None, overwrite=False, split=0, binned=False):

    rs = np.random.RandomState(split)
    
    #---------------------------------
    # 0. SET PATHS (hard-coded relative paths for now)
    # outpath to case/control-joined and window extracted file
    labvital_outpath='mimic_output/full_labvitals_horizon_{}.csv'.format(horizon)
    #Case vitals and labs (input, mimic_output)
    case_vitals_in='mimic_output/case_55h_hourly_vitals_ex1c.csv'
    case_vitals_out='mimic_output/case_55h_hourly_vitals_ex1c_collected.csv'
    case_labs_in='mimic_output/case_55h_hourly_labs_ex1c.csv'
    case_labs_out='mimic_output/case_55h_hourly_labs_ex1c_collected.csv'
    #Control vitals and labs (input, mimic_output)
    control_vitals_in = 'mimic_output/control_55h_hourly_vitals_ex1c.csv'
    control_vitals_out = 'mimic_output/control_55h_hourly_vitals_ex1c_collected.csv'
    control_labs_in = 'mimic_output/control_55h_hourly_labs_ex1c.csv'
    control_labs_out = 'mimic_output/control_55h_hourly_labs_ex1c_collected.csv'

    #Time Series mimic_output (splitted)
    compact_split_out = 'mimic_output/labvitals_tr_te_val_compact_min_length_{}_max_length_{}_horizon_{}_split_{}.pkl'.format(min_length,max_length, horizon,split)
    binned_split_out = 'mimic_output/labvitals_tr_te_val_binned_min_length_{}_max_length_{}_horizon_{}_split_{}.pkl'.format(min_length,max_length, horizon,split)

    #Static data:
    case_static_in='mimic_output/static_variables_cases.csv'
    control_static_in='mimic_output/static_variables_controls.csv'

    #Load static info (with onset hour / times)
    case_static = pd.read_csv(case_static_in)
    for t in ['intime','sepsis_onset']: #convert string (of times) to datetime objects
        case_static[t] = case_static[t].apply( lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") )

    control_static = pd.read_csv(control_static_in)
    for t in ['intime','control_onset_time']: 
        control_static[t] = control_static[t].apply( lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") )

    # for control_id in control_static['matched_case_icustay_id'].unique():
    #     if  control_id in case_static['icustay_id'].unique():
    #         print(f"Match: Control_id {control_id} is in case_static")

    if overwrite or not os.path.isfile(labvital_outpath):
        # IF NOT ALREADY RUN, UNCOMMENT THIS SECTION:
        if True: #if multi-row records data was not collected in single row per observation time before, do it now
            
            #---------------------------------
            # 1. a): Collect Data line-by-line
            print('First run of this setting. Collecting Data...')
            # First collect the data of the sql-queried csv files (case and control vitals) in single rows for each patient & point in time:

            print('Collecting Case vitals...')
            collect_records(case_vitals_in, case_vitals_out)
            #collect_records(args.infile_cases, args.outfile_collected_cases)

            print('Collecting Case labs...')
            collect_records(case_labs_in,case_labs_out)

            print('Collecting Control vitals...')

            collect_records(control_vitals_in,control_vitals_out)

            print('Collecting Control labs...')

            collect_records(control_labs_in,control_labs_out)


        print('Loading collected case records...')
        #Read cases:
        case_vitals = pd.read_csv(case_vitals_out) #read file to pd.dataframe
        print(case_vitals.info())
        case_vitals['chart_time'] = case_vitals['chart_time'].apply( lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") ) # convert time string to datetime object
        case_labs = pd.read_csv(case_labs_out)
        case_labs['chart_time'] = case_labs['chart_time'].apply( lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") )
        print('Loading collected control records...')
        #Read controls:
        control_vitals = pd.read_csv(control_vitals_out) #read file to pd.dataframe
        control_vitals['chart_time'] = control_vitals['chart_time'].apply( lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") ) # convert time string to datetime object
        control_labs = pd.read_csv(control_labs_out)
        control_labs = control_labs.dropna(subset=['chart_time']) # had to include this dropna row as few severly incomplete records in labevents table
        control_labs = control_labs.reset_index(drop=True) # re-enumerate the row index after removing few noisy rows missing chart_time (173)
        control_labs['chart_time'] = control_labs['chart_time'].apply( lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") )
        #---------------------------------
        # 2. a): Merge lab with vital time series (from different sql tables originally) and append case and controls to one dataframe!
        print('Merge lab and vital time series data ..')
        #CASE: Merge vital and lab values into one time series:
        case_labvitals = pd.merge(case_vitals, case_labs, how='outer', left_on=['icustay_id', 'chart_time', 'subject_id', 'sepsis_target'], 
            right_on=['icustay_id', 'chart_time', 'subject_id', 'sepsis_target'], sort=True)
        #CONTROL: Merge vital and lab values into one time series:
        control_labvitals = pd.merge(control_vitals, control_labs, how='outer', left_on=['icustay_id', 'chart_time', 'subject_id', 'pseudo_target'], 
            right_on=['icustay_id', 'chart_time', 'subject_id', 'pseudo_target'], sort=True)
        
        control_groups_labvitals_static, case_groups_labvitals_static = [], []
        
        for _, group in control_labvitals.groupby('icustay_id'):
            icustay_id = group['icustay_id'].iloc[0]

            # Estrai il valore singolo
            admission_age = control_static.loc[
                control_static['icustay_id'] == icustay_id, 'admission_age'
            ].iloc[0]

            gender = control_static.loc[
                control_static['icustay_id'] == icustay_id, 'gender'
            ].iloc[0]

            # Ora puoi replicare facilmente quel valore per tutto il gruppo
            group['admission_age'] = admission_age
            group['gender'] = gender

            control_groups_labvitals_static.append(group)
        
        control_labvitals_static = pd.concat(control_groups_labvitals_static, ignore_index=True)

        for _, group in case_labvitals.groupby('icustay_id'):
            icustay_id = group['icustay_id'].iloc[0]

            # Estrai il valore singolo
            admission_age = case_static.loc[
                case_static['icustay_id'] == icustay_id, 'admission_age'
            ].iloc[0]

            gender = case_static.loc[
                case_static['icustay_id'] == icustay_id, 'gender'
            ].iloc[0]

            # Ora puoi replicare facilmente quel valore per tutto il gruppo
            group['admission_age'] = admission_age
            group['gender'] = gender

            case_groups_labvitals_static.append(group)
        
        case_labvitals_static = pd.concat(case_groups_labvitals_static, ignore_index=True)
    
        #---------------------------------
        # 2. b): Extract case and control window time series
        print('Extract time series window before onset for prediction') 
        case_labvitals = extract_window(data=case_labvitals_static, static_data=case_static, onset_name='sepsis_onset', horizon=horizon)
        control_labvitals = extract_window(data=control_labvitals_static, static_data=control_static, onset_name='control_onset_time', horizon=horizon)
        # in the extract_window() step we drop 633 control stays from 17909 control stays to 17276, as for some controls there is no data in this window (luckily no losses on cases!)
        #---------------------------------
        # 2. c): Join Cases and Controls
        print('Merge case and control data')
        #rename pseudo_target, such that case and controls can be appended to same df..
        control_labvitals = control_labvitals.rename(columns={'pseudo_target': 'sepsis_target'})
        #for joining label cases/controls with label: 1/0
        control_labvitals.insert(loc=0, column='label', value=np.repeat(0,len(control_labvitals)))
        case_labvitals.insert(loc=0, column='label', value=np.repeat(1,len(case_labvitals)))
        #append cases and controls, for spliting/standardizing:
        full_labvitals = pd.concat([case_labvitals, control_labvitals])
        #full_labvitals=full_labvitals.reset_index(drop=True) #drop chart_time index, so that on-the-fly df is identical with loaded one
        full_labvitals.to_csv(labvital_outpath, index=False)

    else:
        print('full_labvitals_horizon_{}.csv exists, cases/control were merged before and window extracted. Load this file..\n'.format(horizon))

    full_labvitals = pd.read_csv(labvital_outpath)
    full_labvitals = full_labvitals.dropna(axis=1, thresh=na_thres) 

    cases = full_labvitals[full_labvitals['label']==1]
    control = full_labvitals[full_labvitals['label']==0]

    if min_length:

        print(f"(Before) N. of cases {cases.shape[0]}")

        window_lengths=[]
        min_length = 2
        cases_minlen = pd.DataFrame()

        for _ , group in cases.groupby(by='icustay_id'):

            if group.shape[0]>=min_length:
                cases_minlen=pd.concat([cases_minlen, group])
                window_lengths.append(group.shape[0])

        window_lengths_np = np.array(window_lengths)

        print(f"(After) N. of cases {cases_minlen.shape[0]}")
        print(f"    Avg. window length for cases: {window_lengths_np.mean().round(3)} +/- {window_lengths_np.std().round(3)}")
        print(f"    Min window length: {window_lengths_np.min()} - Max window length: {window_lengths_np.max()}")

    full_labvitals['adm_age'] = pd.cut(
        full_labvitals['admission_age'],
        bins=[0, 50, 70, float("inf")],
        labels=["(0,50]", "(50,70]", "(70,inf)"]
    )

    full_labvitals = pd.get_dummies(
        full_labvitals,
        columns=['adm_age'],
        dtype=int,       
        drop_first=False
    )

    full_labvitals = full_labvitals.drop(columns=['admission_age'])
    full_labvitals['gender'] = full_labvitals['gender'].map({'M': 0, 'F': 1}).astype(int)
    full_labvitals=full_labvitals.drop(columns=['subject_id', 'sepsis_target']) # sepsis_target?
    full_labvitals['chart_time']=pd.to_datetime(full_labvitals['chart_time'])

    full_labvitals = full_labvitals.sort_values(['icustay_id', 'chart_time']).reset_index(drop=True)

    print(full_labvitals)
    print(full_labvitals.info())

    full_labvitals.to_csv("my_mimic.csv", index=False)

    # Colonne numeric-only da normalizzare
    static_cols = ['label', 'gender', 'adm_age_(0,50]', 'adm_age_(50,70]', 'adm_age_(70,inf)']

    measurement_cols = full_labvitals.columns.drop(static_cols+['icustay_id','chart_time'])

    # --- 1. Split train/val/test ---
    train_df, val_df, test_df = split_stratified(full_labvitals, label_col='label')

    sepsis_pats=0
    non_sepsis_pat=0
    for _ ,pat in train_df.groupby(by='icustay_id'):
        if pat['label'].iloc[0]==1:
            sepsis_pats+=1
        else:
            non_sepsis_pat+=1

    print("Train:")
    print(f"\nN. of patients with sepsis: {sepsis_pats}")
    print(f"N. of patients with no sepsis: {non_sepsis_pat}")

    train_df, val_df, test_df = normalize_df(train_df, val_df, test_df, measurement_cols, save_path='scaler_stats.json')

    print("After normalization:")
    print(train_df)

    train_resampled = resample_propagate_mask(train_df, measurement_cols, static_cols, freq='1h')
    val_resampled   = resample_propagate_mask(val_df, measurement_cols, static_cols, freq='1h')
    test_resampled  = resample_propagate_mask(test_df, measurement_cols, static_cols, freq='1h')

    print("After resampling")
    print(val_resampled)
    print(val_resampled.info())
    print(test_resampled)
    print(test_resampled.info())

    print(measurement_cols)

    icustay_id_col = full_labvitals['icustay_id'].copy()

    # Applica la funzione a ciascun gruppo
    resampled_labvitals = full_labvitals.groupby(icustay_id_col).apply(lambda x: resample_and_propagate(x, measurement_cols=measurement_cols, 
                                                                                                        static_cols=static_cols))

    # Verifica la dimensione e l'assenza di NaN (dovrebbe essercene pochissimi, solo se un'intera colonna era NaN per un soggetto)
    full_labvitals_static = resampled_labvitals.reset_index().drop(columns='level_1')
    print(full_labvitals_static)
    print(full_labvitals_static.info())
    # print(full_labvitals_static.describe())
    
    one_sample_count=0
    sepsis_pats=0
    non_sepsis_pat=0
    for _ ,pat in full_labvitals_static.groupby(by='icustay_id'):
        if pat['label'].iloc[0]==1:
            sepsis_pats+=1
        else:
            non_sepsis_pat+=1

    print(f"N. of patients with sepsis: {sepsis_pats}")
    print(f"N. of patients with no sepsis: {non_sepsis_pat}")

    # print(full_labvitals_static.isna().sum())

    # full_labvitals_static = full_labvitals_static.dropna(axis=1, thresh=na_thres) 

    remained_sepsis_pats = 0
    remained_non_sepsis_pats = 0

    prediction_horizon=6
    input_window_size=2

    full_labvitals_static_clean= pd.DataFrame()
    cols_to_consider = full_labvitals_static.drop(columns=['icustay_id', 'chart_time', 'label']).columns.to_list()

    for _ ,pat in full_labvitals_static.groupby(by='icustay_id'):
        if pat.shape[0]>=input_window_size+prediction_horizon:
            pat_subset = pat[cols_to_consider]
            n_nan_values = pat_subset.isna().sum().sum()
            pat_size = pat_subset.shape[0]*pat_subset.shape[1]
            # print(f"Nan_values: {n_nan_values} - Pat_size: {pat_size}")
            if (n_nan_values/pat_size)>=0.5:
                full_labvitals_static_clean = pd.concat([full_labvitals_static_clean, pat])
                if pat['label'].iloc[0]==1:
                    remained_sepsis_pats+=1
                else:
                    remained_non_sepsis_pats+=1

    print(f"(Remained) N. of patients with sepsis: {remained_sepsis_pats}")
    print(f"(Remained) N. of patients with no sepsis: {remained_non_sepsis_pats}")
    full_labvitals_static_clean = full_labvitals_static_clean.reset_index(drop=True)
    cols_to_convert = ['label', 'gender', 'adm_age_(0,50]', 'adm_age_(50,70]', 'adm_age_(70,inf)']
    full_labvitals_static_clean[cols_to_convert] = full_labvitals_static_clean[cols_to_convert].astype(bool)

    cols_to_move = ["icustay_id", "chart_time", "gender", "adm_age_(0,50]", "adm_age_(50,70]", "adm_age_(70,inf)"]
    other_cols = [c for c in full_labvitals_static_clean.columns if c not in cols_to_move]

    full_labvitals_static_clean = full_labvitals_static_clean[cols_to_move + other_cols]
    
    print(full_labvitals_static_clean)
    print(full_labvitals_static_clean.info())
    # print(full_labvitals_static_clean.describe())



if __name__=="__main__":
    load_data(overwrite=False, min_length=False)