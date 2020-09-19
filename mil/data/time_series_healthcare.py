import numpy as np 
import pandas as pd
from tqdm import tqdm

class BagTimeSeriesHealthcare:
    """
    Create bags of time series data for healthcare problems. 
    Use this class when the dataset has a patient and general timestamp column.The general timestamp should be
    in one unit measure, ex. minutes, hours ...etc.
    This class creates bags of n consecutive windows of t seconds for each patient.
    """
    def __init__(self, df, temporal_column, patient_id_column, label_column):
        """
        Parameters
        ----------
        df : pandas DataFrame where each row is a temporal window with the corresponding features.
        temporal_column : str corresponding to the column name of the temporal column.
        patient_id_column : str corresponding to the column name of the patient id column.
        label_column : str corresponding to the column name containing the label.
        
        """
        assert df.__module__.split('.')[0] == 'pandas'
        
        self.df_original = df.copy()
        self.temporal_column = temporal_column
        self.patient_id_column = patient_id_column
        self.label_column = label_column
        self.patients = np.unique(self.df_original[self.patient_id_column])
        print("Found {} patients in the dataset".format(len(self.patients)))
        
    def split_time_patient(self, patient_id, time_between_wind):
        """ Split in consecutive windows of time_between_wind length. Depending 
        on the precision of the dataset time, np.isclose method should be more precise 

        Parameters
        ----------
        patient_id : identifier of patient
        time_between_wind : time between consecutive windows.

        Returns
        -------
        a vector locating the index of the non consecutive windows.

        """
        time = self.df[self.df[self.patient_id_column]==patient_id][self.temporal_column].values
        vectors = [0]
        for i in range(len(time)-1):
            if not np.isclose(time[i], time[i+1]-time_between_wind):
                vectors.append(i+1)     
        vectors.append(len(time))
        return vectors
        
    def assign_secuences(self, time_between_wind):
        """ Assign secuences for each patient. This means that all the consecutive windows from the same patient
            will have the same secuence id.

        Parameters
        ----------
        time_between_wind : time between consecutive windows.

        """
        self.df = self.df_original.copy()
        self.df.loc[self.df.index.values,'secuence'] = np.nan
        sequencia = 1
        patient = np.unique(self.df[self.patient_id_column])
        for i in tqdm(range(len(patient))):
            index_p = self.split_time_patient(patient[i], time_between_wind)
            df_p = self.df[self.df[self.patient_id_column] == patient[i]]
            index = df_p.index.values
            for i in range(len(index_p)-1):
                self.df.loc[index[index_p[i]:index_p[i+1]],'secuence'] = sequencia
                sequencia +=1
    
    def assign_minisecuences(self, n_windows): 
        """ Assign minisecuences for each patient. This means that all the n_windows consecutive windows from the same patient
            will have the same minisecuence id.

        Parameters
        ----------
        n_windows : number of consecutive windows that compose a bag.

        """
        self.df.loc[self.df.index.values,'minisecuences'] = np.nan
        persones_sequencia = pd.DataFrame(self.df.groupby([self.patient_id_column, 'secuence']))[0].values
        minisequencia = 1
        for i in tqdm(range(len(persones_sequencia))):
            persona = persones_sequencia[i][0]
            seq = persones_sequencia[i][1]
            df_person = self.df[(self.df[self.patient_id_column] == persona) & (self.df['secuence'] == seq)]
            indexs = df_person.index.values
            for i in range(len(indexs)):
                if i % n_windows == 0:
                    minisequencia +=1
                self.df.loc[indexs[i],'minisecuences'] = minisequencia
                
    def delete_non_consistent_matrix(self, n_windows):
        """ Method to delete the bags that does not have n_windows consecutive. 

        Parameters
        ----------
        n_windows : number of consecutive windows that compose a bag.

        """
        print("Deleting bags with less than: {} instances".format(n_windows))
        x = self.df.groupby(['minisecuences'])[self.patient_id_column].count()
        miniseq_to_del = x[x < n_windows].index.values.astype(int)
        rows_to_del = self.df[self.df['minisecuences'].isin(miniseq_to_del)].index.values
        self.df.drop(rows_to_del, inplace=True)
        self.patients = np.unique(self.df[self.patient_id_column])
        print("Modified dataset has {} patients".format(len(self.patients)))

    def update_bags(self, features):
        """ update the bag attribute with the new features

        Parameters
        ----------
        features : column names of the features to use in the bags
        
        """
        miniseq = np.unique(self.df[self.df[self.patient_id_column].isin(self.patients)]['minisecuences'])
        self.bags = []
        bag_correspondence = []
        for bag in miniseq:
            df_seq = self.df[self.df['minisecuences'] == bag]
            instances_bag = df_seq[features].values
            bag_correspondence.append(np.mean(df_seq[['secuence', 'minisecuences', self.temporal_column, self.patient_id_column, self.label_column]].values,axis=0))
            self.bags.append(instances_bag.tolist())
                       
        self.df_correspondence = pd.DataFrame(np.array(bag_correspondence), columns=['secuence', 'minisecuences', 'time', 'patient_id', 'label'])
    
    def get_bags(self):
        """ get the bags, labels and group

        Returns
        -------
        bags, labels and group.

        """
        X = self.bags
        y = self.df_correspondence['label'].values
        group = self.df_correspondence['patient_id'].values
        return X, y, group
    