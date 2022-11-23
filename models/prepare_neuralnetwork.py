import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn import metrics
from sklearn.decomposition import PCA
import models.models as mdls

LIMIT_FOR_LABEL_ENCODING = 2
N_EPOCHS=100
BATCH_SIZE=247

def fillNA(dataset):
  for column in tqdm(dataset.columns[dataset.dtypes!='object']):
    if(dataset[column].isnull().values.any()):
      #dataset[column].fillna(dataset[column].quantile(0.75), inplace=True)
      dataset[column].fillna(dataset[column].median(), inplace=True)
  return dataset

def encode_ftrs(cat, dataset):
  le = LabelEncoder()
  le_count = 0
  all_ftrs = dataset.select_dtypes(include=['object']).shape[1]
  for col in cat:
      if (dataset[col].dtype == 'object'):
          if (len(list(dataset[col].unique())) <= LIMIT_FOR_LABEL_ENCODING):
              le.fit(dataset[col])
              dataset[col] = le.transform(dataset[col])
              le_count += 1
  print(f'%d columns were label encoded.' % le_count)
  dataset = pd.get_dummies(dataset)
  mns = all_ftrs-le_count
  print(f'%d columns were one-hot encoded.' % mns)
  return dataset

def create_tensor_ds(X, y):
  '''
  метод для создания тензоров, которые будут загружены в сеть
  '''
  X_tnsr = torch.tensor(X.values, dtype=torch.float32)
  y_tnsr = torch.tensor(y.values, dtype=torch.float32)
  ds = TensorDataset(X_tnsr, y_tnsr)
  
  return ds

def get_predictions(loader, model, device):
  '''
  метод для получения меток, которые предсказывает нейронная модель
  '''
  model.eval()
  saved_preds = []
  true_labels = []
  
  with torch.no_grad():
      for x,y in loader:
         # x = x.to(device)
         # y = y.to(device)
          scores = model(x)
          saved_preds += scores.tolist()
          true_labels += y.tolist()
  #model.train()
  return saved_preds, true_labels

def train_model(n_hidden_neurons, NN, device):
  '''
  method to train neural network. You will not need it,
  because models are stored in /data folder
  '''
  model = NN(input_size=X.shape[1], n_hidden_neurons=n_hidden_neurons).to(device)
  loss_fn = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.00025, weight_decay=1e-4)

  for epoch in range(N_EPOCHS):
    proba, true = get_predictions(test_loader, model, device=device)
    print(f"MSE: {metrics.mean_squared_error(true, proba)}")
    for batch_idx, (data, targets) in enumerate(train_loader):
      data = data.to(device)
      targets = targets.to(device)

      scores = model(data)
      loss = loss_fn(scores, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  return model

def prepare_dataset_for_neural_networks(data, label_to_del):
  '''
  По сути, ты пользуешься только этим методом
  На вход подаешь датасет, для которого нужно сделать предсказание
  На выходе получаешь лоадер, которые затем грузишь
  '''
  data.drop(['Time', 'Depth Error', 'Depth Seismic Stations', 'Magnitude Error', 'Magnitude Seismic Stations',
          'Azimuthal Gap', 'Horizontal Distance', 'Horizontal Error', 'ID'], axis=1, inplace=True)
  data_no_na = fillNA(data)
  cols = data_no_na.columns
  num_cols = data_no_na._get_numeric_data().columns
  categorical_ftrs = list(set(cols) - set(num_cols))

  data_encoded_ftrs = encode_ftrs(categorical_ftrs, data_no_na)

  del data
  del data_no_na

  X = data_encoded_ftrs.drop(["Magnitude", "Depth"], axis=1)
  y = data_encoded_ftrs[label_to_del]
  scaler = StandardScaler()

  decomposer = PCA(n_components=200)
  X = decomposer.fit_transform(X)

  X = scaler.fit_transform(X)

  mask = np.isnan(X)
  idx = np.where(~mask,np.arange(mask.shape[1]),0)
  np.maximum.accumulate(idx,axis=1, out=idx)
  X = X[np.arange(idx.shape[0])[:,None], idx]

  #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)

  #train_ds, test_ds = create_tensor_ds(pd.DataFrame(X_train), y_train, pd.DataFrame(X_test), y_test)

  #train_loader = DataLoader(dataset = train_ds, batch_size=BATCH_SIZE, shuffle=True)
  #test_loader  = DataLoader(dataset = test_ds,  batch_size=BATCH_SIZE, shuffle=True)
  res_dataset = create_tensor_ds(pd.DataFrame(X), y)
  loader = DataLoader(dataset=res_dataset, batch_size=BATCH_SIZE, shuffle=True)

  return loader


