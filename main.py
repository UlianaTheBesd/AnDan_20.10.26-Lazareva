# Лазарева Ульяна, 24КНТ-5.
# Задание за 20.10.26.

# No = https://drive.google.com/drive/folders/1dRKdXCjIeUl1n2ehNqT56jh_H-7g_GJ_
# Yes = https://drive.google.com/drive/folders/1_3eo5ITOZbIGkTs-DUjXl2oyw4nHcEM5
# task2 = https://drive.google.com/drive/folders/1ENhsQZdq2njlwml5Lpsi8qKycFSVuu9h?usp=sharing

'''
Задание 1, логика: нам дали множество фотографий. Мы не имеем возможности обучать сложную нейросеть
для распознавания опухолей или жестов рук, но у нас есть sklearn - более простая ИИ-модель, которую
мы обучаем.
Идея - взять готовую, уже обученную нейросеть, распознающую фотографии (MobileNetV2), использовать её
для преобразования картинки в набор чисел. Данный набор чисел уже интерпретируется sklearn-моделями.
'''

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from PIL import Image #для открытия фотографий.
import tensorflow
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

warnings.filterwarnings("ignore")

#include_top=False делает так, чтобы возвращался лишь вектор чисел, необходимый для sklearn, а не ответ от MobileNetV2.
extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))

def ways_to_images(folder_path):
    images = []
    DATA_FORMATS = (".jpg", ".jpeg", ".png", ".bmp")
    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith(DATA_FORMATS):
            path = os.path.join(folder_path, file)
            images.append(path)
    return images

'''
Функция ниже обрабатывает 32 картинки (RGB), записывает эти 32 картинки в один массив,
отдавая MobileNetV2, потому что нейросеть умеет обрабатывать картинки блоками по несколько штук.
'''
def take_features(paths, batch_size=32):
    features = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        #.convert("RGB") - привести к трём каналам R,G,B - для случая, если PNG прозрачен или чёрно-белый -
        # - в таком случае, без приведения к RGB, будут ошибки.
        # .resize((224, 224)) - привод фотографии к нужному размеру.
        images = [Image.open(p).convert("RGB").resize((224, 224)) for p in batch_paths]
        batch = preprocess_input(np.stack([np.asarray(i, dtype=np.float32) for i in images]))
        features.append(extractor.predict(batch, verbose=0))
    return np.vstack(features)

def build_feature_data(paths, labels):
    X = take_features(paths)
    # делание таблицы признаков, отписанных MobileNetV2 - формирование DataFrame.
    data = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    data["target"] = labels
    return data

def models():
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "SVC": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }

def metrics(y_true, y_pred, average):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

def run_sklearn_models(train_data, test_data, average):
    features = [c for c in train_data.columns if c != "target"] #все столбцы, кроме "target".
    rows = []
    for name, model in models().items():
        t = time.time()
        # Далее: обучение и предсказание конкретной модели.
        model.fit(train_data[features], train_data["target"])
        pred = model.predict(test_data[features])
        # словарь "модель + метрики + время, которое отняло обучение и предсказание модели".
        # Зачем нужен замер времени? Чтобы оценить модель не только по качеству, но и по её скорости.
        rows.append({"model (name)": name, **metrics(test_data["target"], pred, average), "time (sec.)": round(time.time() - t, 1)})
    return pd.DataFrame(rows)

# ЗАДАНИЕ 1.  ----------------------------------------------
yes_paths = ways_to_images("yes")
no_paths = ways_to_images("no")
paths  = yes_paths + no_paths
'''
1.
Каждой картинке в папке yes присваивается label (как "result") = 1.
Картинкам в no - 0.
2.
data1 = build_feature_data(paths, labels) - функция берёт список путей к файлам,
открывает каждую картинку и превращает её в набор чисел, после чего добавляет туда
столбец ответов = labels (как обучение с учителем).
3.
То есть, эти шаги нужны для формивания нашей собственной "data.csv".
'''
labels = [1] * len(yes_paths) + [0] * len(no_paths)
data1 = build_feature_data(paths, labels)

# stratify=data_bin["target"] - гарантия, что соотношение классов да и нет в train и test выборках будет одинаковым.
train1, test1 = train_test_split(data1, test_size=0.25, stratify=data1["target"], random_state=42)

# обучает модели и возвращает таблицу с их метриками, которая отсортирована по f1-мере.
res1 = run_sklearn_models(train1, test1, average="binary")
res1.sort_values("f1", ascending=False)

# LAMA 1 задания.
lama_1 = TabularAutoML(task=Task("binary"), timeout=600, cpu_limit=4, reader_params={"random_state": 42})
lama_1.fit_predict(train1, roles={"target": "target"}, verbose=1)

# возвращает вероятности принадлежности к классу да = 1.
pred_perem = lama_1.predict(test1).data[:, 0]
# превращает вероятности pred_perem в метки классов: если вероятность > 50%, то это 1 ("да"), иначе 0 ("нет").
pred_lama = (pred_perem > 0.5).astype(int)

LAMA1 = {"model": "LAMA", **metrics(test1["target"], pred_lama, "binary")}
print(LAMA1) # метрики от LAMA (задание 1).
print(confusion_matrix(test1["target"], pred_lama)) # confusion_matrix (задание 1).
final_bin = pd.concat([res1, pd.DataFrame([LAMA1])], ignore_index=True).sort_values("f1", ascending=False)
print(final_bin) # сводка sklearn (задание 1).
print("f1 (>= 0.85):", (final_bin["f1"] >= 0.85).any())

# ЗАДАНИЕ 2. ----------------------------------------------
podpapki = sorted([d for d in os.listdir("asl_dataset") if os.path.isdir(os.path.join("asl_dataset", d))])

paths2 = []
class_indexes = []
for idx, cname in enumerate(podpapki): # podpapki = подпапки; каждая папка в asl_dataset -  это отдельный жест.
    p = ways_to_images(os.path.join("asl_dataset", cname)) 
    paths2 += p
    class_indexes += [idx] * len(p)

data2 = build_feature_data(paths2, class_indexes)
train2, test2 = train_test_split(data2, test_size=0.25, stratify=data2["target"], random_state=42)

res2 = run_sklearn_models(train2, test2, average="macro")
res2.sort_values("f1", ascending=False)

# LAMA.
LAMA2 = TabularAutoML(task=Task("multiclass"), timeout=900, cpu_limit=4, reader_params={"random_state": 42})
LAMA2.fit_predict(train2, roles={"target": "target"}, verbose=1)
pred_param2 = LAMA2.predict(test2).data
pred_lama_mc = pred_param2.argmax(axis=1)
lama_mc = {"model": "LAMA", **metrics(test2["target"], pred_lama_mc, "macro")} # macro - 1) метрика считается отдельно для каждого класса независимо +
#                                                                              # + 2) полученные значения складываются и делятся на количество классов (среднее).
print(lama_mc)
print(classification_report(test2["target"], pred_lama_mc, target_names=podpapki))
final_mc = pd.concat([res2, pd.DataFrame([lama_mc])], ignore_index=True).sort_values("f1", ascending=False)
print(final_mc)
