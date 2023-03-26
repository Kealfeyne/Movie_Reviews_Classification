# 1. Обучение модели
В качестве предобученной модели я выбрал [bert-base-uncased](https://huggingface.co/bert-base-uncased) и дообучил ее на предоставленном [датасете](https://ai.stanford.edu/~amaas/data/sentiment/) классифицировать по 9-ти классам.  
Статус отзыва (Положительный или отрицательный) расчитывается как:  
- Отрицательный, если предсказание модели < 5;
- Положительный, если предсказание модели > 5;

Использовал библиотеки transformers, pytorch

# 2-3. Веб-сервис в открытом доступе
[Веб-сервис](https://nasty-wings-argue-217-197-0-81.loca.lt/reviews/) разработан на Django, доступно окно для ввода текста отзыва и кнопка, по нажатию которой модель предсказывает оценку.
> При запуске проекта локально\на различных хостингах веса дообученной модели с помощью функции [download_model](https://github.com/Kealfeyne/reviews_classification/blob/BertForSequenceClassification/ml_model/download_model.py) автоматически загрузятся с диска.

# 4. Оценка точности полученного результата
- [Отчет](https://github.com/Kealfeyne/reviews_classification/blob/BertForSequenceClassification/%D0%9E%D1%82%D1%87%D0%B5%D1%82%20%D0%BE%20%D0%BF%D1%80%D0%BE%D0%B4%D0%B5%D0%BB%D0%B0%D0%BD%D0%BD%D0%BE%D0%B9%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B5.pdf) доступен в формате PDF
- Трекинг модели велся с помощью сервиса wandb  
Полученные метрики для бинарной классификации (позитивный\негативный):
![metrics](https://github.com/Kealfeyne/reviews_classification/blob/BertForSequenceClassification/metrics.jpg)
