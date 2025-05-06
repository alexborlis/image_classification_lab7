# Lab 7: Основи класифікації зображень з використанням KNN

Цей проєкт демонструє основи класифікації зображень за допомогою алгоритму K-найближчих сусідів (KNN) із використанням бібліотеки `scikit-learn` на основі набору рукописних цифр (`digits` dataset).

---

## 🎯 Мета

- Ознайомитися з методами класифікації зображень.
- Освоїти обробку зображень та підготовку їх до навчання.
- Реалізувати класифікатор KNN та оцінити його точність.

---

## 📁 Структура проєкту

```
image_classification_lab7/
├── main.py                    # Запуск повного пайплайну
├── classifier.py             # Тренування та оцінка класифікатора
├── dataset_loader.py         # Завантаження та попередня обробка даних
├── utils.py                  # Додаткові функції (необов’язково)
├── output/
│   ├── accuracy_report.txt   # Збереження точності
│   └── sample_predictions.png # Візуалізація передбачень
├── requirements.txt          # Список залежностей
└── README.md                 # Поточний опис
```

---

## 🛠️ Встановлення

```bash
git clone https://github.com/your_username/image-classification-lab7.git
cd image-classification-lab7
pip install -r requirements.txt
```

---

## ▶️ Запуск

```bash
python main.py
```

---

## 🧠 Алгоритм

1. Завантаження набору зображень (`digits`)
2. Перетворення 8×8 зображень у вектори довжини 64
3. Розділення даних на тренувальні та тестові
4. Тренування KNN-моделі (k=5)
5. Оцінка точності моделі
6. Візуалізація прикладів передбачень

---

## 📊 Результати

- Очікувана точність ~0.98 на тестовому наборі
- Вивід збережено у `output/accuracy_report.txt`
- Зображення з передбаченнями: `output/sample_predictions.png`

---

## 📚 Джерела

- https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
- https://medium.com/@nikhilanandikam/handwritten-digit-recognition-hdr-using-k-nearest-neighbors-knn-f4c794a0282a
