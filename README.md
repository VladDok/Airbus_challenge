# Airbus Challenge

Технічним завданням є створення конвертаційної нейронної моделі, яка здатна сегментувати зображення. Прикладом зображення є знімки Airbus, які були дропнуті на дрібніші фото для подальшої роботи. \
![alt text](https://github.com/[VladDok]/[Airbus_challenge]/blob/[branch]/ship.png?raw=true)

Метою моделі є сегментація кораблів на знімках.

# Overview

## Connection
У програмі, на початку, налаштуй шлях, де зберігатиметься набір даних, які вказані нижче. Для цього необхідно створити папку, де буде розміщені зображення тренувального набору (у своїй папці), файл з масками, файл з параметрами навченої моделі, а також файл з результатами даного навчання для відслідковування прогресу під час проведеної підгонки моделі. \
Далі слідкуй за інструкціями.

## Data
Для роботи встанови 4 файли - файл зі знімками для навчання, файл із масковою послідовністю, файл з параметрами навченої моделі та файл з історією навчання. \
Далі, під час розвідувального аналізу, з наявних двох файлів підготується спеціальний датафрейм, який буде розіділений на три частини: тренувальний, валідаційний та тестовий. Для використання їх надалі під час перевірки моделі файли будуть збережені у форматі .pkl і завантажені у файлі інгерітанс.

## Preprocessing
Під час розвідувального аналізу видалятимуться пошкоджені зображення та зображення на яких площа кораблів менша, ніж 50 пікселів.
Для перетворення маскової послідовності у матричну форму використовується спеціальна функція декодування.

Зображення та маски масштабовані в межах [0, 1] та зменшені до розміру до 256x256 пікселів. 

## Training
Для тренування набір розбивався на 3 частини - тренувальний (безпосередньо тренувальний та валідаційний) і тестовий. \
У кінці параметри моделі та її результат були збережені і доступі у репозиторії.

Функція втрат - Adam. /
Метрика - Dice_score (F1_score).
