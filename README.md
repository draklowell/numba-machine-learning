# Проєкт із застосуванням Генетичного Алгоритму

## Мета проєкту

Метою цього проєкту є створення фреймворку для реалізації генетичних алгоритмів, які можуть бути використані для оптимізації параметрів нейронних мереж. Проєкт дозволяє експериментувати з різними методами мутації, кросоверу та селекції, а також підтримує роботу як на CPU, так і на GPU.

## Структура проєкту

Проєкт складається з наступних модулів:

### `nml` - Фреймворк нейронних мереж
Містить базові компоненти для побудови та тренування нейронних мереж:
- Шари:
  - `Linear` - повнозв'язний шар з лінійною трансформацією
  - `Convolutional` - згортковий шар для обробки зображень
  - `CellularAutomata` - шар, що реалізує клітинний автомат
  - `Flatten` - перетворює багатовимірний тензор у одновимірний
  - `Cast` - шар для зміни типу даних тензора
- Активаційні функції:
  - `ReLU` - випрямлена лінійна функція активації
  - `Softmax` - нормалізована експоненційна функція для класифікації
- Клас `Sequential` - контейнер послідовних шарів нейронної мережі
- Утиліти для роботи з пристроями: клас `Device` для абстракції CPU/GPU
- Утиліти для роботи з багатовимірними масивами (тензорами): класи `Tensor`, `Scalar`, `CPUTensor`, `GPUTensor`

### `genetic` - Генетичний алгоритм
Реалізує компоненти для еволюційної оптимізації:
- Мутації:
  - `GaussianMutation` - мутація на основі нормального розподілу
  - `ScaledMutation` - мутація зі змінним масштабом
- Кросовери:
  - `SinglePoint` - схрещування в одній точці
  - `TwoPoint` - схрещування у двох точках
  - `Uniform` - рівномірне схрещування
- Селекції:
  - `RouletteSelection` - селекція методом рулетки
  - `TournamentSelection` - турнірна селекція
  - `BestSelection` - вибір найкращих особин
  - `RankSelection` - ранжована селекція
- Пайплайни:
  - `ChromosomePipeline` - обробка окремих хромосом
  - `GenomePipeline` - комплексна обробка геномів

### `loader` - Завантаження та обробка даних
Відповідає за роботу з датасетами:
- `Downloader` - завантаження датасетів з мережі
- `DataManager` - абстрактний клас для керування даними
- `SklearnBalancedDataLoader` - завантаження збалансованих даних з використанням sklearn
- Утиліти для квантизації та нормалізації даних для ефективного навчання

### `project` - Управління експериментами
Координує взаємодію між компонентами:
- `FitnessEvaluator` - обчислення функції пристосованості
- `FitnessMetric` - метрики для оцінки якості моделей
- `GenerationHandler` - базовий клас для обробки поколінь
- `Manager` - головний клас для керування процесом еволюції, відповідає за:
  - Ініціалізацію популяції
  - Оцінку функції пристосованості
  - Застосування генетичних операторів
  - Формування нового покоління

### `handlers` - Обробники подій
Містить компоненти для моніторингу та збереження прогресу:
- `PrintHandler` - виведення інформації про прогрес еволюції
- `SaveHandler` - збереження стану популяції у файли
- `TableHandler` - запис статистики у табличному форматі (CSV)

## Підготовка та запуск

Для роботи з фреймворком потрібно:

1. Клонувати репозиторій:
```bash
git clone https://github.com/draklowell/numba-machine-learning.git
cd numba-machine-learning
```

2. Встановити залежності:
```bash
pip install -r requirements.txt
```

3. Створити модель нейронної мережі:
```python
sequential = Sequential(
    Input((8, 8), np.dtype("uint8")),
    CellularAutomata(rule_bitwidth=1, neighborhood="moore_1", iterations=80),
    Flatten(),
    Cast(np.dtype("float32")),
    Linear(768),
    ReLU(),
    Linear(10),
    Softmax(),
)
```

4. Налаштувати пайплайн генетичного алгоритму:
```python
# Створення пайплайнів для хромосом
chromosome_pipelines = [
    ChromosomePipeline(
        mutation=GaussianMutation(0.1, 0.5),
        crossover=SinglePoint(),
    ),
    # Додаткові пайплайни за потребою
]

pipeline = GenomePipeline(
    selection=RouletteSelection(14),
    elitarism_selection=BestSelection(3),
    pipelines=chromosome_pipelines,
)
```

5. Створити менеджер даних:
```python
sklear_manager = SklearnBalancedDataLoader(
    batch_size=10,
    process_device=Device.CPU,
    storage_device=Device.CPU,
    random_state=42,
)
```

6. Запустити алгоритм:
```python
# Створення директорії для збереження результатів
os.makedirs("generations", exist_ok=True)

manager = Manager(
    sequential=sequential,
    fitness_evaluator=FitnessEvaluator(),
    data_manager=sklear_manager,
    genome_pipeline=pipeline,
    handlers=[
        TableHandler(
            log_file=open("log.csv", "w"),
            log_period=1,
        ),
        PrintHandler(period=1),
        SaveHandler(path="generations/{generation}.pkl", period=10)
    ],
    device=Device.CPU,
    population_size=10,
)

manager.run(10)  # запуск на 10 поколінь
```

## Розподіл роботи

- **Кривий Андрій** - фреймворк нейромереж
- **Леник Нікіта** - дані, квантизація та фітнес
- **Шимановський Владислав** - генетичний алгоритм (Mutation and Crossover), README
- **Максимчук Іван** - генетичний алгоритм (Selection), звіт

---

Проєкт створено студентами Українського Католицького Університету в межах курсу "Дискретна Математика 2".