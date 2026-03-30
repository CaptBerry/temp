# Классификация геокуба с использованием 3D CNN (PyTorch)

Скрипт `geocube_3dcnn.py` реализует полный пайплайн:
1. Загрузка геокуба и скважин.
2. Проверка координат и попадания точек в границы.
3. Нормализация и обработка `NaN`.
4. Разделение по скважинам (не случайно по точкам).
5. Формирование 3D patch-ов и `GeoDataset`.
6. Обучение простой 3D CNN.
7. Сохранение:
   - `model.pth`
   - `norm_stats.npz`
   - `probability_cube.txt`

## Форматы входа

- Геокуб: `.txt`, размер `[Nx, Ny, Nz, C]` (в файле значения должны быть в количестве `Nx*Ny*Nz*C`).
- Скважины: `.txt`, колонки:
  - минимум `[x, y, z, label]`
  - опционально 5-я колонка `well_id` для явного группового split.

## Быстрый запуск

```bash
python geocube_3dcnn.py \
  --cube_txt data/cube.txt \
  --wells_txt data/wells.txt \
  --nx 100 --ny 80 --nz 60 --c 4 \
  --patch 5 5 5 \
  --batch_size 32 \
  --norm zscore \
  --epochs 10 \
  --out_dir artifacts
```

### Важно для Windows PowerShell

В PowerShell нельзя переносить команду через `\` или добавлять буквальный `\n` как аргумент — это и вызывает ошибку `unrecognized arguments: \n`.

Используйте **одну строку**:

```powershell
python .\geocube_3dcnn.py --cube_txt data/cube.txt --wells_txt data/wells.txt --nx 100 --ny 80 --nz 60 --c 4 --patch 5 5 5 --batch_size 32 --norm zscore --epochs 10 --out_dir artifacts
```

Или перенос через **обратную кавычку**:

```powershell
python .\geocube_3dcnn.py `
  --cube_txt data/cube.txt `
  --wells_txt data/wells.txt `
  --nx 100 --ny 80 --nz 60 --c 4 `
  --patch 5 5 5 `
  --batch_size 32 `
  --norm zscore `
  --epochs 10 `
  --out_dir artifacts
```

## Загрузка из ваших путей (marks + not_marks)

Добавлен режим подготовки таблиц с путями по умолчанию:

- `W:/3d/kovikta/Work/Well_new_pogl_filled.csv`
- `W:/3d/kovikta/Work/New_kov_pravki_more.txt`

Внутри используется логика:

- `pd.read_csv(file_path_marks)`
- `pd.read_csv(file_path_not_marks, sep="\t")`
- переименование `X/Y/Z -> Cube_X/Cube_Y/Cube_Z`
- проверка набора `FEATURES = ["Cube_X","Cube_Y","Cube_Z","layer","Rho","S","thgr","Sgr","krgr","fml","kp"]`

Запуск:

```bash
python geocube_3dcnn.py \
  --prepare_kovikta_only \
  --marks_csv "W:/3d/kovikta/Work/Well_new_pogl_filled.csv" \
  --unlabeled_txt "W:/3d/kovikta/Work/New_kov_pravki_more.txt" \
  --label_col label \
  --out_dir artifacts
```

Результаты:

- `artifacts/wells_from_marks.txt` (массив `[x,y,z,label]`)
- `artifacts/unlabeled_features_preview.csv` (выгрузка `FEATURES`)

## Примечания

- PyTorch вход: `[B, C, Px, Py, Pz]`.
- Обработка границ patch: `padding` (`reflect`).
- Балансировка классов: веса в `CrossEntropyLoss`.
- Для `multiclass` поддерживаются метки `0..K-1`.
