# Flattern-Script v.1
## Состав проекта
- Директория dist: содержит все исходные файлы и библиотеки, а также сам исполняемый файл
- Директория model: содержит бинарник модели-классификатора
- Файл predict.py: собственно скрипт. Получает путь до данных, загружает модель, вырабатывает предсказания и выдаёт csv файл (сохраняется в корень проекта)

## Использование (из корневой папки проекта)
`./dist/predict/predict.exe --path <путь_до_csv_файла_с_данными>`
либо
`./dist/predict/predict.exe -p <путь_до_csv_файла_с_данными>`

`./dist/predict/predict.exe --help` - краткая справка по использованию скрипта.

_./dist/predict/predict.exe - исполняемый файл должен находиться в указанном месте, чтобы не ломались внутренние зависимости. Можно установить псевдоним для более быстрого использования._
