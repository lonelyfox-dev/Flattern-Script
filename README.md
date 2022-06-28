# Flattern-Script v.2
## Состав проекта
- Директория predict: основная директория. Содержит бинарник с моделью и исполняемый файл скрипта
- Директория data (_демонстрационная_): содержит пример входных данных для скрипта
- Файл predictions.csv (_демонстрационный_): содержит выходные предсказания для данных в директории data 
- Файл empty_data.csv (_демонстрационный_): содержит информацию о строках входных данных, содержащих пустые значения 

## Использование
`./predict/predict.exe --path <путь_до_csv_файла_с_данными>`
либо
`./predict/predict.exe -p <путь_до_csv_файла_с_данными>`

Путь до файла с данными может быть как абсолютным, так и относительным от места запуска скрипта.

`./predict/predict.exe --help` - краткая справка по использованию скрипта.

_Выходные файлы сохраняются в директорию, откуда был вызван скрипт._
