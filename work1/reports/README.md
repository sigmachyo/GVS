# Отчётные материалы

Эта папка содержит только материалы сдачи и инструменты их генерации. Файлы
`core/`, `tests/` и `benchmarks/` относятся к функциональной части проекта и
находятся уровнем выше.

`benchmark_results.json`, PNG-графики и HTML-графики намеренно не хранятся в
репозитории: они создаются после запуска на машине с NVIDIA CUDA и зависят от
конкретной видеокарты.

Запуск из каталога `work1`:

```bash
./run.sh
```

Для Docker нужен NVIDIA Container Toolkit:

```bash
docker build -t gvs_work1 .
docker run --gpus all gvs_work1
```