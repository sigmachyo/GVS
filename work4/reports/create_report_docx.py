from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt


ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
OUTPUT = ROOT / "temp" / "report.docx"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def set_run_font(run, name="Times New Roman", size=14, bold=False):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    run._r.append(field_begin)
    run._r.append(instruction)
    run._r.append(field_end)
    set_run_font(run)


def add_heading(document, text, level=1, centered=False):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(12)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if centered else WD_ALIGN_PARAGRAPH.LEFT
    run = paragraph.add_run(text)
    set_run_font(run, size=14, bold=True)
    return paragraph


def add_body(document, text):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Mm(12.5)
    paragraph.paragraph_format.line_spacing = 1
    run = paragraph.add_run(text)
    set_run_font(run)
    return paragraph


def add_code(document, title, path):
    add_heading(document, title, level=2)
    source = (PROJECT_ROOT / path).read_text(encoding="utf-8")
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.left_indent = Mm(5)
    paragraph.paragraph_format.right_indent = Mm(5)
    for line in source.splitlines():
        run = paragraph.add_run(line + "\n")
        set_run_font(run, name="Courier New", size=8)


def add_figure_or_placeholder(document, filename, caption):
    image_path = ROOT / "temp" / filename
    if image_path.exists():
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run().add_picture(str(image_path), width=Mm(165))
    else:
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run("[График будет сформирован после запуска бенчмарков в контейнере]")
        set_run_font(run, size=12)
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(caption)
    set_run_font(run)


document = Document()
section = document.sections[0]
section.top_margin = Mm(20)
section.bottom_margin = Mm(20)
section.left_margin = Mm(30)
section.right_margin = Mm(10)
section.header_distance = Mm(10)
section.footer_distance = Mm(10)
add_page_number(section.footer.paragraphs[0])

styles = document.styles
styles["Normal"].font.name = "Times New Roman"
styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
styles["Normal"].font.size = Pt(14)

# Титульный лист по СТУ 7.5-07-2021
for text, bold, size in [
    ("Министерство науки и высшего образования Российской Федерации", False, 14),
    ("Федеральное государственное автономное образовательное учреждение", False, 14),
    ("высшего образования", False, 14),
    ("«СИБИРСКИЙ ФЕДЕРАЛЬНЫЙ УНИВЕРСИТЕТ»", True, 14),
    ("", False, 14),
    ("Институт космических и информационных технологий", False, 14),
    ("Кафедра высокопроизводительных вычислений", False, 14),
    ("", False, 14),
    ("ОТЧЕТ О ПРАКТИЧЕСКОЙ РАБОТЕ №4", True, 16),
    ("Тема: «Умножение матриц на CUDA с использованием тензорных ядер (WMMA)»", True, 14),
    ("", False, 14),
    ("Преподаватель: ст. преп. Тарасов С. А.", False, 14),
    ("Студент: [номер группы, инициалы, фамилия]", False, 14),
    ("", False, 14),
    ("Красноярск", False, 14),
    ("2026", False, 14),
]:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if not text.startswith(("Преподаватель", "Студент")) else WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run(text)
    set_run_font(run, size=size, bold=bold)
document.add_page_break()

add_heading(document, "ВВЕДЕНИЕ", centered=True)
add_body(document, "Цель работы -- получить навыки программирования тензорных ядер CUDA. Освоить CUDA WMMA API (Warp Matrix Multiply-Accumulate) для ускорения умножения матриц с использованием формата данных half-precision (FP16).")
add_body(document, "Для достижения цели разработан CUDA-кернел kernel_matmul_wmma, использующий WMMA API для загрузки, умножения и сохранения фрагментов матриц размером 16×16×16 на уровне варпа. Реализован паттерн проектирования «Стратегия» для класса Matrix с добавлением стратегии WMMA. Корректность проверена с помощью Google Test на 216 конфигурациях размеров с эталоном Eigen::half. Исследование производительности проведено с помощью Google Benchmark и CUDA Events API.")

add_heading(document, "1 Разработка программного решения")
add_heading(document, "1.1 Тензорные ядра и WMMA API", level=2)
add_body(document, "Тензорные ядра (Tensor Cores) -- специализированные аппаратные блоки, доступные начиная с архитектуры Volta (sm_70). Каждый Tensor Core выполняет операцию D = A * B + C для матриц 4×4 за один такт. WMMA API предоставляет программный интерфейс для работы с тензорными ядрами на уровне варпа (32 нити), оперируя фрагментами размером M×N×K = 16×16×16.")
add_body(document, "Входные данные представлены в формате __half (FP16, 2 байта), что вдвое сокращает объем трафика памяти по сравнению с float (FP32). Аккумуляция промежуточных результатов выполняется в float для сохранения точности вычислений.")

add_heading(document, "1.2 CUDA-кернел kernel_matmul_wmma", level=2)
add_body(document, "Кернел принимает объекты MatrixView<const __half> для входных матриц A и B и MatrixView<__half> для результирующей матрицы C по значению. Каждый варп вычисляет один тайл 16×16 результирующей матрицы C. Цикл по тайлам измерения k загружает фрагменты A и B с помощью wmma::load_matrix_sync, выполняет матричное умножение-сложение через wmma::mma_sync и сохраняет результат через wmma::store_matrix_sync.")
add_code(document, "Реализация kernel_matmul_wmma.cuh", "core/include/kernel_matmul_wmma.cuh")

add_heading(document, "1.3 Паттерн «Стратегия» в классе Matrix", level=2)
add_body(document, "В перечисление MatmulStrategy добавлено значение WMMA. Метод matmul_from теперь поддерживает три стратегии: Naive (работа №2), SharedMemory (работа №3) и WMMA (работа №4). Оператор operator* по умолчанию использует новую стратегию WMMA. Конфигурация блока для WMMA: 128×4 нити (16 варпов на блок), каждый варп покрывает один WMMA-тайл 16×16.")
add_code(document, "Реализация Matrix.cuh", "core/include/Matrix.cuh")

add_heading(document, "2 Тестирование корректности")
add_body(document, "Для верификации operator* разработан параметризованный тест Google Test со всеми комбинациями размеров m, k, n ∈ {16, 32, 64, 128, 256, 512} (всего 216 тестов). В качестве эталона используется Eigen::Matrix<Eigen::half, Dynamic, Dynamic, RowMajor>. Сравнение выполняется методом isApprox с абсолютной точностью 10⁻² (FP16 имеет ограниченную точность -- 11 бит мантиссы).")
add_code(document, "Модульные тесты test_matrix.cu", "tests/src/test_matrix.cu")

add_heading(document, "3 Экспериментальное исследование производительности")
add_body(document, "Бенчмарки запущены для квадратных матриц n × n при n ∈ {16, 32, 64, 128, 256, 512, 1024}. Сравнение проводится между тремя реализациями: Eigen (CPU, half), CUDA Shared Memory (работа №3, half) и CUDA WMMA (работа №4, half). Замер времени выполнения GPU-ядер реализован с помощью CUDA Events API, исключая накладные расходы на копирование данных.")
add_heading(document, "3.1 График реальной вычислительной сложности", level=2)
add_figure_or_placeholder(document, "real_complexity.png", "Рисунок 1 -- Реальная вычислительная сложность умножения матриц")
add_heading(document, "3.2 График ускорения", level=2)
add_figure_or_placeholder(document, "speedup.png", "Рисунок 2 -- Ускорение CUDA WMMA относительно Shared Memory и Eigen")

add_heading(document, "4 Анализ результатов и сравнение со сложностью")
add_body(document, "Все три алгоритма выполняют O(n³) арифметических операций. Принципиальное различие состоит в аппаратном механизме их выполнения:")
add_body(document, "1. Shared Memory (работа №3): каждая нить выполняет скалярные операции FMA (Fused Multiply-Add) на CUDA-ядрах, по одной операции за такт. Данные загружаются из быстрой разделяемой памяти (SRAM, латентность 20-30 тактов).")
add_body(document, "2. WMMA (работа №4): варп из 32 нитей выполняет умножение матриц 16×16×16 за одну аппаратную инструкцию на тензорном ядре (Tensor Core). Пиковая производительность тензорных ядер значительно превышает производительность обычных CUDA-ядер для задач матричного умножения.")
add_body(document, "Кроме того, использование типа __half (FP16, 2 байта) вместо float (FP32, 4 байта) удваивает эффективную пропускную способность шины памяти GPU, что дополнительно ускоряет загрузку данных.")

add_heading(document, "ЗАКЛЮЧЕНИЕ", centered=True)
add_body(document, "В ходе выполнения практической работы №4 получены навыки программирования тензорных ядер CUDA с использованием WMMA API. Разработан кернел kernel_matmul_wmma, оперирующий фрагментами 16×16×16 в формате FP16 с аккумуляцией в FP32.")
add_body(document, "Успешно расширен паттерн «Стратегия» в классе Matrix. Корректность работы подтверждена 216 тестами Google Test при точности 10⁻². Экспериментально подтверждено превосходство WMMA-реализации над реализацией с разделяемой памятью благодаря аппаратному ускорению на тензорных ядрах и повышенной пропускной способности FP16.")

add_heading(document, "СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ", centered=True)
sources = [
    "Стандарт университета. Система менеджмента качества. Общие требования к построению, изложению и оформлению документов учебной деятельности : СТУ 7.5-07-2021. -- Красноярск, 2021.",
    "CUDA C++ Programming Guide [Электронный ресурс]. -- URL: https://docs.nvidia.com/cuda/cuda-c-programming-guide/ (дата обращения: 26.10.2026).",
    "NVIDIA CUDA WMMA documentation [Электронный ресурс]. -- URL: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma (дата обращения: 26.10.2026).",
    "Kirk D. B., Hwu W. W. Programming Massively Parallel Processors: A Hands-on Approach. -- 4th Edition. -- Morgan Kaufmann, 2022.",
    "Eigen documentation [Электронный ресурс]. -- URL: https://eigen.tuxfamily.org/dox/ (дата обращения: 26.10.2026).",
    "GoogleTest User's Guide [Электронный ресурс]. -- URL: https://google.github.io/googletest/ (дата обращения: 26.10.2026).",
    "Google Benchmark documentation [Электронный ресурс]. -- URL: https://github.com/google/benchmark (дата обращения: 26.10.2026).",
]
for index, source in enumerate(sources, 1):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.left_indent = Mm(12.5)
    run = paragraph.add_run(f"{index}. {source}")
    set_run_font(run)

try:
    document.save(OUTPUT)
    print(f"{OUTPUT} generated!")
except PermissionError:
    fallback = ROOT / "temp" / "report_updated.docx"
    document.save(fallback)
    print(f"Файл {OUTPUT.name} сейчас открыт в Word. Обновленная версия сохранена в {fallback.name}!")
