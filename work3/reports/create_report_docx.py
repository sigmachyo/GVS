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
    ("ОТЧЕТ О ПРАКТИЧЕСКОЙ РАБОТЕ №3", True, 16),
    ("Тема: «Умножение матриц на CUDA с использованием разделяемой памяти»", True, 14),
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
add_body(document, "Цель работы -- закрепить навыки работы с двумерными сетками нитей CUDA, освоить работу с быстрой разделяемой памятью (__shared__) и примитивами синхронизации нитей (__syncthreads()).")
add_body(document, "Для достижения цели разработан блочный CUDA-кернел kernel_matmul_shmem, применяющий тайлинг в разделяемой памяти. Реализован паттерн проектирования «Стратегия» для класса Matrix, позволяющий переключаться между наивной реализацией и реализацией с разделяемой памятью. Корректность проверена с помощью Google Test на 343 конфигурациях размеров с эталоном Eigen. Исследование производительности проведено с помощью Google Benchmark и CUDA Events API.")

add_heading(document, "1 Разработка программного решения")
add_heading(document, "1.1 Организация разделяемой памяти и тайлинга", level=2)
add_body(document, "Разделяемая память (Shared Memory, SRAM) расположена непосредственно на мультипроцессоре (SM) и обладает латентностью порядка 20-30 тактов при пропускной способности свыше 1-2 ТБ/с, в то время как глобальная память (DRAM) имеет задержку 400-800 тактов. В блочном алгоритме нити блока размером BLOCK_SIZE x BLOCK_SIZE совместно загружают квадратные тайлы матриц A и B в разделяемую память, после чего многократно используют загруженные данные для вычисления частичных сумм.")
add_body(document, "Для синхронизации нитей блока между фазами загрузки и фазами вычисления используется примитив __syncthreads(). Вызов барьера выполняется безусловно всеми нитями блока, что предотвращает возникновение взаимных блокировок (deadlock).")

add_heading(document, "1.2 CUDA-кернел kernel_matmul_shmem", level=2)
add_body(document, "Кернел принимает легковесные объекты MatrixView по значению. Корректно обрабатываются граничные случаи: если индекс выходит за пределы матрицы, ячейка разделяемой памяти заполняется нулём, что обеспечивает правильный результат для матриц произвольной размерности.")
add_code(document, "Реализация kernel_matmul_shmem.cuh", "core/include/kernel_matmul_shmem.cuh")

add_heading(document, "1.3 Паттерн «Стратегия» в классе Matrix", level=2)
add_body(document, "В соответствии с рекомендациями задания применён паттерн проектирования «Стратегия». Метод matmul_from принимает параметр MatmulStrategy (Naive или SharedMemory), что позволяет динамически выбирать используемое CUDA-ядро. Оператор operator* по умолчанию использует новую блочную стратегию SharedMemory.")
add_code(document, "Реализация Matrix.cuh", "core/include/Matrix.cuh")

add_heading(document, "2 Тестирование корректности")
add_body(document, "Для всесторонней верификации operator* разработан параметризованный тест Google Test со всеми комбинациями размеров m, k, n ∈ {1, 2, 3, 127, 128, 129, 512} (всего 343 теста). В качестве эталона используется Eigen::MatrixXf с row-major порядком хранения. Сравнение выполняется методом isApprox с абсолютной точностью 10^-5.")
add_code(document, "Модульные тесты test_matrix.cu", "tests/src/test_matrix.cu")

add_heading(document, "3 Экспериментальное исследование производительности")
add_body(document, "Бенчмарки запущены для квадратных матриц n x n при n ∈ {16, 32, 64, 128, 256, 512, 1024}. Сравнение проводится между тремя реализациями: Eigen (CPU), CUDA Naive (work2) и CUDA Shared Memory (work3). Замер времени выполнения GPU-ядер реализован с помощью CUDA Events API, исключая накладные расходы на копирование данных.")
add_heading(document, "3.1 График реальной вычислительной сложности", level=2)
add_figure_or_placeholder(document, "real_complexity.png", "Рисунок 1 -- Реальная вычислительная сложность умножения матриц")
add_heading(document, "3.2 График ускорения", level=2)
add_figure_or_placeholder(document, "speedup.png", "Рисунок 2 -- Ускорение CUDA Shared Memory относительно наивной версии и Eigen")

add_heading(document, "4 Анализ результатов и сравнение со сложностью")
add_body(document, "Оба алгоритма умножения матриц выполняют O(n³) арифметических операций с плавающей точкой (FLOPs). При N = 1024 для вычисления результата требуется порядка 2 · 10⁹ операций умножения и сложения. Принципиальное различие между реализациями заключается в объеме трафика глобальной памяти:")
add_body(document, "1. Наивный алгоритм (работа №2): каждый поток считывает n элементов строки A и n элементов столбца B напрямую из глобальной памяти. Суммарный объем обращений к DRAM составляет 2n³ операций чтения. Высокая латентность глобальной памяти (400–800 тактов) создает узкое место по пропускной способности памяти (memory bound).")
add_body(document, "2. Блочный алгоритм с разделяемой памятью (работа №3): нити блока 16x16 кооперативно загружают тайлы матриц A и B в быструю разделяемую память (SRAM с латентностью 20–30 тактов). Каждый элемент загружается из DRAM ровно один раз и повторно используется BLOCK_SIZE = 16 раз нитями блока. Это снижает теоретическую нагрузку на шину глобальной памяти в 16 раз (до 2n³ / 16 обращений).")
add_body(document, "Экспериментальные результаты подтверждают теоретические предпосылки:")
add_body(document, "– На малых размерах (N = 16, 32) время выполнения ядер на GPU ограничено фиксированными накладными расходами запуска ядра и синхронизации (~45–50 мкс), из-за чего однопоточная реализация на CPU (Eigen) оказывается быстрее в 3.5–20 раз (0.0022 мс против 0.0446 мс на N = 16).")
add_body(document, "– Начиная с N = 64, GPU опережает CPU в 2.22 раза (0.0488 мс против 0.1082 мс), а при N = 1024 блочная реализация достигает ускорения 120.4 раза по сравнению с CPU (3.68 мс против 443.56 мс).")
add_body(document, "– По сравнению с наивной CUDA-реализацией использование разделяемой памяти на рабочих размерах матриц обеспечивает стабильный выигрыш: при N = 128 ускорение составляет 1.42 раза (0.058 мс против 0.0825 мс), а при N = 1024 — 1.34 раза (3.68 мс против 4.94 мс, чистое время сокращается на ~25–30%).")
add_body(document, "– Отличие практического ускорения (1.34x–1.42x) от теоретического предела (16x) объясняется эффективной работой аппаратного L2-кэша современных GPU в наивной версии, а также накладными расходами на барьеры синхронизации __syncthreads() и тайловую адресацию.")

add_heading(document, "ЗАКЛЮЧЕНИЕ", centered=True)
add_body(document, "В ходе выполнения практической работы №3 закреплены навыки работы с двумерными сетками нитей CUDA и освоена работа с разделяемой памятью и барьерной синхронизацией __syncthreads(). Разработан блочный кернел kernel_matmul_shmem с безопасной обработкой произвольных размерностей матриц.")
add_body(document, "Успешно реализован паттерн «Стратегия» в классе Matrix. Корректность работы подтверждена 343 тестами Google Test при точности 10^-5. Экспериментально подтверждено превосходство блочной реализации с разделяемой памятью над наивной версией благодаря многократному сокращению обращений к глобальной памяти.")

add_heading(document, "СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ", centered=True)
sources = [
    "Стандарт университета. Система менеджмента качества. Общие требования к построению, изложению и оформлению документов учебной деятельности : СТУ 7.5-07-2021. -- Красноярск, 2021.",
    "CUDA C++ Programming Guide [Электронный ресурс]. -- URL: https://docs.nvidia.com/cuda/cuda-c-programming-guide/ (дата обращения: 12.10.2026).",
    "Kirk D. B., Hwu W. W. Programming Massively Parallel Processors: A Hands-on Approach. -- 4th Edition. -- Morgan Kaufmann, 2022.",
    "Eigen documentation [Электронный ресурс]. -- URL: https://eigen.tuxfamily.org/dox/ (дата обращения: 12.10.2026).",
    "GoogleTest User's Guide [Электронный ресурс]. -- URL: https://google.github.io/googletest/ (дата обращения: 12.10.2026).",
    "Google Benchmark documentation [Электронный ресурс]. -- URL: https://github.com/google/benchmark (дата обращения: 12.10.2026).",
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
