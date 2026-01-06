import sys
import os

# Словарь сопоставления расширений с типами блоков Markdown
EXT_MAP = {
    '.py': 'python',
    '.yml': 'yaml',
    '.yaml': 'yaml',
    '.json': 'json',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.html': 'html',
    '.css': 'css',
    '.sh': 'bash',
    '.bash': 'bash',
    '.sql': 'sql',
    '.md': 'markdown',
    '.dockerfile': 'dockerfile',
    '.txt': ''  # Обычный блок без подсветки
}

def process_files(files):
    for filepath in files:
        if not os.path.exists(filepath):
            print(f"--- Ошибка: Файл {filepath} не найден ---\n")
            continue
            
        if os.path.isdir(filepath):
            print(f"--- Ошибка: {filepath} является директорией ---\n")
            continue

        _, ext = os.path.splitext(filepath)
        # Определяем язык для блока, по умолчанию используем расширение без точки
        lang = EXT_MAP.get(ext.lower(), ext.lower().strip('.'))

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            print(f"**Файл: {filepath}**")
            print(f"```{lang}")
            print(content)
            print("```")
            print() # Пустая строка между файлами
        except Exception as e:
            print(f"--- Ошибка при чтении файла {filepath}: {e} ---\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python md_wrap.py <файл1> <файл2> ...")
    else:
        # Передаем все аргументы командной строки, кроме имени самого скрипта
        process_files(sys.argv[1:])
