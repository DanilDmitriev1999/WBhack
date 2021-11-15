import pymorphy2

morph = pymorphy2.MorphAnalyzer()
STOP_PUNCT = list(',./!@#$%^&*()_+=-<>?\|{}[]`~/')
STOP = set(
    ["скидка", "скидкой", "скидки", "скидке", "скидкой", "скидке", "недорого", "дешево",
     "в", "на", "для", "о", "у", "и", "с", "из"] + STOP_PUNCT
     )

def counter(s: str) -> dict:
    """
     Словарь, который позволяет нам считать количество неизменяемых объектов

     Args:
        s: Входная строка, по которой строится словарь
     Returns:
         Количество неизменяемых объектов
    """
    d = {}
    for i in s:
        if i not in d:
            d[i] = 0
        d[i] += 1
    return d
    
def prepare4check(s1: str, s2: str, STOP: set = STOP, morph=morph) -> list:
    """
     Предобработка данных для проверки

     Args:
        s1: Первая сравнимая строка
        s2: Вторая сравнимая строка
        STOP: множество стоп слов, которые мы хотели бы исключать
        morph: Морфологический анализатор, для лемматизации слов
     Returns:
         Список предобработанных данных:
             set_s1: уникальные слова первой строки с учетом удаленных стоп слов
             set_s2: уникальные слова второй строки с учетом удаленных стоп слов
             diff_s1: Разница между множеством слов 1 строки и множеством слов 2 строки
             diff_s2: Разница между множеством слов 2 строки и множеством слов 1 строки
    """
    s1 = s1.lower()
    s2 = s2.lower()
    s1 = [morph.parse(i)[0].normal_form for i in s1.split(' ')]
    s2 = [morph.parse(i)[0].normal_form for i in s2.split(' ')]
    set_s1 = set(s1) - STOP
    set_s2 = set(s2) - STOP

    diff_s1 = ' '.join(list(set_s1 - set_s2))
    diff_s2 = ' '.join(list(set_s2 - set_s1))
    
    return [set_s1, set_s2, diff_s1, diff_s2]

def easy_check(s1: str, s2: str, STOP: set = STOP) -> bool:
    """
     Простой уровень проверки. Есть 3 типа проверки:
         1: если s1 имеет такие же слова, как и s2
         2: если s1 входит в множество слов s2 (предполагаем, что s2 хранит дополнительные признаки, например s1=обувь, а s2=обувь Адидас)
         3: если s2 входит в множество слов s1 (предполагаем, что s2 не хранит никакой дополнительной информацией, а является частью s1)
     Args:
        s1: Первая сравнимая строка
        s2: Вторая сравнимая строка
        STOP: множество стоп слов, которые мы хотели бы исключать
     Returns:
         результат всех условий первой проверки
    """
    set_s1, set_s2, diff_s1, diff_s2 = prepare4check(s1, s2, STOP)
    if set_s1 == set_s2:
        return False
    if len(diff_s1) == 0:
        return True
    if len(diff_s2) == 0:
        return False
    return True

def check(s1: str, s2: str, STOP: set = STOP, morph=morph) -> bool:
    """
     Более сложный уровень проверки. Есть 4 типа проверки:
         1: если s1 имеет такие же слова, как и s2
         2: если s1 входит в множество слов s2 (предполагаем, что s2 хранит дополнительные признаки, например s1=обувь, а s2=обувь Адидас)
         3: если s2 входит в множество слов s1 (предполагаем, что s2 не хранит никакой дополнительной информацией, а является частью s1)
         4: проверяем частотность минимальной строки, к максимальной, чтобы определить разницу между количеством уникальных токенов
     Args:
        s1: Первая сравнимая строка
        s2: Вторая сравнимая строка
        STOP: множество стоп слов, которые мы хотели бы исключать
        morph: Морфологический анализатор, для лемматизации слов
     Returns:
         результат всех условий второй проверки
    """
    set_s1, set_s2, diff_s1, diff_s2 = prepare4check(s1, s2, STOP)
    if set_s1 == set_s2:
        return False
    
    if len(diff_s1) == 0:
        return True
    if len(diff_s2) == 0:
        return False

    dt = {len(diff_s1): diff_s1, len(diff_s2): diff_s2}

    c = 0
    max_s, min_s = dt[max(len(diff_s1), len(diff_s2))], dt[min(len(diff_s1), len(diff_s2))]
    c_s1 = counter(min_s)
    c_s2 = counter(max_s)
    for i in min_s:
        if i in c_s2 and c_s2[i] > 0:
            c += 1
            c_s2[i] -= 1
        else:
            c -= 1
    if (c / len(min_s)) < 1.0:
        return True
    return False