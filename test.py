'''
S — существительное
V — глагол
A — прилагательное
ADV — наречие или частица
NI — местоимение
PR — предлог
CONJ — союз
'''
from lemmatizer_old import *
d=load_dictionary(DATA_DIR)
print(process_line('Праймовые лабубы могают нормисных тюбиков. Сигмы рофлят, а нормисы тильтуют', d))