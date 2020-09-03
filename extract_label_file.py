import sys
import shutil
from pathlib import Path
import re

if len(sys.argv) < 2:
    print('formato: <programa> arquivo.txt diretorio_leitura')
    exit()

dirpath = Path(sys.argv[2])

print(list(dirpath.glob('*.jpg')))

f_out = open(sys.argv[1], 'w')

for f in list(dirpath.glob('*.jpg')):
    label = re.search('_(\d+)\.', str(f)).group(1)
    f_out.write(str(f) + ' ' + label + '\n')
    # print(str(f) + ' ' + label + '\n')

# arq = open(sys.argv[1], 'r')
# arq_texto = arq.read()
# files = arq_texto.split('\n')
# files.remove('')
# files.sort()
# print('Arquivos: ', len(files))
#
# for f in files:
#     path, label = f.split(' ')
#     name, ext = path.split('.')
#     newname = name + '_' + label + '.' + ext
#     shutil.copy(diretorio_origem + '/' + path, diretorio_destino + '/' + newname)
#     print('newname', newname)
