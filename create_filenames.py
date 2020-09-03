import sys
import shutil
from pathlib import Path

if len(sys.argv) < 3:
    print('formato: <programa> arquivo.txt diretorio_origem diretorio_destino')
    exit()

diretorio_origem = sys.argv[2]

diretorio_destino = sys.argv[3]

dirpath = Path(sys.argv[3])

if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)

dirpath.mkdir()

origem = Path(sys.argv[2])

# print(list(origem.glob('*.jpg')))

arq = open(sys.argv[1], 'r')
arq_texto = arq.read()
files = arq_texto.split('\n')
files.remove('')
files.sort()
print('Arquivos: ', len(files))

for f in files:
    path, label = f.split(' ')
    name, ext = path.split('.')
    newname = name + '_' + label + '.' + ext
    shutil.copy(diretorio_origem + '/' + path, diretorio_destino + '/' + newname)
    print('newname', newname)
