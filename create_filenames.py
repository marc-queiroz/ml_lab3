import sys
import shutil
from pathlib import Path

if len(sys.argv) < 2:
    print('formato: <programa> arquivo.txt diretorio_destino')
    exit()

destino = sys.argv[2]

dirpath = Path(destino)

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
    shutil.copy(path, destino + '/' + newname.split('/')[-1])
    print('newname', newname)
