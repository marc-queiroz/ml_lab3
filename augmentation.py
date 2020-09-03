import sys
import Augmentor
from pathlib import Path

if len(sys.argv) == 3:
    delete = None
    origem = sys.argv[1]
    destino = sys.argv[2]
elif len(sys.argv) == 4:
    delete = sys.argv[1]
    origem = sys.argv[2]
    destino = sys.argv[3]
else:
    print('formato: <programa> [--delete] diretorio_origem diretorio_destino')
    exit()

origem = Path(origem)
destino = Path(destino)

if not destino.exists():
    destino.mkdir()

if delete and delete == '--delete':
    path = destino
    for f in path.glob('*.jpg'):
        f.unlink()

print('output', destino.absolute().as_posix())

p = Augmentor.Pipeline(source_directory=origem.absolute().as_posix(), output_directory=destino.absolute().as_posix())
# Point to a directory containing ground truth data.
# Images with the same file names will be added as ground truth data
# and augmented in parallel to the original data.
# p.ground_truth("/path/to/ground_truth_images")
# Add operations to the pipeline as normal:
#p.flip_left_right(probability=0.5)
# p.zoom_random(probability=0.5, percentage_area=1.2)
#p.flip_top_bottom(probability=0.5)
p.invert(probability=1)
# p.shear(probability=1, max_shear_left=5, max_shear_right=5)
p.rotate(probability=1, max_left_rotation=3, max_right_rotation=3)
# p.skew_corner(probability=1, magnitude=0.5)
p.skew_tilt(probability=1, magnitude=0.1)
p.random_distortion(probability=0.3, grid_width=10, grid_height=5, magnitude=8)
# p.gaussian_distortion(probability=1, grid_width=10, grid_height=5,
p.zoom(probability=1, min_factor=0.4, max_factor=1)
p.invert(probability=1)
p.sample(5000)
