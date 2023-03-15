from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

root = './data/out/'

start = 1
start = 9
fig, ax = plt.subplots(1,8,figsize=(24,3),dpi=200)
for name in range(8):
    img = f'{root}recon-{name+start}0999.png'
    img = np.asarray(Image.open(img))
    title = f'recon-{name+start}0999.png'
    ax[name].imshow(img)
    ax[name].set_axis_off()
    ax[name].set_title(title)
fig.savefig(f'{root}recon-sel.png')

fig, ax = plt.subplots(1,8,figsize=(24,3),dpi=200)
for name in range(8):
    img = f'{root}random-{name+start}0999.png'
    img = np.asarray(Image.open(img))
    title = f'random-{name+start}0999.png'
    ax[name].imshow(img)
    ax[name].set_axis_off()
    ax[name].set_title(title)
fig.savefig(f'{root}random-sel.png')

