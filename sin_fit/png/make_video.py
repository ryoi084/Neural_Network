import os

os.system('ffmpeg -r 2 -i result_epoch=%03d.png -vcodec libx264 -pix_fmt yuv420p -r 4 out.mp4')
os.system('ffmpeg -i out.mp4 out.gif')
