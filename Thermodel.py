import numpy as np
import matplotlib.pyplot as plt

mp_ = list()
a_ = list()

#тут задається висота
h=4

def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

def rgb_to_hex(r, g, b, alpha):
    return '#'+str(hex(r))[2:]+str(hex(g))[2:]+str(hex(b))[2:]+str(hex(alpha))[2:]

CSVData = open('temp0.csv')
a_ = np.loadtxt(CSVData, delimiter=",")
a = np.zeros((len(a_),len(a_[0]),h))
mp = np.zeros((len(a_),len(a_[0]),h))
zmt = np.zeros((len(a_),len(a_[0]),h))
colors = np.zeros((len(a_),len(a_[0]),h))

for k in range(h):
    CSVData = open('temp'+str(k)+ '.csv')
    a_ = np.loadtxt(CSVData, delimiter=",")
    CSVData = open('map'+str(k)+'.csv')
    mp_ = np.loadtxt(CSVData, delimiter=",")
    for i in range(len(a)):
        for j in range(len(a[0])):
            a[i][j][k] = a_[i][j]
            mp[i][j][k] = mp_[i][j]


mp = mp.astype('bool')
colors = colors.astype('str')

t_max=a[0][0][0]
t_min=a[0][0][0]
t_def=20.0

dt = 10 #проміжок часу 10с
dx = 0.1 #довжина грані кубика
dS=0.01

xa = 0.026 #коефіцієнт теплопередачі повітря
xh = 237 #коефіцієнт теплопередачі батареї
xw = 0.7 #коефіцієнт теплопередачі стіни з бетону
ca = 1000 #питома теплоємність повітря

th = 60 #температура батареї
tw = 15 #температура стіни

m = 0.00129 #маса повітря в одному кубику

for t in range(1,5):
    for z in range(h-1):
        for y in range(0,len(a)-1):
            for x in range(0,len(a[0])-1):
                    #рахуємо зміну температури для правої клітинки
                    zt = (xa*(a[y][x+1][z] - a[y][x][z])*dS*dt)/(dx*ca*m)
                    if zt>0:
                        if mp[y][x][z] == False:
                            zmt[y][x][z] = zmt[y][x][z] + abs(zt)
                        if mp[y][x+1][z] == False:
                            zmt[y][x+1][z] = zmt[y][x+1][z] - abs(zt)
                    else:
                        if mp[y][x][z] == False:
                            zmt[y][x][z] = zmt[y][x][z] - abs(zt)
                        if mp[y][x+1][z] == False:
                            zmt[y][x+1][z] = zmt[y][x+1][z] + abs(zt)
                    #рахуємо зміну температури для нижньої клітинки
                    zt = (xa*(a[y+1][x][z] - a[y][x][z])*dS*dt)/(dx*ca*m)
                    if zt>0:
                        if mp[y][x][z] == False:
                            zmt[y][x][z] = zmt[y][x][z] + abs(zt)
                        if mp[y+1][x][z] == False:
                            zmt[y+1][x][z] = zmt[y+1][x][z] - abs(zt)
                    else:
                        if mp[y][x][z] == False:
                            zmt[y][x][z] = zmt[y][x][z] - abs(zt)
                        if mp[y+1][x][z] == False:
                            zmt[y+1][x][z] = zmt[y+1][x][z] + abs(zt)

                    #рахуємо зміну температури для нижньої по Z клітинки
                    zt = (xa*(a[y][x][z+1] - a[y][x][z])*dS*dt)/(dx*ca*m)
                    if zt>0:
                        if mp[y][x][z] == False:
                            zmt[y][x][z] = zmt[y][x][z] + abs(zt)
                        if mp[y][x][z+1] == False:
                            zmt[y][x][z+1] = zmt[y][x][z+1] - abs(zt)
                    else:
                        if mp[y][x][z] == False:
                            zmt[y][x][z] = zmt[y][x][z] - abs(zt)
                        if mp[y][x][z+1] == False:
                            zmt[y][x][z+1] = zmt[y][x][z+1] + abs(zt)
        for y in range(1,len(a)-1):
            for x in range(1,len(a[0])-1):
                a[y][x][z] = round(a[y][x][z]+zmt[y][x][z],1)
                zmt[y][x][z] = 0
                if a[y][x][z] > t_max:
                  t_max=a[y][x][z]
                if a[y][x][z] < t_min:
                  t_min=a[y][x][z]


for elem in a:
    print(*elem)

n_voxels = np.zeros((len(a), len(a[0]), h), dtype=bool)

alpha = 0.9
for z in range(h):
      for y in range(0,len(a)):
          for x in range(0,len(a[0])):
            if a[y][x][z] != t_def:
              n_voxels[y, x, z] = True
              colors[y][x][z] = rgb_to_hex(255, int(round(255-(t_max-a[y][x][z])/(t_max-t_min)*240,0)), int(round(255-(t_max-a[y][x][z])/(t_max-t_min)*240,0)), int(round(alpha*255,0)))
              #print(int(round(255-(t_max-a[y][x][z])/(t_max-t_min)*240,0)))
              #print(a[y][x][z])
              #print(colors[y][x][z])
              #print(t_max)
              #print('\n')

ax = plt.figure().add_subplot(projection='3d')
#ax.voxels(n_voxels, facecolors=colors)
ax.voxels(n_voxels)
ax.set_aspect('equal')

plt.show()
#axes = [len(a), len(a[0]), h]
#data = np.ones(axes)
#alpha = 0.9
#colors = np.empty(axes + [4], dtype=np.float32)
#colors[:] = [1, 1-(t_max-a[y][x][z])/(t_max-t_min)*240/255, 1-(t_max-a[y][x][z])/(t_max-t_min)*240/255, alpha]  # red