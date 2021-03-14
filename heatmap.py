import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

def heatmap(grid, iteration, ghostcells, im_dir, number_processes):
    #plot heatmap using seaborn
    #andere farbe YlGnBu /coolwarm
    sb.heatmap(grid, cmap="YlGnBu")
    plt.title("{} Iterationenen".format((iteration + 1)*ghostcells))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    
    plt.savefig(str(im_dir)+ "hm_N{:{}{}}.png".format((iteration + 1) * ghostcells, 0, 5), dpi=200)
    plt.close()

