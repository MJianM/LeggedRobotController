"""
~~~~~~~~~~~~~~~~~~~~~~~~~~
Copyright @ Changda Tian, Ziqi MA
2022.12
SJTU RL2 LAB
~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
import numpy as np
import math
from scipy.spatial.transform import Rotation as Rot
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import cv2


class Gridmap:
    def __init__(self,len=0,wid=0,res=10) -> None:
        self.Length = len
        self.Width = wid
        self.Resolution = res # how many sample points in 1m.

        self.map = np.zeros((self.Length*self.Resolution+1,self.Width*self.Resolution+1))

        self.max_height = 0

        self.map_name:str = "gm"
        self.xml_file:str = "littledog.xml"

        self.rbt_pos = np.zeros(3)
        self.rbt_dir = np.zeros(3)
        self.rbt_dir_quat = np.array([1,0,0,0])
        
    def show_map(self):
        print("\n Global map shape: ",self.map.shape)
        print(self.map)

    def set_map_name(self,map_name:str):
        self.map_name = map_name

    def set_xml_name(self,xml_name:str):
        self.xml_file = xml_name


    def __get_plane(self, p1:np.ndarray, p2:np.ndarray, p3:np.ndarray):
        '''
            p: ndarray (3,1)
            
            ax+by+cz=1

            return: ndarray (3,) a,b,c
        '''
        A = np.hstack([p1,p2,p3]).T
        assert np.linalg.det(A) != 0
        params = np.linalg.inv(A) @ np.ones((3,1))
        return params.flatten()

    # def __get_plane(self,p1,p2,p3):
    #     '''
    #     a*x+b*y+c*z+d = 0
    #     '''
    #     a = (p2.y-p1.y)*(p3.z-p1.z)-(p2.z-p1.z)*(p3.y-p1.y) 
    #     b = (p2.z-p1.z)*(p3.x-p1.x)-(p2.x-p1.x)*(p3.z-p1.z)
    #     c = (p2.x-p1.x)*(p3.y-p1.y)-(p2.y-p1.y)*(p3.x-p1.x)
    #     d = 0-(a*p1.x+b*p1.y+c*p1.z)
    #     return a,b,c,d

    def height(self,x,y):
        '''
        对落脚点周围地形进行平面拟合，得到高度值
        '''
        row_point = (x + self.Length/2) * self.Resolution
        col_point = (y + self.Width/2) * self.Resolution
        left_row_point = math.floor(row_point)
        right_row_point = left_row_point + 1
        left_col_point = math.floor(col_point)
        right_col_point = left_col_point +1
        hlu = self.map[left_row_point,left_col_point]
        hld = self.map[right_row_point,left_col_point]
        hru = self.map[left_row_point,right_col_point]
        # hrd = self.map[right_row_point,right_col_point]

        p1 = np.array([left_row_point, left_col_point, hlu]).reshape((3,1))
        p2 = np.array([right_row_point, left_col_point, hld]).reshape((3,1))
        p3 = np.array([left_row_point, right_col_point, hru]).reshape((3,1))

        params = self.__get_plane(p1,p2,p3)

        return (1 - params[0] * row_point - params[1] * col_point)/ params[2]

    def visualize(self):
        '''
            纵向为X, 横向为Y
        '''
        x = np.linspace(-self.Length/2,self.Length/2, self.Resolution*self.Length+1)
        y = np.linspace(-self.Width/2,self.Width/2, self.Resolution*self.Width+1)
        X,Y = np.meshgrid(x, y)
        X = X.T
        Y = Y.T
        Z = np.copy(self.map)

        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(12,10))
        ls = LightSource(270, 20)
        rgb = ls.shade(Z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
                                linewidth=0, antialiased=False, shade=False)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def gen_rand_tough_terrain(self,variance):
        # tm = np.zeros((self.Length*self.Resolution+1,self.Width*self.Resolution+1))
        # for i in range(self.Length*self.Resolution+1):
        #     tmp = np.random.random(self.Width*self.Resolution+1)
        #     tm[i] = tmp
        tm = np.random.random((self.Length*self.Resolution+1,self.Width*self.Resolution+1))
        tm = tm * variance

        np.copyto(self.map, tm)    
        self.max_height = np.max(self.map)

        # for i in range(len(self.map)):
        #     for j in range(len(self.map[i])):
        #         self.map[i][j] = tm[i][j]
        #         if tm[i][j] > self.max_height:
        #             self.max_height = tm[i][j]
        
    def set_block_height(self,x_range,y_range,height):
        x_left = math.floor((x_range[0]+self.Length/2)*self.Resolution)
        x_right = math.ceil((x_range[1]+self.Length/2)*self.Resolution)
        y_left = math.floor((y_range[0]+self.Width/2)*self.Resolution)
        y_right = math.ceil((y_range[1]+self.Width/2)*self.Resolution)

        self.map[x_left:x_right+1, y_left:y_right+1] = height
        # for i in range(x_left,x_right+1):
        #     for j in range(y_left,y_right+1):
        #         self.map[i,j] = height
        
        if height > self.max_height:
            self.max_height = height

    def put_robot(self,x,y,z, row, pitch, yaw):
        # put the robot to world.
        # x-y-z in extrinstic frame
        self.rbt_pos[0] = x
        self.rbt_pos[1] = y
        self.rbt_pos[2] = z
        self.rbt_dir[0] = row
        self.rbt_dir[1] = pitch
        self.rbt_dir[2] = yaw

        # x-y-z in extrinstic frame
        tmp = Rot.as_quat(Rot.from_euler("xyz", self.rbt_dir, False))
        self.rbt_dir_quat[0] = tmp[3]
        self.rbt_dir_quat[1:4] = tmp[0:3]
        print("rbt_pos: ",self.rbt_pos.flatten(),"rbt_dir_quat: ",self.rbt_dir_quat)


    def parse_xml(self):
        '''
            robot pos, quat , map 写入xml文件
        '''
        template_xml = []
        with open("./model/littledog.xml",'r') as f:
            for i in f:
                template_xml.append(i)

        hf_pos = []
        for i in range(len(template_xml)):
            if "hfield" in template_xml[i]:
                hf_pos.append(i)
        change_string = template_xml[hf_pos[0]]
        file_index = change_string.find("file=")
        png_index = change_string.find(".png") 
        change_string = change_string[:file_index+6] + self.map_name + change_string[png_index:]

        size_index = change_string.find("size=")
        end_index = change_string.find(" />")
        size_str = f"{self.Length/2} {self.Width/2} {self.max_height} 0.001"
        change_string = change_string[:size_index+6] + size_str + change_string[end_index-1:]

        template_xml[hf_pos[0]] = change_string

        body_pos_dir_index = 0
        for i in range(len(template_xml)):
            if '''body name="base"''' in template_xml[i]:
                body_pos_dir_index = i
                break
        pos_string = template_xml[body_pos_dir_index]
        pos_index = pos_string.find("pos=")
        pos_end = pos_string.find(">")
        pos_string = pos_string[:pos_index+4] + f'''"{self.rbt_pos[0]} {self.rbt_pos[1]} {self.rbt_pos[2]+0.2}"''' + pos_string[pos_end:]


        pos_end = pos_string.find(">")
        pos_string = pos_string[:pos_end] + f''' quat="{self.rbt_dir_quat[0]} {self.rbt_dir_quat[1]} {self.rbt_dir_quat[2]} {self.rbt_dir_quat[3]}">\n'''
        template_xml[body_pos_dir_index] = pos_string

        with open(f"./model/{self.xml_file}",'w') as f:
            for i in template_xml:
                f.writelines(i)

    def __min_max_normalize(self,v,min_v,max_v):
        if max_v-min_v ==0:
            return 0
        return (v-min_v) / (max_v-min_v)

    def map_to_img(self,res=100):
        # res must larger than self.Resolution and be its mutiple.
        img = np.zeros((self.Length*res+1,self.Width*res+1))
        max_h = np.max(self.map)
        min_h = np.min(self.map)
        for i in range(self.Length*res+1):
            for j in range(self.Width*res+1):
                tmp_h = self.map[math.floor(i*(self.Resolution/res)),math.floor(j*(self.Resolution/res))]
                img[i,j] = self.__min_max_normalize(tmp_h,min_h,max_h)
        img = img*255
        # rotate because in mujoco sim, the robot is horizontal view. ???
        img = np.rot90(img,1)
        cv2.imwrite(f'./model/{self.map_name}.png',img)
        # cv2.imwrite(f'./model/meshes/{self.map_name}.png',img)

    def __min_max_denormalize(self,v,min_v,max_v):
        return min_v+v*(max_v-min_v) 

    def img_to_map(self,img_file,min_h,max_h,res=100):
        img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
        img = np.rot90(img,-1)
        for i in range(self.Length*self.Resolution+1):
            for j in range(self.Width*self.Resolution+1):
                kernel_h = int(res / self.Resolution)
                avg_mat = np.ones((kernel_h,kernel_h))
                avg_mat = avg_mat / kernel_h**2 / 255
                self.map[i,j] = self.__min_max_denormalize(np.sum(img[i*kernel_h:(i+1)*kernel_h,j*kernel_h:(j+1)*kernel_h] * avg_mat),min_h,max_h)


if __name__ == '__main__':

    m = Gridmap(len=5, wid=3, res=10)
    m.gen_rand_tough_terrain(0.5)
    m.map_to_img(res=10)
    m.visualize()
