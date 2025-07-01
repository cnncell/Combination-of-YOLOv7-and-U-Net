import cv2
import numpy as np
from PIL import Image
import pandas as pd
from yolo import YOLO, YOLO_ONNX
import csv
import math
import time
import os
import subprocess
import logging
import shutil
#---------------------------------------------------#
#   Synchronization
#---------------------------------------------------#
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='sync.log'
)

def generate_script(session_name, host, username, password, local_path, remote_path):
    """生成WinSCP脚本"""
    # 验证本地路径是否存在
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"本地路径不存在: {local_path}")
    
    # 禁用主机密钥验证的正确方式
    script = f"""# 同步脚本 - 接受任何主机密钥
open sftp://{username}:{password}@{host}/ -hostkey=*
option batch on
option confirm off
lcd {local_path}
cd {remote_path}
synchronize both -delete -criteria=time -mirror
close
exit
"""
    script_path = f"{session_name}.txt"
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    
    logging.info(f"脚本已保存到: {os.path.abspath(script_path)}")
    return script_path

def run_winscp(script_path, winscp_path=r'C:/Program Files (x86)/WinSCP/WinSCP.exe'):
    """执行WinSCP脚本"""
    try:
        # 验证脚本文件是否存在
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"脚本文件不存在: {script_path}")

        # 检查WinSCP可执行文件是否存在
        if not os.path.exists(winscp_path):
            raise FileNotFoundError(f"WinSCP可执行文件不存在: {winscp_path}")

        logging.info("开始执行同步任务")
        
        # 构建并执行命令
        command = [
            winscp_path, 
            '/script=' + script_path, 
            '/log=sync.log', 
            '/loglevel=2',
            '/noverifycert'  # 备选方案：禁用证书验证
        ]
        logging.info(f"执行命令: {' '.join(command)}")

        # 使用communicate方法并设置超时
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            stdout, stderr = process.communicate(timeout=300)  # 设置超时时间为5分钟
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise TimeoutError("同步操作超时")
        
        return_code = process.returncode
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command, stdout, stderr)
            
        # 打印WinSCP输出信息
        if stdout:
            logging.info(f"WinSCP标准输出:\n{stdout}")
            
        logging.info("同步任务执行完成")
        return stdout
        
    except subprocess.CalledProcessError as e:
        logging.error(f"同步过程中发生错误 (返回代码: {e.returncode}):")
        if e.stdout:
            logging.error(f"标准输出:\n{e.stdout}")
        if e.stderr:
            logging.error(f"错误输出:\n{e.stderr}")
            
            # 提取可能的 WinSCP 错误信息
            if e.stderr:
                for line in e.stderr.splitlines():
                    if "Error" in line or "Authentication failed" in line:
                        logging.error(f"关键错误: {line}")
                        
        raise
    except Exception as e:
        logging.error(f"执行WinSCP时发生意外错误: {str(e)}")
        raise

def cleanup(script_path):
    """清理临时文件"""
    try:
        if os.path.exists(script_path):
            os.remove(script_path)
            logging.info(f"已删除临时脚本: {script_path}")
    except Exception as e:
        logging.error(f"清理临时文件时出错: {str(e)}")

def Synchronize():
    # 配置信息
    config = {
        "session_name": "UbuntuSync",
        "host": "10.254.254.1",
        "username": "jpkuser",
        "password": "jpkjpk",
        "local_path": r"C:/Users/qixia/Desktop/RUNJPK",
        "remote_path": r"/home/jpkuser/Desktop/RUNJPK"
    }

    try:
        script_path = generate_script(**config)
    except Exception as e:
        print(f"生成脚本失败: {str(e)}")
        return

    try:
        # 执行同步
        output = run_winscp(script_path)
        print("同步成功!")
        if output:
            print(output)
    except Exception as e:
        print(f"同步失败: {str(e)}")
        print(f"详细信息请查看日志文件: {os.path.abspath('sync.log')}")
    finally:
        cleanup(script_path)  # 无论成功与否都清理临时文件
#---------------------------------------------------#
#   template matching
#---------------------------------------------------#
def template_matching_and_save_center(img):
            template_image_path = "template/tp15.jpg"
            csv_path = 'center_coordinates.csv'
            src_image = Image.open(img)
            src_image = np.array(src_image)
            template_image = Image.open(template_image_path)
            template_image = np.array(template_image)
            g_nMatchMethod = 5  
            resultImage_rows = src_image.shape[0] - template_image.shape[0] + 1
            resultImage_cols = src_image.shape[1] - template_image.shape[1] + 1
            g_resultImage = np.zeros((resultImage_rows, resultImage_cols, 3), np.uint8)
            g_resultImage_gray = cv2.matchTemplate(src_image[:, :, 0], template_image[:, :, 0], g_nMatchMethod)
            cv2.normalize(g_resultImage_gray, g_resultImage_gray, 0, 1, cv2.NORM_MINMAX)
            minValue, maxValue, minLocation, maxLocation = cv2.minMaxLoc(g_resultImage_gray)
            matchLocation = maxLocation  
            center_x = matchLocation[0] + template_image.shape[1] // 2
            center_y = matchLocation[1] + template_image.shape[0] // 2
            with open(csv_path, 'w') as f:
                pass
            data = {'x': [center_x], 'y': [center_y]}
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
            result_image = Image.fromarray(src_image)
            return result_image   
        
#---------------------------------------------------#
#  寻找最密的点
#---------------------------------------------------#          
def find_nearest_center():
    # 阶段1：读取并过滤原始数据
    points = []
    with open(input_file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                x = float(row[0])
                y = float(row[1])
                if (abs(x) > 4.9 and abs(y) > 4.9) and y < 0:  # 注意这里的逻辑运算符
                    points.append((x, y))
            except (IndexError, ValueError):
                continue

    # 检查数据量是否足够
    if len(points) < 1:  # 修改为至少需要1个点
        print("错误：至少需要一个符合条件的坐标点")
        
        return None

    # 阶段2：遍历所有点，计算以每个点为中心10*10矩形框内的点数
    max_point_count = 0
    best_point = None
    for center_point in points:
        point_count = 0
        for point in points:
            if (center_point[0] - 3 <= point[0] <= center_point[0] + 3) and (
                    center_point[1] - 6 <= point[1] <= center_point[1] + 6):
                point_count += 1

        if point_count > max_point_count:
            max_point_count = point_count
            best_point = center_point

    return (
        round(best_point[0], 2),
        round(best_point[1], 2)
    )
#---------------------------------------------------#
#  画图
#---------------------------------------------------#   
def draw_points_on_image(image, csv_file_paths, point_color=(0, 125, 0), point_size=50):
    """
    在图像上绘制 CSV 文件中指定的点
    :param image: 要绘制点的图像（numpy 数组）
    :param csv_file_paths: 包含点坐标的 CSV 文件路径列表
    :param point_color: 点的颜色，默认为绿色
    :param point_size: 点的大小，默认为 5
    :return: 绘制了点的图像
    """
    for csv_file_path in csv_file_paths:
        try:
            with open(csv_file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) == 2:
                        try:
                            x = int(float(row[0]))
                            y = int(float(row[1]))
                            # 检查坐标是否在图像范围内
                            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                                cv2.circle(image, (x,y), point_size, point_color, -1)
                        except ValueError:
                            print(f"无效的坐标值: {row}")
        except FileNotFoundError:
            print(f"错误: 未找到文件 {csv_file_path}")
        except Exception as e:
            print(f"发生未知错误: {e}")
    return image
#---------------------------------------------------#
# TIF格式图片转JPG
#---------------------------------------------------#    
def tif_to_jpg(input_path, output_path):
    try:
        # 打开 TIF 图片
        image = Image.open(input_path)
        # 转换为 RGB 模式，因为 JPG 不支持某些 TIF 可能包含的模式
        rgb_image = image.convert('RGB')
        # 保存为 JPG 格式
        rgb_image.save(output_path, 'JPEG')
        print(f"成功将 {input_path} 转换为 {output_path}")
    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        
#---------------------------------------------------#
# 清空文件夹
#---------------------------------------------------#         
def find_nearest(current, points):
    """找到距离当前点最近的下一个点"""
    if not points:
        return None
    nearest = min(points, key=lambda p: math.hypot(p[0]-current[0], p[1]-current[1]))
    return nearest

def clear_folder(folder_path):
    """清空指定文件夹中的所有文件和子文件夹"""
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return
    
    # 遍历文件夹中的所有内容
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)  # 删除文件
                print(f"已删除文件: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # 递归删除子文件夹
                print(f"已删除文件夹: {item_path}")
        except Exception as e:
            print(f"无法删除 {item_path}: {e}")
#---------------------------------------------------#
# 删除某个文件
#---------------------------------------------------#  
def delete_file(file_path):
    """删除指定路径的文件，并处理可能的错误"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"成功删除文件: {file_path}")
        else:
            print(f"文件不存在，无法删除: {file_path}")
    except PermissionError:
        print(f"权限错误: 无法删除文件 {file_path}，可能被其他程序占用")
    except Exception as e:
        print(f"发生未知错误: {e}，无法删除文件 {file_path}")

#---------------------------------------------------#
#  main function
#---------------------------------------------------#    

 
if __name__ == "__main__":
    yolo = YOLO()
    crop= True
    count= False     
input_file = 'coordinates.csv'
output_file = 'C:/Users/qixia/Desktop/RUNJPK/output.csv' 
output_file2 = 'C:/Users/qixia/Desktop/RUNJPK/output2.csv'  
test_file = 'C:/Users/qixia/Desktop/RUNJPK/test.csv'   
# 定义图片文件的扩展名
image_extensions = ('.jpg', '.jpeg', '.png', '.gif','.tif')       
# 假设 U 盘挂载在 'E:/'，你需要根据实际情况修改
usb_drive_path = 'C:/Users/qixia/Desktop/RUNJPK'
input_tif_file = 'C:/Users/qixia/Desktop/RUNJPK/1.tif'
test_tif = 'C:/Users/qixia/Desktop/RUNJPK/2.tif'
# 输出 JPG 文件路径
output_jpg_file = 'C:/Users/qixia/Desktop/RUNJPK/1.jpg'    
while True:
        clear_folder(usb_drive_path)
        print(f"文件夹已清空: {usb_drive_path}")
        Synchronize() 
        print(f"开始检测: {usb_drive_path}")
        while not os.path.exists(test_tif):
                print("未检测到测试图片，等待中...")
                time.sleep(5)  # 每 5 秒检查一次
                delete_file(output_file)
                delete_file(output_file2) 
                Synchronize() 
                print(f"开始检测: {usb_drive_path}")
        try: 
            delete_file(output_file)
            delete_file(output_file2)
            img = Image.open(input_tif_file)          
            img.verify()
            img.close()
            is_corrupted = False
        except (IOError, SyntaxError):
            delete_file(output_file)
            delete_file(output_file2)
            is_corrupted = True
        if is_corrupted:
            print("检测到图片损坏。")
            with open(output_file, 'w') as f:
                f.write("0,0")
            with open(output_file2, 'w') as f:
                f.write("0,0")   
            with open(test_file, 'w') as f:
                pass  # 不写入任何内容
            print("test_CSV文件已生成!")  
        else:        
            with open("coordinates.csv", "w") as f:
                    pass  # 打开文件后不做任何写入操作
            image_path = os.path.join(usb_drive_path, output_jpg_file)  
            tif_to_jpg(input_tif_file, output_jpg_file)           
            result_image = template_matching_and_save_center(image_path)
            r_image = yolo.detect_image(result_image, crop = crop, count = count)  
            ############             将(x,y)小于5的坐标写入到output.csv文件中        ################
            try:              
                # 读取所有坐标点
                coords = []
                with open(input_file, 'r', encoding='utf-8', newline='') as infile:
                    reader = csv.reader(infile)
                    for row in reader:
                        try:
                            x = float(row[0])
                            y = float(row[1])
                            if abs(x)  < 5 and abs(y)  <  5:
                                coords.append((x, y))
                        except (IndexError, ValueError):
                            continue
                
                if not coords:
                    x = float(0)
                    y = float(0)
                    coords.append((x, y))
                    print("没有符合条件的坐标点！")

                # 贪心算法排序（最近邻策略）
                path = []
                unvisited = set(coords)
                
                # 从距离原点最近的点开始
                start = min(coords, key=lambda p: math.hypot(p[0], p[1]))
                path.append(start)
                unvisited.remove(start)
                
                # 依次寻找最近点
                current = start
                while unvisited:
                    next_point = find_nearest(current, unvisited)
                    path.append(next_point)
                    unvisited.remove(next_point)
                    current = next_point
                
                # 写入排序后的坐标
                with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
                    writer = csv.writer(outfile)
                    for x, y in path:
                        writer.writerow([x, y])
                        print(f"目标点 ({x}, {y}) 已保存到 {output_file}")

            except FileNotFoundError:
                print(f"错误：未找到文件 {input_file}。")
            except Exception as e:
                print(f"发生未知错误：{e}")
                        
            ###################### 将寻找最近的两个点   #################################
            result = find_nearest_center()                
            # 写入CSV文件
            with open(output_file2, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(result)
            print(f"最近中点坐标 {result} 已保存到 {output_file2}")

            ###################### 寻找最近的两个点   #################################              
            print(r_image)
            image_obj = r_image[0]  
            # 将 PIL 图像转换为 numpy 数组
            image_np = np.array(image_obj)
            # 如果图像是 RGB 模式，需要转换为 BGR 模式，因为 OpenCV 使用 BGR
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
            # 打开 CSV 文件
            #csv_file_paths = ['output.csv', 'output2.csv']

            # 在图像上绘制点
            #result_image = draw_points_on_image(image_np, csv_file_paths)

            # 保存图像为 output.jpg
            cv2.imwrite('output.jpg', image_np)
            print("图像已成功保存为 output.jpg")
            # 遍历指定路径下的所有文件和文件夹
            with open(test_file, 'w') as f:
                pass  # 不写入任何内容
            print("test_CSV文件已生成!")  
            for root, dirs, files in os.walk(usb_drive_path):
                for file in files:
                    # 检查文件扩展名是否为图片扩展名
                    if file.lower().endswith(image_extensions):
                        file_path = os.path.join(root, file)
                        # 删除图片文件
                        os.remove(file_path)
                        print(f"已删除文件: {file_path}")  
            Synchronize() 
            wait_time = 1
            print(f"同步完成，等待 {wait_time} 秒")
            time.sleep(wait_time)