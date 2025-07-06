import cv2
import numpy as np
from PIL import Image
import pandas as pd
from yolo import YOLO
import csv
import math
import time
import os
import subprocess
import logging
import shutil
from typing import List, Tuple, Optional, Dict

#-------------------------------------#
#       Synchronization
#-------------------------------------#
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='sync.log'
)

def generate_script(session_name: str, host: str, username: str, password: str, 
                   local_path: str, remote_path: str) -> str:
    """Generate WinSCP script"""
    # Verify local path existence
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local path does not exist: {local_path}")
    
    # Proper way to disable host key verification
    script = f"""# Sync script - accept any host key
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
    
    logging.info(f"Script saved to: {os.path.abspath(script_path)}")
    return script_path

def run_winscp(script_path: str, winscp_path: str = r'C:/Program Files (x86)/WinSCP/WinSCP.exe') -> str:
    """Execute WinSCP script"""
    try:
        # Verify script file existence
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script file does not exist: {script_path}")

        # Check WinSCP executable existence
        if not os.path.exists(winscp_path):
            raise FileNotFoundError(f"WinSCP executable does not exist: {winscp_path}")

        logging.info("Starting synchronization task")
        
        # Build and execute command
        command = [
            winscp_path, 
            '/script=' + script_path, 
            '/log=sync.log', 
            '/loglevel=2',
            '/noverifycert'  # Alternative: disable certificate verification
        ]
        logging.info(f"Executing command: {' '.join(command)}")

        # Use communicate with timeout
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            stdout, stderr = process.communicate(timeout=300)  # 5-minute timeout
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise TimeoutError("Synchronization operation timed out")
        
        return_code = process.returncode
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command, stdout, stderr)
            
        # Log WinSCP output
        if stdout:
            logging.info(f"WinSCP standard output:\n{stdout}")
            
        logging.info("Synchronization task completed")
        return stdout
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during synchronization (return code: {e.returncode}):")
        if e.stdout:
            logging.error(f"Standard output:\n{e.stdout}")
        if e.stderr:
            logging.error(f"Error output:\n{e.stderr}")
            
            # Extract potential WinSCP error messages
            if e.stderr:
                for line in e.stderr.splitlines():
                    if "Error" in line or "Authentication failed" in line:
                        logging.error(f"Critical error: {line}")
                        
        raise
    except Exception as e:
        logging.error(f"Unexpected error executing WinSCP: {str(e)}")
        raise

def cleanup(script_path: str) -> None:
    """Clean up temporary files"""
    try:
        if os.path.exists(script_path):
            os.remove(script_path)
            logging.info(f"Temporary script deleted: {script_path}")
    except Exception as e:
        logging.error(f"Error cleaning up temporary file: {str(e)}")

def synchronize() -> None:
    """Perform file synchronization"""
    # Configuration
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
        print(f"Failed to generate script: {str(e)}")
        return

    try:
        # Execute synchronization
        output = run_winscp(script_path)
        print("Synchronization successful!")
        if output:
            print(output)
    except Exception as e:
        print(f"Synchronization failed: {str(e)}")
        print(f"Detailed information in log file: {os.path.abspath('sync.log')}")
    finally:
        cleanup(script_path)  # Clean up temp file regardless of success/failure

#-------------------------------------#
#       Template Matching
#-------------------------------------#
def template_matching_and_save_center(img_path: str) -> Image.Image:
    """Perform template matching and save center coordinates"""
    template_image_path = "template/tp15.jpg"
    csv_path = 'center_coordinates.csv'
    
    # Read source image
    src_image = Image.open(img_path)
    src_np = np.array(src_image)
    
    # Read template image
    template_image = Image.open(template_image_path)
    template_np = np.array(template_image)
    
    match_method = 5  # CV_TM_CCOEFF_NORMED
    result_rows = src_np.shape[0] - template_np.shape[0] + 1
    result_cols = src_np.shape[1] - template_np.shape[1] + 1
    
    # Perform matching on grayscale
    gray_src = src_np[:, :, 0]
    gray_template = template_np[:, :, 0]
    result_gray = cv2.matchTemplate(gray_src, gray_template, match_method)
    cv2.normalize(result_gray, result_gray, 0, 1, cv2.NORM_MINMAX)
    
    # Find best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_gray)
    match_loc = max_loc  # Use max for TM_CCOEFF_NORMED
    
    # Calculate center
    center_x = match_loc[0] + template_np.shape[1] // 2
    center_y = match_loc[1] + template_np.shape[0] // 2
    
    # Save to CSV
    data = {'x': [center_x], 'y': [center_y]}
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    return src_image

#-------------------------------------#
#       Find Densest Point
#-------------------------------------#
def find_nearest_center(input_file: str) -> Optional[Tuple[float, float]]:
    """Find the center of the densest point cluster"""
    points = []
    with open(input_file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                x = float(row[0])
                y = float(row[1])
                if abs(x) > 4.9 and abs(y) > 4.9 and y < 0:
                    points.append((x, y))
            except (IndexError, ValueError):
                continue

    # Check data sufficiency
    if len(points) < 1:
        print("Error: At least one valid coordinate point is required")
        return None

    # Find densest point
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
    ) if best_point else None

#-------------------------------------#
#       Draw Points on Image
#-------------------------------------#
def draw_points_on_image(image: np.ndarray, csv_file_paths: List[str], 
                        point_color: Tuple[int, int, int] = (0, 125, 0), 
                        point_size: int = 5) -> np.ndarray:
    """Draw points from CSV files on an image"""
    for csv_file_path in csv_file_paths:
        try:
            with open(csv_file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) == 2:
                        try:
                            x = int(float(row[0]))
                            y = int(float(row[1]))
                            # Check if coordinates are within image bounds
                            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                                cv2.circle(image, (x, y), point_size, point_color, -1)
                        except ValueError:
                            print(f"Invalid coordinate value: {row}")
        except FileNotFoundError:
            print(f"Error: File not found {csv_file_path}")
        except Exception as e:
            print(f"Unknown error: {e}")
    return image

#-------------------------------------#
#       Convert TIF to JPG
#-------------------------------------#
def tif_to_jpg(input_path: str, output_path: str) -> bool:
    """Convert TIF image to JPG"""
    try:
        # Open TIF image
        image = Image.open(input_path)
        # Convert to RGB mode
        rgb_image = image.convert('RGB')
        # Save as JPG
        rgb_image.save(output_path, 'JPEG')
        print(f"Successfully converted {input_path} to {output_path}")
        return True
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

#-------------------------------------#
#       Utility Functions
#-------------------------------------#
def find_nearest(current: Tuple[float, float], points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Find the nearest point to the current point"""
    if not points:
        return None
    return min(points, key=lambda p: math.hypot(p[0]-current[0], p[1]-current[1]))

def clear_folder(folder_path: str) -> None:
    """Clear all files and subfolders in a folder"""
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return
    
    # Iterate through all items
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
                # print(f"File deleted: {item_path}")  # Commented for performance
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                # print(f"Folder deleted: {item_path}")  # Commented for performance
        except Exception as e:
            print(f"Failed to delete {item_path}: {e}")

def delete_file(file_path: str) -> None:
    """Delete a file with error handling"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            # print(f"File deleted: {file_path}")  # Commented for performance
        else:
            # print(f"File not found: {file_path}")  # Commented for performance
            pass
    except PermissionError:
        print(f"Permission error: Cannot delete {file_path}, possibly in use by another program")
    except Exception as e:
        print(f"Unknown error: {e}, cannot delete {file_path}")

#-------------------------------------#
#       Main Function
#-------------------------------------#
if __name__ == "__main__":
    yolo = YOLO()
    crop = True
    count = False     

    # File paths
    input_file = 'coordinates.csv'
    output_file = 'C:/Users/qixia/Desktop/RUNJPK/output.csv' 
    output_file2 = 'C:/Users/qixia/Desktop/RUNJPK/output2.csv'  
    test_file = 'C:/Users/qixia/Desktop/RUNJPK/test.csv'   
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.tif')       
    usb_drive_path = 'C:/Users/qixia/Desktop/RUNJPK'
    input_tif_file = 'C:/Users/qixia/Desktop/RUNJPK/1.tif'
    test_tif = 'C:/Users/qixia/Desktop/RUNJPK/2.tif'
    output_jpg_file = 'C:/Users/qixia/Desktop/RUNJPK/1.jpg'    
    csv_file_paths = [output_file, output_file2]

    while True:
        clear_folder(usb_drive_path)
        print(f"Folder cleared: {usb_drive_path}")
        synchronize() 
        print(f"Start detection: {usb_drive_path}")
        
        # Wait for test image
        while not os.path.exists(test_tif):
            print("Test image not detected, waiting...")
            time.sleep(5)  # Check every 5 seconds
            delete_file(output_file)
            delete_file(output_file2) 
            synchronize() 
            print(f"Start detection: {usb_drive_path}")
        
        try: 
            delete_file(output_file)
            delete_file(output_file2)
            
            # Verify image integrity
            img = Image.open(input_tif_file)
            img.verify()
            img.close()
            is_corrupted = False
        except (IOError, SyntaxError):
            delete_file(output_file)
            delete_file(output_file2)
            is_corrupted = True
        
        if is_corrupted:
            print("Detected corrupted image.")
            with open(output_file, 'w') as f:
                f.write("0,0")
            with open(output_file2, 'w') as f:
                f.write("0,0")   
            with open(test_file, 'w') as f:
                pass  # Empty file
            print("Test CSV file generated!")  
        else:        
            # Initialize coordinates file
            with open("coordinates.csv", "w") as f:
                pass  # Empty file
            
            # Convert TIF to JPG
            if tif_to_jpg(input_tif_file, output_jpg_file):
                # Perform template matching
                result_image = template_matching_and_save_center(output_jpg_file)
                
                # Perform object detection
                r_image = yolo.detect_image(result_image, crop=crop, count=count)  
                
                ############ Write coordinates with |x|<5 and |y|<5 to output.csv ################
                try:              
                    # Read all coordinates
                    coords = []
                    with open(input_file, 'r', encoding='utf-8', newline='') as infile:
                        reader = csv.reader(infile)
                        for row in reader:
                            try:
                                x = float(row[0])
                                y = float(row[1])
                                if abs(x) < 5 and abs(y) < 5:
                                    coords.append((x, y))
                            except (IndexError, ValueError):
                                continue
                    
                    # Handle empty coordinates
                    if not coords:
                        x, y = 0.0, 0.0
                        coords.append((x, y))
                        print("No valid coordinates found!")

                    # Greedy algorithm for nearest neighbor ordering
                    path = []
                    unvisited = set(coords)
                    
                    # Start from point nearest to origin
                    start = min(coords, key=lambda p: math.hypot(p[0], p[1]))
                    path.append(start)
                    unvisited.remove(start)
                    
                    # Find nearest points sequentially
                    current = start
                    while unvisited:
                        next_point = find_nearest(current, unvisited)
                        path.append(next_point)
                        unvisited.remove(next_point)
                        current = next_point
                    
                    # Write ordered coordinates
                    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
                        writer = csv.writer(outfile)
                        for x, y in path:
                            writer.writerow([x, y])
                            # print(f"Target point ({x}, {y}) saved to {output_file}")  # Commented for performance

                except FileNotFoundError:
                    print(f"Error: File not found {input_file}.")
                except Exception as e:
                    print(f"Unknown error: {e}")
                        
            ###################### Find nearest center point #################################
            # Find nearest center point
            result = find_nearest_center(input_file)                
            # Write to CSV file
            if result:
                with open(output_file2, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(result)
                print(f"Nearest center coordinates {result} saved to {output_file2}")
            else:
                with open(output_file2, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([0, 0])
                print(f"Default coordinates (0, 0) saved to {output_file2}")

            ###################### Process and save result image #################################              
            if r_image and len(r_image) > 0:
                image_obj = r_image[0]  # Assuming r_image is a tuple containing the image
                # Convert PIL image to numpy array
                image_np = np.array(image_obj)
                # Convert RGB to BGR for OpenCV
                if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                # Draw points on image
                try:
                    result_image = draw_points_on_image(image_np, csv_file_paths)
                    # Save image as output.jpg
                    cv2.imwrite('output.jpg', result_image)
                    print("Image successfully saved as output.jpg")
                except Exception as e:
                    print(f"Error saving image: {e}")
                    # Fallback to saving original image
                    cv2.imwrite('output.jpg', image_np)
            else:
                print("No detection result image available")
                # Create a blank image as fallback
                blank_image = np.zeros((600, 800, 3), np.uint8)
                cv2.imwrite('output.jpg', blank_image)

            # Generate empty test CSV file
            with open(test_file, 'w') as f:
                pass  # Empty file
            print("Test CSV file generated!")  
            
            # Clean up image files
            for root, dirs, files in os.walk(usb_drive_path):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            # print(f"File deleted: {file_path}")  # Commented for performance
                        except Exception as e:
                            print(f"Failed to delete {file_path}: {e}")
                            
            # Perform synchronization
            synchronize() 
            wait_time = 1
            print(f"Synchronization completed, waiting {wait_time} second")
            time.sleep(wait_time)