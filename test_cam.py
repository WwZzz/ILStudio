import cv2
import time

# --- 参数设置 ---
CAMERA_INDEX = 1         # 相机索引，0 通常是默认摄像头
TARGET_FPS = 30          # 目标帧率
RESOLUTION_W = 640      # 分辨率宽度
RESOLUTION_H = 480       # 分辨率高度

# 目标单帧时间 (秒)
FRAME_TIME = 1.0 / TARGET_FPS 

# --- 初始化相机 ---
# 尝试使用 MSMF 或 DSHOW 后端，它们通常提供更好的参数控制
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW) 

if not cap.isOpened():
    print("错误：无法打开相机")
    exit()

# --- 设置相机参数 ---
# 1. 设置期望的帧率
if cap.set(cv2.CAP_PROP_FPS, TARGET_FPS):
    print(f"成功请求帧率: {TARGET_FPS} FPS")
else:
    print(f"警告：无法设置期望的帧率: {TARGET_FPS} FPS")

# 2. 设置分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_H)

# 3. 关闭自动曝光，切换到手动模式
#    对于V4L2 (Linux) 通常是设置为 1
#    对于DSHOW (Windows) 通常是设置为 0
if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0): 
    print("成功关闭自动曝光")
else:
    print("警告：无法关闭自动曝光")

# 4. 设置一个固定的曝光时间
#    这个值的单位是 100 微秒的对数。-4 约等于 10ms, -5 约 20ms, -6 约 40ms
#    为了达到 30 FPS (33.33ms/帧), 曝光时间应小于33.33ms。我们设为 -5 (20ms)
exposure_val = -5.0
if cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val):
    print(f"成功设置曝光时间为: {exposure_val} (约 20ms)")
else:
    print("警告：无法设置曝光时间")

# 验证设置是否成功
print("-" * 30)
print(f"实际帧率: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"实际分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"实际曝光值: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
print("-" * 30)


# --- 帧率测试 ---
prev_time = 0
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取帧失败，退出...")
        break

    # 计算并显示实时帧率
    current_time = time.time()
    elapsed_time = current_time - start_time
    frame_count += 1

    if elapsed_time > 1.0: # 每秒更新一次
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Reset
        start_time = current_time
        frame_count = 0

    cv2.imshow('Stable FPS Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 释放资源 ---
cap.release()
cv2.destroyAllWindows()