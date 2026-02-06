from alicia_d_sdk import create_robot

# 创建机器人实例（自动搜索串口）
robot = create_robot()

# 连接机械臂
if robot.connect():
    print("Connection successful!")
    
    # 打印当前状态
    robot.print_state()
    
    # 移动到初始位置
    robot.set_home()
    
    # 断开连接
    robot.disconnect()
else:
    print("Connection failed, please check serial port")