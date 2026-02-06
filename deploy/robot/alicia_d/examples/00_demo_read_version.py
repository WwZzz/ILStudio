# Copyright (c) 2025 Synria Robotics Co., Ltd.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""
Demo: Read robot firmware version
"""

import alicia_d_sdk
from alicia_d_sdk.utils.logger import logger

def main(args):
    """Read and print robot firmware version.

    :param args: Command line arguments containing port
    """
    # Initialize robot instance
    robot = alicia_d_sdk.create_robot(port=args.port)

    try:
        robot_version = robot.get_robot_state("version")
        if robot_version:
            logger.info(
                "Version info: "
                f"Unique ID = {robot_version.get('serial_number')}, "
                f"Hardware Version = {robot_version.get('hardware_version')}, "
                f"Firmware Version = {robot_version.get('firmware_version')}"
            )
        
        gripper_type = robot.get_robot_state("gripper_type")
        if gripper_type:
            logger.info(f"Gripper type: {gripper_type}")

    except KeyboardInterrupt:
        logger.info("\nOperation interrupted by user")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

    finally:
        robot.disconnect()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Read robot firmware version")

    # Robot configuration
    parser.add_argument('--port', type=str, default="", help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    args = parser.parse_args()

    main(args)