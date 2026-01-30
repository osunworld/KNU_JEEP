import os, glob, sys
import time
import pygame
import argparse
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime


class annotator:
    def __init__(self, system, height, width, path):
        pygame.init()
        self.height = height
        self.width = width

        self.upper_line = None
        self.middle_line = None
        self.lower_line = None

        self.masked_img = None
        self.info_img = None
        self.pause = False

        self.font = ImageFont.truetype("fonts/Arial.ttf", size=20)
        self.directory = path
        self.dash_length = 5
        self.radius = 6

        self.count = 0
        self._count = 0
        self.goal_x = None
        self.goal_y = None
        self.coord_x = []
        self._coord_x = []

        if system == "Jetson":
            os.system("echo soda | sudo -S systemctl restart nvargus-daemon")
            self.speed = 0
            self.steer = 0
            self.speed_limit = 10
            self.steer_limit = 0.6

            cap_mode_table = {
                0: (3280, 2464, 21),
                1: (3280, 1848, 28),
                2: (1920, 1080, 30),
                3: (1640, 1232, 30),
                4: (1280, 720, 60),
            }
            capture_width, capture_height, fps = cap_mode_table[0]
            self.gst_str = (
                "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=%s ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink"
                % (
                    capture_width,
                    capture_height,
                    fps,
                    2,
                    self.width,
                    self.height,
                )
            )
            self.cap = cv2.VideoCapture(self.gst_str, cv2.CAP_GSTREAMER)

            try:
                from pop import driving

                self.car = driving.Driving()
            except ImportError:
                print(
                    "[INFO] This process uses Pilot module which will be deprecated in the future version."
                )
                from pop import Pilot

                self.car = Pilot.AutoCar()

            pygame.display.set_caption("Annotator (Video) ")

        else:
            print(
                "[INFO] This process is not running on ARM architecture(Jetson board)."
            )
            print("[INFO] Video mode is surpressed")
            pygame.display.set_caption("Annotator (Picture) ")

        self.screen = pygame.display.set_mode((self.width + 150, self.height))

    def display(self, image):
        image_np = np.array(image)
        pygame_image = pygame.image.frombuffer(
            image_np.tobytes(), image.size, image.mode
        )
        self.screen.blit(pygame_image, (0, 0))

        pygame.display.update()

    def mask_image(self, image):
        mask = np.zeros(image.shape, dtype=np.uint8)

        mask[self.upper_line : self.lower_line, :] = (
            mask[self.upper_line : self.lower_line, :]
            + image[self.upper_line : self.lower_line, :]
        )

        result = cv2.addWeighted(image, 0.3, mask, 1, 0)

        result = cv2.line(
            result,
            (0, self.upper_line),
            (self.width, self.upper_line),
            (0, 0, 255),
            1,
        )
        result = cv2.line(
            result,
            (0, self.lower_line),
            (self.width, self.lower_line),
            (0, 0, 255),
            1,
        )

        current_length = 0

        while current_length < self.width:
            start = (current_length, self.middle_line)
            end = ((current_length + self.dash_length), self.middle_line)
            cv2.line(result, start, end, (0, 255, 0), 1)
            current_length = current_length + (self.dash_length * 3)
        
        result = cv2.hconcat([result, self.info_img])
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(result)
        return result

    def click_event(self, event):
        draw = ImageDraw.Draw(self.masked_img)
        x, y = event.pos

        if system == "Jetson":
            self.speed = 0
            self.steer = 0
            self.car.steering = 0
            self.car.stop()

        if not self.pause:
            self.pause = True

        if event.button == 1:
            if self.upper_line < y < self.lower_line and 0 <= x <= self.width:
                if self.count < 2:
                    center = (x, self.middle_line)
                    draw.ellipse(
                        [
                            center[0] - self.radius,
                            center[1] - self.radius,
                            center[0] + self.radius,
                            center[1] + self.radius,
                        ],
                        outline="skyblue",
                    )

                    self.count += 1
                    self.coord_x.append(x)

                    if self.count == 2:
                        mean_x = int(sum(self.coord_x) / len(self.coord_x))
                        self.goal_x = int((2 * mean_x) - (self.width / 2))
                        self.goal_x = min(max(0, self.goal_x), self.width)

                        center = (self.goal_x, self.middle_line)
                        for i in range(3):
                            i = i * 0.5
                            draw.ellipse(
                                [
                                    center[0] - self.radius - i,
                                    center[1] - self.radius - i,
                                    center[0] + self.radius + i,
                                    center[1] + self.radius + i,
                                ],
                                outline="blue",
                            )

                    self.display(self.masked_img)
        elif event.button == 3:
            if self.upper_line < y < self.lower_line and 0 <= x <= self.width:
                if self._count < 2:
                    center = (x, self.middle_line)
                    draw.ellipse(
                        [
                            center[0] - self.radius,
                            center[1] - self.radius,
                            center[0] + self.radius,
                            center[1] + self.radius,
                        ],
                        outline="lightpink",
                    )
                    self._count += 1
                    self._coord_x.append(x)

                    if self._count == 2:
                        mean_x = int(sum(self._coord_x) / len(self._coord_x))
                        self.goal_y = int((2 * mean_x) - (self.width / 2))
                        self.goal_y = min(max(0, self.goal_y), self.width)

                        center = (self.goal_y, self.middle_line)
                        for i in range(3):
                            i = i * 0.5
                            draw.ellipse(
                                [
                                    center[0] - self.radius - i,
                                    center[1] - self.radius - i,
                                    center[0] + self.radius + i,
                                    center[1] + self.radius + i,
                                ],
                                outline="red",
                            )

                    self.display(self.masked_img)

    def pic_introduction(self):
        if system == "Jetson":
            print("###############Picture Mode Instrunction###############")
            print(
                "q: Previous Frame \tw: Reset Frame \te: Next Frame\na: Auto save on/off \ts: Save \td: Delete"
            )
            print("#######################################################")
        else:
            print("###############Picture Mode Instrunction###############")
            print(
                "q: Previous Frame \tw: Reset Frame \te: Next Frame\na: Auto save on/off \ts: Save \td: Delete"
            )
            print("#######################################################")

    def cam_introduction(self):
        print("################Video Mode Instrunction################")
        print(
            "a: Auto save on/off \ts: Save \td: Reset Frame\nspacebar: Switch to Picture Mode"
        )
        print("#######################################################")

    def update_command_img(self, mode):
        info_img = np.ones((self.height, 150, 3), dtype=np.uint8) * 255
        info_img = cv2.cvtColor(info_img, cv2.COLOR_BGR2RGB)
        info_img = Image.fromarray(info_img)
        font = ImageFont.truetype("fonts/Arial.ttf", size=18)
        draw = ImageDraw.Draw(info_img)

        if mode == "Picture":
            draw.text((10, 10), "Command list", fill=(0, 0, 0), font=font)

            texts = [
                "q : Prev Frame",
                "e : Next Frame",
                "r : Reset Frame",
                "a : Auto Save On/Off",
                "s : Save",
                "d : Delete Frame",
            ]
            positions = [
                (10, 50),
                (10, 75),
                (10, 100),
                (10, 125),
                (10, 150),
                (10, 175),
                (10, 200),
            ]
            font = ImageFont.truetype("fonts/Arial.ttf", size=13)

            for text, position in zip(texts, positions):
                draw.text(position, text, fill=(0, 0, 0), font=font)

            texts = [
                "1st path",
                "2nd path",
            ]
            centers = [
                (15, 240),
                (15, 270),
            ]
            colors = [
                "blue",
                "red",
            ]
            for text, center, color in zip(texts, centers, colors):
                draw.ellipse(
                    [
                        center[0] - 4,
                        center[1] - 4,
                        center[0] + 4,
                        center[1] + 4,
                    ],
                    outline=color,
                    fill=color,
                )
                draw.text((center[0] + 15, center[1] - 8), text, fill=color, font=font)
        elif mode == "Video":
            draw.text((10, 10), "Command list", fill=(0, 0, 0), font=font)

            texts = [
                "a : Auto Save On/Off",
                "s : Save",
                "r : Reset Frame",
                "space : Switch mode",
            ]
            positions = [
                (10, 50),
                (10, 75),
                (10, 100),
                (10, 125),
                (10, 150),
            ]
            font = ImageFont.truetype("fonts/Arial.ttf", size=13)

            for text, position in zip(texts, positions):
                draw.text(position, text, fill=(0, 0, 0), font=font)

        info_img = np.array(info_img)
        self.info_img = cv2.cvtColor(info_img, cv2.COLOR_RGB2BGR)

    def annotation(self):
        if system == "Jetson":
            mode = "Video"
            slash = "/"
        else:
            mode = "Picture"
            slash = "\\"

        self.update_command_img(mode)

        self.steer = 0
        self.speed = 0

        page = 0
        save = False
        auto_save = False
        save_mode = "Off"
        file_name = None
        save_release_time = 0
        speed_release_time = 0
        steer_release_time = 0

        running = True

        if mode == "Picture":
            file_list = glob.glob(f"{path}/*.jpg")
            file_list.sort(key=lambda x: os.path.getmtime(x))
            full_page = len(file_list)
            self.pic_introduction()
            print(f"[INFO] Current page: ({page + 1}/{full_page}).")
        else:
            self.cam_introduction()

        while running:
            keys = pygame.key.get_pressed()

            if not self.pause:
                if mode == "Picture":
                    try:
                        image = cv2.imread(file_list[page])
                        file_name = file_list[page]
                        if time.time() - save_release_time > 1:
                            x = file_name.split(slash)[-1].split("_")[0]
                            y = file_name.split(slash)[-1].split("_")[1]
                    except IndexError:
                        print("[INFO] There is no picture to annotate.")
                        print("[INFO] Terminate the process.")
                        break
                else:
                    _, image = self.cap.read()

                    if keys[pygame.K_UP]:
                        speed_release_time = time.time()
                        self.speed = self.speed_limit
                        self.car.forward(self.speed)
                    elif keys[pygame.K_DOWN]:
                        speed_release_time = time.time()
                        self.speed = -self.speed_limit
                        self.car.backward(self.speed)
                    else:
                        if time.time() - speed_release_time > 0.1:
                            self.speed = 0
                            self.car.stop()

                    if keys[pygame.K_RIGHT]:
                        steer_release_time = time.time()
                        self.steer += 0.02
                        self.steer = min(self.steer_limit, self.steer)
                        self.car.steering = self.steer
                    elif keys[pygame.K_LEFT]:
                        steer_release_time = time.time()
                        self.steer -= 0.02
                        self.steer = max(-self.steer_limit, self.steer)
                        self.car.steering = self.steer
                    else:
                        if time.time() - steer_release_time > 0.5:
                            self.steer = 0
                            self.car.steering = self.steer

                self.coord_x.clear()
                self._coord_x.clear()
                self.count = 0
                self._count = 0

                self.upper_line = int(0.5 * self.height) - 20
                self.middle_line = int(self.height * 0.8)
                self.lower_line = self.height - 20

                if time.time() - save_release_time > 1:
                    self.masked_img = self.mask_image(image)

                draw = ImageDraw.Draw(self.masked_img)
                try:
                    centers = [(int(y), self.middle_line), (int(x), self.middle_line)]
                    colors = ["red", "blue"]

                    for center, color in zip(centers, colors):
                        draw.ellipse(
                            [
                                center[0] - 5,
                                center[1] - 5,
                                center[0] + 5,
                                center[1] + 5,
                            ],
                            outline=color,
                            fill=color,
                        )
                except:
                    pass

            info_image = self.masked_img.copy()
            draw = ImageDraw.Draw(info_image)

            if mode == "Picture":
                texts = [
                    f"Auto Save: {save_mode}",
                    f"Page: ({page + 1} / {full_page})",
                ]
                positions = [(10, 10), (10, 45)]

                for text, position in zip(texts, positions):
                    draw.text(position, text, fill=(255, 0, 0), font=self.font)
            elif mode == "Video":
                texts = [
                    f"Auto Save: {save_mode}",
                    f"Speed: {self.speed}",
                    f"Steering: {self.steer:.2f}",
                ]
                positions = [(10, 10), (10, 45), (10, 80)]

                for text, position in zip(texts, positions):
                    draw.text(position, text, fill=(255, 0, 0), font=self.font)

            for event in pygame.event.get():
                if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
                    if system == "Jetson":
                        self.car.stop()
                        self.car.steering = 0
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if time.time() - save_release_time > 1:
                        self.click_event(event)

                if mode == "Picture":
                    if event.type == pygame.KEYUP:
                        if keys[pygame.K_a]:
                            if auto_save:
                                print("[INFO] Auto save is off.")
                                auto_save = False
                                save_mode = "Off"
                            else:
                                print("[INFO] Auto save is on.")
                                auto_save = True
                                save_mode = "On"
                            break
                        elif keys[pygame.K_s]:
                            if self.count >= 2 or self._count >= 2:
                                save = True
                            else:
                                print("[INFO] Set two points before you save.")
                            break
                        elif keys[pygame.K_d]:
                            os.remove(file_list[page])
                            print(f"[INFO] {file_list[page]} is deleted.")
                            file_list = glob.glob(f"{path}/*.jpg")
                            file_list.sort(key=lambda x: os.path.getmtime(x))
                            full_page = len(file_list)
                            if full_page == page:
                                page -= 1
                            print(f"[INFO] Current page: ({page + 1}/{full_page}).")
                            break
                        elif keys[pygame.K_SPACE] and system == "Jetson":
                            page = 0
                            self.pause = False
                            mode = "Video"
                            self.update_command_img(mode)
                            pygame.display.set_caption("Annotator (Video) ")
                            self.cam_introduction()
                            del x, y
                            break

                    if keys[pygame.K_q]:
                        self.pause = False
                        if page > 0:
                            page -= 1
                            print(f"[INFO] Current page: ({page + 1}/{full_page}).")
                        else:
                            page = full_page - 1
                            print(f"[INFO] Current page: ({page + 1}/{full_page}).")
                        break
                    elif keys[pygame.K_e]:
                        self.pause = False
                        if page < (full_page - 1):
                            page += 1
                            print(f"[INFO] Current page: ({page + 1}/{full_page}).")
                        else:
                            page = 0
                            print(f"[INFO] Current page: ({page + 1}/{full_page}).")
                        break
                    elif keys[pygame.K_r]:
                        self.pause = False
                        break

                elif mode == "Video":
                    if event.type == pygame.KEYUP:
                        if keys[pygame.K_a]:
                            if auto_save:
                                print("[INFO] Auto save is off.")
                                auto_save = False
                                save_mode = "Off"
                            else:
                                print("[INFO] Auto save is on.")
                                auto_save = True
                                save_mode = "On"
                            break
                        elif keys[pygame.K_s]:
                            if self.count >= 2:
                                save = True
                            else:
                                print("[INFO] Set two points before you save.")
                            break
                        elif keys[pygame.K_r]:
                            self.pause = False
                            break
                        elif keys[pygame.K_SPACE]:
                            page = 0
                            self.pause = False
                            mode = "Picture"
                            self.update_command_img(mode)
                            pygame.display.set_caption("Annotator (Picture) ")
                            file_list = glob.glob(f"{path}/*.jpg")
                            file_list.sort(key=lambda x: os.path.getmtime(x))
                            full_page = len(file_list)
                            self.pic_introduction()
                            print(f"[INFO] Current page is ({page + 1}/{full_page}).")
                            break

            if auto_save:
                if self.count >= 2:
                    save = True
                elif self._count >= 2:
                    if mode == "Picture":
                        save = True

            if save:
                self.pause = False
                save = False
                page += 1

                if mode == "Picture":
                    if page >= full_page:
                        page = 0
                    print(f"[INFO] Current page: ({page + 1}/{full_page}).")

                    if self._count == 2:
                        if self.count == 2:
                            new_filename = (
                                str(self.goal_x) + "_" + str(self.goal_y) + "_"
                            )
                        else:
                            new_filename = str(x) + "_" + str(self.goal_y) + "_"
                    else:
                        new_filename = str(self.goal_x) + "_" + str(self.goal_x) + "_"

                    # new_filename = new_filename + "_".join(
                    #     file_name.split(slash)[-1].split("_")[2:] 
                    # ) + str(datetime.now()) + '.jpg'
                    new_filename += str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.jpg'
                    destination_file = os.path.join(self.directory, new_filename)
                    os.rename(file_name, destination_file)
                    file_list = glob.glob(f"{path}/*.jpg")
                    file_list.sort(key=lambda x: os.path.getmtime(x))
                elif mode == "Video":
                    print(f"[INFO] Collected {page} page(s).")
                    timestamp = datetime.now()
                    time_line = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

                    if self._count == 2:
                        new_filename = (
                            str(self.goal_x)
                            + "_"
                            + str(self.goal_y)
                            + "_"
                            + time_line
                            + ".jpg"
                        )
                    else:
                        new_filename = (
                            str(self.goal_x)
                            + "_"
                            + str(self.goal_x)
                            + "_"
                            + time_line
                            + ".jpg"
                        )

                    print(new_filename)
                    destination_file = os.path.join(self.directory, new_filename)
                    os.makedirs(os.path.dirname(destination_file), exist_ok=True)
                    cv2.imwrite(destination_file, image)

                if auto_save:
                    save_release_time = time.time()

            self.display(info_image)
            time.sleep(0.01)

        pygame.quit()
        sys.exit()


# def valid_mode(mode):
#     if mode not in ["Picture", "Camera"]:
#         raise argparse.ArgumentTypeError(
#             f"Invalid mode: {mode}. mode must be either 'Picture' or 'Camera'."
#         )
#     return mode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set mode for Data Frame.")

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="",
        help="Please set the path for the ouput directory.",
    )

    args = parser.parse_args()

    path = args.path
    width = 400
    height = 274
    # mode = args.mode

    release_file = "/etc/nv_tegra_release"
    system = "Not-Jetson"

    if path != "":
        system = "Not-Jetson"
        pass
    elif os.path.exists(release_file):
        try:
            with open(release_file, "r") as file:
                file_contents = file.read()
                info_pairs = file_contents.split(",")

                for pair in info_pairs:
                    keys = pair.split(": ")
                    for key in keys:
                        if key == "aarch64":
                            system = "Jetson"
                            path = "annotated_dataset/"
        except PermissionError:
            print(f"Permission denied to read file '{release_file}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        system = "Not-Jetson"
        path = "track_dataset/"

    annot = annotator(system, height, width, path)
    annot.annotation()
