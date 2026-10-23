
import serial
import time
import pygame

SERIAL_PORT = "COM3"
BAUD_RATE = 115200


def main():
    pygame.init()

    if pygame.joystick.get_count() == 0:
        print("No USB controller detected.")
        pygame.quit()
        return

    controller = pygame.joystick.Joystick(0)

    print(f"Controller: {controller.get_name()}")

    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)

    try:
        while True:
            command = input(
                "\nEnter Arduino command (1-4), or q to quit: "
            ).strip()

            if command.lower() == "q":
                break

            if command not in ("1", "2", "3", "4"):
                print("Invalid command.")
                continue

            # Clear pending events before sending the command
            pygame.event.clear()

            arduino.write(command.encode("ascii"))
            arduino.flush()
            send_time = time.perf_counter()

            print("Command sent. Press a controller button or D-pad...")

            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

                    if event.type == pygame.JOYBUTTONDOWN:
                        if event.joy == controller.get_instance_id():
                            detected = f"Button {event.button}"

                    elif event.type == pygame.JOYAXISMOTION:
                        if event.joy != controller.get_instance_id():
                            continue

                        # D-pad horizontal: Axis 0
                        if event.axis == 0:
                            if event.value < -0.5:
                                detected = "D-pad LEFT"
                            elif event.value > 0.5:
                                detected = "D-pad RIGHT"

                        # D-pad vertical: Axis 1
                        elif event.axis == 1:
                            if event.value < -0.5:
                                detected = "D-pad UP"
                            elif event.value > 0.5:
                                detected = "D-pad DOWN"

                    if detected:
                        receive_time = time.perf_counter()
                        latency_ms = (receive_time - send_time) * 1000

                        print(f"Input: {detected}")
                        print(f"Latency: {latency_ms:.3f} ms")
                        break
                else:
                    time.sleep(0.001)
                    continue

                break


    except KeyboardInterrupt:
        print("\nExiting.")

    finally:
        arduino.close()
        pygame.quit()


if __name__ == "__main__":
    main()
