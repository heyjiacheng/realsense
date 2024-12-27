from realsense import MultiRealsense
import time
from multiprocessing.managers import SharedMemoryManager


def main():
    realsenses = MultiRealsense(enable_depth=True, verbose=False)
    realsenses.start()
    data = realsenses.allocate_empty()

    while True:
        try:
            realsenses.get(out=data)
            for key, value in data.items():
                print(key, value["step_idx"])
            time.sleep(1)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()


