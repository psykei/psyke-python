from psyke.gui import main
from screeninfo import get_monitors

if __name__ == '__main__':
    monitor = get_monitors()[0]
    main(int(monitor.width * .95), int(monitor.height * .9), True)
