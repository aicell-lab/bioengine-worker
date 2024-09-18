
from scaler import Scaler
import init

if __name__ == "__main__":
    if init.setup():
        Scaler().run()
        init.shutdown()

