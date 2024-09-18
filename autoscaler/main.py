
from scaler import Scaler
import init

if __name__ == "__main__":
    init.setup()
    autoscaler = Scaler()
    autoscaler.run()
    init.shutdown()

