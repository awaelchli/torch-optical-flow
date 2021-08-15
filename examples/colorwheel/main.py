import os
import optical_flow

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    optical_flow.colorwheel(method="baker", size=512, file="./outputs/baker.png")
    optical_flow.colorwheel(method="hsv", size=512, file="./outputs/hsv.png")
    optical_flow.colorwheel(method="meister", size=512, file="./outputs/meister.png")
