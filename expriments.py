import os


def runs(base_cmd: str):
    os.system(f"{base_cmd}")


if __name__ == "__main__":
    runs("python main.py -l -ds \"PascalVOC2012\" -m \"DEBUG\" -md \"ResNet152\"")
    runs("python main.py -l -pt -ds \"PascalVOC2012\" -m \"DEBUG\" -md \"ResNet152\"")

    runs("python main.py -l -ds \"PascalVOC2012\" -m \"DEBUG\" -md \"ResNet101\"")
    runs("python main.py -l -pt -ds \"PascalVOC2012\" -m \"DEBUG\" -md \"ResNet101\"")

    runs("python main.py -l -ds \"PascalVOC2012\" -m \"DEBUG\" -md \"ResNet50\"")
    runs("python main.py -l -pt -ds \"PascalVOC2012\" -m \"DEBUG\" -md \"ResNet50\"")

    runs("python main.py -l -ds \"PascalVOC2012\" -m \"DEBUG\" -md \"ResNet34\"")
    runs("python main.py -l -pt -ds \"PascalVOC2012\" -m \"DEBUG\" -md \"ResNet34\"")
