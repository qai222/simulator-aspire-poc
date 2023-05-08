from casymda.bpmn.bpmn_parser import parse_bpmn
from pandas._typing import FilePath


def generate_model(
        bpmn: FilePath, json_path: FilePath, temp_path: FilePath, model_output: FilePath
):
    parse_bpmn(bpmn, json_path, temp_path, model_output)


if __name__ == '__main__':
    generate_model(
        "../assets/two_heater.bpmn",
        "temp.json",
        "../assets/model_template.py",
        "../model/model_two_heater.py"
    )
