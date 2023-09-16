import json
import re

from tqdm import tqdm


RE_NAME = re.compile(r"(\[[a-zA-Z]+\])")


def remove_name(text):
    return RE_NAME.sub("", text)


def extract_data_for_classification(original_json_path: str, target_json_path: str, threshold: int=6) -> None:
    """
        Extract data from the original source, and save data for classification model.
    :param original_json_path: the file path from where the function extracts data for classification
    :param target_json_path: the target json file to save the information
    :return:
    """

    with open(original_json_path, "r", encoding="utf-8", errors="ignore") as file:
        data = json.load(file)

    res_data = []
    for conv in tqdm(data, desc="Extracting data for classification."):
        idx = conv["id"]
        for i, item in enumerate(conv["conversations"]):
            if item["from"] == "character":
                res_item = {"text": "",
                            "id": idx,
                            "class": int(item["information"] == "send_picture")}
                text = ""
                for j in range(max(0, i - threshold), i + 1):
                    text += f" [{conv['conversations'][j]['from']}] " + remove_name(conv["conversations"][j]["value"])

                res_item["text"] = text
                res_data.append(res_item)

    with open(target_json_path, "w", encoding="utf-8", errors="ignore") as f:
        json.dump(res_data, f)

    print("Extracted data for classification has saved.")


if __name__ == '__main__':
    extract_data_for_classification("data/dialogs_pics.json",
                                    "data/classification_data.json")