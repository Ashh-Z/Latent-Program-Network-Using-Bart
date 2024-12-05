import json
from torch.utils.data import Dataset

paths = {
    "train": {
        "chlng": "arc-prize-2024/arc-agi_training_challenges.json",
        "soln": "arc-prize-2024/arc-agi_training_solutions.json",
    },
    "eval": {
        "chlng": "arc-prize-2024/arc-agi_evaluation_challenges.json",
        "soln": "arc-prize-2024/arc-agi_evaluation_solutions.json",
    },
}


class MetaData:
    def __init__(self, data, solutions=None):
        data, solutions = self.tuple_data(data, solutions)
        self.data = data
        self.solutions = solutions

    def tuple_data(self, data, solutions):
        ast = lambda g: tuple(tuple(r) for r in g)
        for idx in range(len(data)):
            data[idx]["input"] = ast(data[idx]["input"])
            if solutions is None:
                data[idx]["output"] = ast(data[idx]["output"])
            else:
                solutions[idx] = ast(solutions[idx])

        return data, solutions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = self.data[idx]["input"]
        output_data = (
            self.data[idx]["output"] if self.solutions is None else self.solutions[idx]
        )
        return {"input": input_data, "output": output_data}

    def apply(self, idx, func):
        input_data = self.data[idx]["input"]
        output_data = (
            self.data[idx]["output"] if self.solutions is None else self.solutions[idx]
        )

        return {"input": func(input_data), "output": output_data}


class ARCData(Dataset):
    def __init__(self, split):
        self.challenges, self.solutions = self.load_tasks(paths[split])
        self.keys = list(self.challenges.keys())

    def load_tasks(self, path):
        with open(path["chlng"], "r") as ch, open(path["soln"], "r") as sol:
            return json.load(ch), json.load(sol)

    def __len__(self):
        return len(self.solutions)

    def __getitem__(self, idx):
        key = self.keys[idx]
        return MetaData(self.challenges[key]["train"]), MetaData(
            self.challenges[key]["test"], self.solutions[key]
        )
