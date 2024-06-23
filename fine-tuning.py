import torch.optim.lr_scheduler
import json
from datetime import datetime
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import evaluate
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model, TaskType

device = torch.device("cpu")
if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda")


def get_optimizer(model, optimizer_name, epochs, train_loader, params, config):
    """
    Returns the optimizer&scheduler for the model (scheduler=None if not relevant)
    """
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=params[0])
        return optimizer, None
    elif optimizer_name == "AdamW exp":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=params[0])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
        config["scheduler"] = "ExponentialLR(optimizer, gamma=0.9)"
        return optimizer, scheduler
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=params[0])
        return optimizer, None
    elif optimizer_name == "Adam LinearLR":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=params[0])
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
        config["scheduler"] = "LinearLR(optimizer)"
        return optimizer, scheduler
    elif optimizer_name == "Adam exp":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=params[0])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
        config["scheduler"] = "ExponentialLR(optimizer, gamma=0.9)"
        return optimizer, scheduler
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=params[0], momentum=0.9)
        return optimizer, None
    elif optimizer_name == "SGD EXPO-LR":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=params[0], momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        config["scheduler"] = "ExponentialLR(optimizer, gamma=0.9)"
        return optimizer, scheduler
    else:
        raise ValueError("Optimizer name not recognized")


def encoder(tokenizer):
    def encode(examples):
        result = tokenizer(examples['sentence1'], examples['sentence2'], max_length=128, truncation=True,
                           padding='max_length')
        return result

    return encode


def get_loaders(model_name, batch_size, tokenizer):
    train_set = load_dataset('glue', 'mrpc', split='train')
    train_set = train_set.map(encoder(tokenizer), batched=True)
    train_set = train_set.map(lambda examples: {'labels': examples['label']}, batched=True)
    if model_name == "google/gemma-2b":
        train_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    else:
        train_set.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    eval_set = load_dataset('glue', 'mrpc', split='test')
    eval_set = eval_set.map(encoder(tokenizer), batched=True)
    eval_set = eval_set.map(lambda examples: {'labels': examples['label']}, batched=True)
    if model_name == "google/gemma-2b":
        eval_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    else:
        eval_set.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader


def valuate_model(model, metric, eval_loader):
    """Runs an evaluation epoch according to the given metric"""
    print("\nEvaluating:")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)[1]  # logits
            predictions = torch.argmax(outputs, dim=1)  # predicted labels
            metric.add_batch(predictions=predictions, references=batch['labels'])

    return metric.compute()


def train_eval(model, num_epochs, optimizer, scheduler, train_loader, eval_loader, metric, config):
    evaluations = []
    for epoch in range(1, num_epochs + 1):  # Train:
        print(f"\nStarting epoch {epoch}")
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            # if i == np.floor(len(train_loader) / 2):  # Optional - Evaluate halfway. Take with a grain(s) of salt!
            #     score = valuate_model(model, metric, eval_loader)
            #     to_app = {f"\nEval accuracy for epoch {epoch - 1}.5": round(score["accuracy"], ndigits=4)}
            #     print(to_app)
            #     evaluations.append(to_app)
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            # if i % 150 == 0:  # For feedback during training
            #     print(f"\nloss: {loss}")
        if scheduler is not None:
            scheduler.step()

        # Evaluate:
        score = valuate_model(model, metric, eval_loader)
        to_app = {f"Eval accuracy for epoch {epoch}": round(score["accuracy"], ndigits=4)}
        print("\n", to_app)
        evaluations.append(to_app)
    config["evaluations"] = evaluations
    return


def print_results(config: dict):
    print("\n        Config & Results:")
    for k, v in config.items():
        if isinstance(v, list):
            print(f"{k}:")
            for acc_dict in v:
                print(acc_dict)
        else:
            print(f"{k}: {v}")


def save_log(i: int, config_dict: dict):
    """
    Takes a dictionary of the configuration + results and saves it to a json
    """

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Define the file name using the current date-time
    file_name = f"log {i} {current_datetime}.json"

    # Save the dictionary to a JSON file
    with open(file_name, 'w') as file:
        json.dump(config_dict, file, indent=4)

    print(f"Log saved to {file_name}")


def load_model(model_name, lora, lora_rank):
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    print(model)
    lora_config = None
    if lora:
        if lora_rank is None:
            raise ValueError("lora set to true, but lora_rank was not given!")
        if model_name == "google/gemma-2b":
            modules = ["q_proj", "v_proj", "score"]
        else:  # deberta modules
            modules = ["query_proj", "value_proj", "classifier"]
        print("LoRA fine-tuning")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_rank,  # Rank
            lora_alpha=lora_rank,  # Scaling factor (alpha/rank)
            target_modules=modules
        )
        model = get_peft_model(model=model, peft_config=lora_config)
        model.print_trainable_parameters()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer, lora_config


def main(epochs: int, model_name: str, optimizer: str, lr, batch_size: int, lora: bool = False, lora_rank: int = None):
    """
    Trains the model according to the configuration given on the glue-mprc dataset, evaluating it every epoch and
    reporting the results (by the returned dictionary)
    """
    # Load up model, tokenizer and metric:
    model, tokenizer, lora_config = load_model(model_name, lora, lora_rank)

    metric = evaluate.load("accuracy", )

    # Configuration dictionary to print out at the end (evaluations will be added during training)
    config = {"epochs": epochs, "optimizer": optimizer, "lr": lr, "batch_size": batch_size}
    if lora:
        config["lora rank = alpha"] = lora_rank

    # Get data to finetune on
    train_loader, eval_loader = get_loaders(model_name, batch_size, tokenizer)
    # set optimizer & scheduler:
    optimizer, scheduler = get_optimizer(model, optimizer, epochs, train_loader, (lr,), config)
    # Finally, train and print out the results together with the hyperparameters
    train_eval(model, epochs, optimizer, scheduler, train_loader, eval_loader, metric, config)
    return config


def Q1():
    params = [
        {'epochs': 5, 'optimizer': 'Adam', 'lr': 1.1e-5, 'batch_size': 12},
        {'epochs': 5, 'optimizer': 'AdamW', 'lr': 1.2e-5, 'batch_size': 12},
        {'epochs': 5, 'optimizer': 'Adam', 'lr': 1.2e-5, 'batch_size': 12},
        {'epochs': 5, 'optimizer': 'Adam exp', 'lr': 1.1e-5, 'batch_size': 20},
    ]
    configs = []
    for i in range(len(params)):
        params[i]["model_name"] = "microsoft/deberta-v3-base"
        config = main(**params[i])
        configs.append(config)
        save_log(i, config)

    for config in configs:
        print_results(config)
    print("Q1 done")
    return


def Q2():
    params = [
        {'epochs': 5, 'optimizer': 'Adam exp', 'lr': 1.7e-04, 'batch_size': 12, "lora": True, "lora_rank": 4},
        {'epochs': 5, 'optimizer': 'AdamW', 'lr': 1.6e-04, 'batch_size': 12, "lora": True, "lora_rank": 8},
        {'epochs': 5, 'optimizer': 'Adam', 'lr': 1.5e-04, 'batch_size': 12, "lora": True, "lora_rank": 16},
        {'epochs': 5, 'optimizer': 'Adam', 'lr': 1.4e-04, 'batch_size': 12, "lora": True, "lora_rank": 24},
        {'epochs': 5, 'optimizer': 'Adam exp', 'lr': 1.8e-04, 'batch_size': 12, "lora": True, "lora_rank": 32},
        {'epochs': 5, 'optimizer': 'Adam exp', 'lr': 2.7e-04, 'batch_size': 12, "lora": True, "lora_rank": 32},
        {'epochs': 5, 'optimizer': "Adam LinearLR", 'lr': 2.5e-04, 'batch_size': 6, "lora": True, "lora_rank": 32},
    ]
    configs = []
    for i in range(len(params)):
        params[i]["model_name"] = "microsoft/deberta-v3-base"
        config = main(**params[i])
        configs.append(config)
        save_log(i, config)

    for config in configs:
        print_results(config)
    print("Q2 done")


def Q3_deberta_large():
    params = [
        {'epochs': 5, 'optimizer': "Adam LinearLR", 'lr': 2.5e-04, 'batch_size': 6, "lora": True, "lora_rank": 32},
    ]
    configs = []
    for i in range(len(params)):
        params[i]["model_name"] = "microsoft/deberta-v3-large"
        config = main(**params[i])
        configs.append(config)
        save_log(i, config)

    for config in configs:
        print_results(config)
    print("Q3 deberta done")


def Q3_gemma_2b():
    login(token="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")
    params = [
        {'epochs': 5, 'optimizer': 'Adam exp', 'lr': 1.5e-04, 'batch_size': 1, "lora": True, "lora_rank": 30}
    ]
    configs = []
    for i in range(len(params)):
        params[i]["model_name"] = "google/gemma-2b"
        config = main(**params[i])
        configs.append(config)
        save_log(i, config)

    for config in configs:
        print_results(config)
    print("Q3 deberta done")


if __name__ == '__main__':
    Q1()
    Q2()
    Q3_deberta_large()
    Q3_gemma_2b()  # can't reasonably run locally at all
    print("fine-tuning.py done")
