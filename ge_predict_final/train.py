import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR




def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        current_batch_size = targets.shape[0]
        if current_batch_size != config["batch_size"]:
            print(f"Warning: Batch has a different size ({current_batch_size}) than the specified batch size ({config['batch_size']}).")

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)



def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            current_batch_size = targets.shape[0]
            if current_batch_size != config["batch_size"]:
                print(f"Warning: Batch has a different size ({current_batch_size}) than the specified batch size ({config['batch_size']}).")
                print(f"Inputs size: {inputs.shape}, Outputs size: {outputs.shape}")

            if outputs.dim() == 0:
                print("Warning: outputs is a scalar")
                all_preds.append(outputs.item())
            else:
                all_preds.extend(outputs.squeeze().cpu().numpy())

            if targets.dim() == 0:
                print("Warning: targets is a scalar")
                all_labels.append(targets.item())
            else:
                all_labels.extend(targets.cpu().numpy())
    return all_preds, all_labels





def train_and_evaluate(config, data_tensor_filtered, rna_seq_values, rna_seq_values_df, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = StepLR(optimizer, step_size=config["scheduler_step_size"], gamma=config["scheduler_gamma"])

    results = []
    all_tissue_predictions_df = pd.DataFrame()

    for tissue in rna_seq_values_df['rna_seq_tissue'].unique():
        print(f"Holding out {tissue}")
        train_df = rna_seq_values_df.loc[rna_seq_values_df['rna_seq_tissue'] != tissue]
        leave_out_df = rna_seq_values_df.loc[rna_seq_values_df['rna_seq_tissue'] == tissue]
        train_indices = list(train_df.index)
        leave_out_indices = list(leave_out_df.index)

        train_array = data_tensor_filtered[train_indices]
        test_array = data_tensor_filtered[leave_out_indices]
        train_target = rna_seq_values[train_indices]
        test_target = rna_seq_values[leave_out_indices]

        train_data, val_data, train_labels, val_labels = train_test_split(train_array, train_target, test_size=0.2, random_state=config["random_seed"])

        train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32).unsqueeze(1), torch.tensor(train_labels, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32).unsqueeze(1), torch.tensor(val_labels, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(test_array, dtype=torch.float32).unsqueeze(1), torch.tensor(test_target, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

        model = model.to(device)
        model.apply(weights_init)

        if config["use_wandb"]:
            wandb.watch(model, log_freq=100)

        best_val_loss = float('inf')
        counter = 0
        best_model = None

        for epoch in range(config["epochs"]):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            val_preds, val_labels = evaluate(model, val_loader, device)
            val_mae = mean_absolute_error(val_labels, val_preds)
            print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}")

            if val_mae < best_val_loss:
                best_val_loss = val_mae
                counter = 0
                best_model = model.state_dict()
            else:
                counter += 1
                if counter >= config["patience"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            scheduler.step()

        model.load_state_dict(best_model)

        combined_data = np.concatenate((train_data, val_data), axis=0)
        combined_labels = np.concatenate((train_labels, val_labels), axis=0)

        combined_dataset = TensorDataset(torch.tensor(combined_data, dtype=torch.float32).unsqueeze(1), torch.tensor(combined_labels, dtype=torch.float32))
        combined_loader = DataLoader(combined_dataset, batch_size=config["batch_size"], shuffle=True)

        model = model.to(device)
        model.apply(weights_init)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = StepLR(optimizer, step_size=config["scheduler_step_size"], gamma=config["scheduler_gamma"])

        for epoch in range(config["epochs"]):
            train_loss = train(model, combined_loader, criterion, optimizer, device)
            print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}")
            scheduler.step()

        train_preds, train_labels = evaluate(model, combined_loader, device)
        test_preds, test_labels = evaluate(model, test_loader, device)

        leave_out_df['model_preds'] = test_preds
        all_tissue_predictions_df = pd.concat([all_tissue_predictions_df, leave_out_df])

        predictions_dict[tissue] = {
            'train_preds': train_preds,
            'train_labels': train_labels,
            'test_preds': test_preds,
            'test_labels': test_labels
        }

        train_mae = mean_absolute_error(train_labels, train_preds)
        test_mae = mean_absolute_error(test_labels, test_preds)

        train_r2 = r2_score(train_labels, train_preds)
        test_r2 = r2_score(test_labels, test_preds)

        print(f"[{tissue}] Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        print(f"[{tissue}] Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")

        results.append({
            'train_samples': combined_data.shape[0],
            'test_samples': test_array.shape[0],
            'Tissue': tissue,
            'Train MAE': train_mae,
            'Test MAE': test_mae,
            'Train R2': train_r2,
            'Test R2': test_r2
        })

        if config["use_wandb"]:
            wandb.log({
                'train_data': train_array[10,:,:],
                'test_data': test_array[10,:,:],
                'Tissue': tissue,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Train R2': train_r2,
                'Test R2': test_r2
            })

        model_save_path = config["model_save_dir"] + f"{config['model_name']}{tissue}.trained_model.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(config["results_save_dir"] + config["model_name"] + 'tissue_results.csv', sep='\t')

    with open(config["results_save_dir"] + config["model_name"] + "predictions_dict.pkl", "wb") as f:
        pickle.dump(predictions_dict, f)

    all_tissue_predictions_df.to_csv(config["results_save_dir"] + config["model_name"] + "model_predictions_df.csv")

    return results_df, all_tissue_predictions_df