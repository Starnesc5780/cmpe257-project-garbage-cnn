import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset
from data_processing.process_data import BATCH_SIZE, split_dataset_indices


def build_test_loader(dataset, batch_size=BATCH_SIZE):
	train_indices, val_indices, test_indices = split_dataset_indices(dataset)
	test_dataset = Subset(dataset, test_indices)
	return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def load_checkpoint(model, checkpoint_path, device):
	try:
		state_dict = torch.load(checkpoint_path, map_location=device)
	except FileNotFoundError as exc:
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}") from exc
	model.load_state_dict(state_dict)
	return model


def evaluate_model(model, test_loader, device):
	model.eval()
	test_preds = []
	test_labels = []
	with torch.no_grad():
		for inputs, labels in test_loader:
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			_, predicted = torch.max(outputs.data, 1)
			test_preds.extend(predicted.cpu().numpy())
			test_labels.extend(labels.cpu().numpy())

	return test_labels, test_preds


def compute_metrics(test_labels, test_preds, class_names):
	#Compute the evaluation metrics used by all experiment scripts
	accuracy = accuracy_score(test_labels, test_preds)
	macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
	report = classification_report(test_labels, test_preds, target_names=class_names, zero_division=0, output_dict=True)
	conf_matrix = confusion_matrix(test_labels, test_preds)
	return {
		"accuracy": accuracy,
		"macro_f1": macro_f1,
		"report": report,
		"confusion_matrix": conf_matrix,
	}


def print_metrics(experiment_name, checkpoint_path, metrics):
	print(f"{experiment_name} checkpoint: {checkpoint_path}")
	print(f"{experiment_name} accuracy: {metrics['accuracy'] * 100:.2f}%")
	print(f"{experiment_name} macro F1: {metrics['macro_f1']:.4f}")
	print("Per-class precision/recall/F1:")
	for class_name, scores in metrics["report"].items():
		if class_name in ("accuracy", "macro avg", "weighted avg"):
			continue
		print(
			f"  {class_name}: precision={scores['precision']:.4f}, "
			f"recall={scores['recall']:.4f}, f1={scores['f1-score']:.4f}"
		)
	print("Confusion matrix:")
	print(metrics["confusion_matrix"])