import matplotlib.pyplot as plt
import numpy as np
def show_box_plot(data, title="Cross Validation Accuracies"):
	f = plt.figure()
	f.set_figwidth(4)
	f.set_figheight(1)

	# Create a box and whiskers plot
	plt.boxplot(data, vert=False)

	plt.scatter(data, [1]*len(data), color='blue', alpha=0.3, s=20)

	# Add title and labels
	plt.title(title)
	plt.yticks([])  # Removes all y-ticks and their labels

	# Show the plot
	plt.show()

def show_confusion_matrix(cm, title="Confusion Matrix"):
	fig, ax = plt.subplots()
	cax = ax.matshow(cm, cmap=plt.cm.Blues)

	for (i, j), z in np.ndenumerate(cm):
		ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
				bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'),
                fontsize=20)
	
	ax.set_xticks([0, 1])
	ax.set_yticks([0, 1])   
	ax.set_xticklabels(['Predicted Negative', 'Predicted Positive'])
	ax.set_yticklabels(['True Negative', 'True Positive'])
	plt.title(title)
	plt.savefig("test.svg", format="svg")
	plt.savefig("confusion.svg")
     
def show_train_and_test_accuracy_over_epochs(history):
    # Initialize lists to store accuracies for each fold
	train_fold_accuracies = np.zeros((len(history["train"][-1]), len(history["train"])))
	test_fold_accuracies = np.zeros((len(history["train"][-1]), len(history["train"])))

	# Extract training and testing accuracies for each fold across all epochs
	for epoch_index, epoch_data in enumerate(history["train"]):
		for fold_index, fold_data in enumerate(epoch_data):
			train_fold_accuracies[fold_index][epoch_index] = fold_data[1]

	for epoch_index, epoch_data in enumerate(history["test"]):
		for fold_index, fold_data in enumerate(epoch_data):
			test_fold_accuracies[fold_index][epoch_index] = fold_data[1]

	# Plot training and testing accuracies for each fold on one graph
	plt.figure(figsize=(10, 5))
	for fold_index, accuracies in enumerate(train_fold_accuracies):
		plt.plot(accuracies, color='orange', alpha=.4)
	for fold_index, accuracies in enumerate(test_fold_accuracies):
		plt.plot(accuracies, color='blue', alpha=.4)

	# Add custom legend
	leg = plt.legend(['Training', 'Testing'], loc='upper left')
	leg.legend_handles[0].set_color('orange')
	leg.legend_handles[1].set_color('blue')

	plt.xlabel('Timestamps')
	plt.ylabel('Accuracy')
	plt.xticks(np.arange(1, len(history["train"]), step=1))
	plt.title('Training and Testing Accuracies Over Timestamps')
	plt.show()

from sklearn.metrics import roc_curve, auc
# Unused
def show_roc_curve(y_true, y_scores, title="ROC Curve"):
	"""
	Show ROC curve for binary classification
	
	Args:
		y_true: True binary labels
		y_scores: Target scores (probabilities or confidence scores)
		title: Title for the plot
	"""
	# Calculate ROC curve
	
	fpr, tpr, thresholds = roc_curve(y_true, y_scores)
	roc_auc = auc(fpr, tpr)
	
	# Create the plot
	plt.figure(figsize=(6, 6))
	plt.plot(fpr, tpr, color='darkorange', lw=2, 
			 label=f'ROC curve (AUC = {roc_auc:.2f})')
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
			 label='Random classifier')
	
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(title)
	plt.legend(loc="lower right")
	plt.grid(True, alpha=0.3)
	plt.show()
	
	return roc_auc

def show_multiple_roc_curves(history, title="ROC Curves - Last Epoch by Fold", dataset_type="test"):
    """
    Show ROC curves for each fold from the last epoch only
    Each fold gets its own line, with train/test distinguished by color
    
    Args:
        history: The history dictionary from cv_test_model
        title: Title for the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Get last epoch data
    test_last_epoch = history[dataset_type][-1]
    
    test_aucs = []
    
    # Plot ROC curve for each fold - Test data
    for fold_idx, fold_data in enumerate(test_last_epoch):
        conf_matrix, accuracy, probabilities, true_labels = fold_data
        if probabilities is not None and true_labels is not None:
            fpr, tpr, _ = roc_curve(true_labels, probabilities)
            roc_auc = auc(fpr, tpr)
            test_aucs.append(roc_auc)
            plt.plot(fpr, tpr, color='blue', alpha=0.4, linewidth=2)
    
    # Plot random classifier line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    plt.tight_layout()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()

def show_roc_curves_over_epochs(history, title="ROC"):
    """
    Show ROC AUC scores over epochs for train/test data
    Similar to the accuracy over epochs plot
    
    Args:
        history: The history dictionary from cv_test_model
        title: Title for the plot
    """
    num_folds = len(history["train"][0])
    num_epochs = len(history["train"])
    
    # Initialize arrays to store AUC scores
    train_fold_aucs = np.zeros((num_folds, num_epochs))
    test_fold_aucs = np.zeros((num_folds, num_epochs))
    
    # Extract AUC scores for each fold across all epochs
    for epoch_index, epoch_data in enumerate(history["train"]):
        for fold_index, fold_data in enumerate(epoch_data):
            conf_matrix, accuracy, probabilities, true_labels = fold_data
            if probabilities is not None and true_labels is not None:
                fpr, tpr, _ = roc_curve(true_labels, probabilities)
                train_fold_aucs[fold_index][epoch_index] = auc(fpr, tpr)
    
    for epoch_index, epoch_data in enumerate(history["test"]):
        for fold_index, fold_data in enumerate(epoch_data):
            conf_matrix, accuracy, probabilities, true_labels = fold_data
            if probabilities is not None and true_labels is not None:
                fpr, tpr, _ = roc_curve(true_labels, probabilities)
                test_fold_aucs[fold_index][epoch_index] = auc(fpr, tpr)
    
    # Plot AUC scores for each fold on one graph
    plt.figure(figsize=(10, 5))
    for fold_index, aucs in enumerate(train_fold_aucs):
        plt.plot(aucs, color='orange', alpha=0.4)
    for fold_index, aucs in enumerate(test_fold_aucs):
        plt.plot(aucs, color='blue', alpha=0.4)
    
    # Add custom legend
    leg = plt.legend(['Training', 'Testing'], loc='upper left')
    leg.legend_handles[0].set_color('orange')
    leg.legend_handles[1].set_color('blue')
    
    plt.xlabel('Epochs')
    plt.ylabel('ROC AUC')
    plt.xticks(np.arange(0, num_epochs, step=1))
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.0, 1.0])  # AUC is always between 0 and 1
    plt.show()
    
    return train_fold_aucs, test_fold_aucs