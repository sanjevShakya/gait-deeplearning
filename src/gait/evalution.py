from gait.config import pd
from gait.path import get_log_file_path
from gait.utils import create_dir
import json
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np

history_filename = 'history.csv'
evaluation_filename = 'evaluation.csv'
model_accuracy_loss_filename = "accuracy_loss.jpg"
confusion_matrix_figure_filename = "confusion_matrix_figure_filename.jpg"
classification_report_filename = 'classification_report.csv'
validations_predictions_filename = "validation_predictions.csv"

def classification_report_csv(report, filePath):
    report_data = []
    lines = report.split('\n')
    print('--------CLASSIFICATION REPORT--------')
    print(report)
    print('-------------------------------------')
    try: 
        for line in lines[2:-5]:
            row = {}
            row_data = line.split('      ')
            row['class'] = row_data[1].strip()
            row['precision'] = float(row_data[2])
            row['recall'] = float(row_data[3])
            row['f1_score'] = float(row_data[4])
            row['support'] = float(row_data[5])
            report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv(filePath, index = False)
    except:
        pass


def save_history(history, overlap_percent):
    df = pd.DataFrame(history.history)
    dir_path, log_path = get_log_file_path(overlap_percent, history_filename)
    create_dir(dir_path)
    df.to_csv(log_path)
    print('Saved training details to {}'.format(log_path))


def save_test_history(evaluation_history, overlap_percent):
    evalution_loss, evalution_accuracy = evaluation_history
    data = {
        "loss": evalution_loss,
        "accuracy": evalution_accuracy
    }

    dir_path, log_path = get_log_file_path(
        overlap_percent, evaluation_filename)
    create_dir(dir_path)
    with open(log_path, "w") as outfile:
        json.dump(data, outfile)
        print("Saved validation details to {}".format(log_path))


def save_accuracy_loss_figure(history, overlap_percent):
    dir_path, log_path = get_log_file_path(
        overlap_percent, model_accuracy_loss_filename)
    create_dir(dir_path)

    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], 'r',
             label='Accuracy of training data')
    plt.plot(history.history['val_accuracy'], 'b',
             label='Accuracy of validation data')
    plt.plot(history.history['loss'], 'r--', label='Loss of training data')
    plt.plot(history.history['val_loss'], 'b--',
             label='Loss of validation data')
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.draw()
    plt.savefig(log_path, dpi=100)
    plt.show()


def save_classification_report(validations, predictions, overlap_percent):
    dir_path, file_path = get_log_file_path(
        overlap_percent, classification_report_filename)
    create_dir(dir_path)
    target_names = ['1', '2',
                    '3', '4', '5']
    report = classification_report(
        validations, predictions, target_names=target_names)
    classification_report_csv(report, file_path)


def save_confusion_matrix_figure(validations, predictions, overlap_percent, sns):
    dir_path, figure_path = get_log_file_path(
        overlap_percent, confusion_matrix_figure_filename)
    create_dir(dir_path)
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='magma_r',
                linecolor='white',
                linewidths=1,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(figure_path, dpi=100)
    plt.show()
    pass


def compute_validations_predictions(model, X_test, y_test):
    y_pred_test = model.predict(X_test)
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(y_test, axis=1)

    return max_y_pred_test, max_y_test


def compute_validations_predictions_with_stats(model, X_test, y_test, X_test_stats):
    y_pred_test = model.predict([X_test, X_test_stats])
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(y_test, axis=1)

    return max_y_pred_test, max_y_test


def compute_validations_predictions_cnn_multihead_with_stats(model, X_test, y_test, X_test_stats):
    y_pred_test = model.predict([X_test, X_test, X_test, X_test_stats])
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(y_test, axis=1)

    return max_y_pred_test, max_y_test
