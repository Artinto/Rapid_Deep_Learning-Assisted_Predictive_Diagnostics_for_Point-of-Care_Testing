from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os


def plot_roc_curve(y, y_pred, save_path=''):
    fper, tper, th = roc_curve(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    plt.plot(fper, tper, color='red', label=f'ROC (area={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    img_path = os.path.join(save_path, 'roc_curve.png')
    plt.savefig(img_path, dpi=300)