import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
from scipy.signal import savgol_filter

class Visualizer:
    def __init__(self):
        """
        Visualizador para los CSVs generados por el programa C++
        """
        self.class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        
        # Configurar estilo de gráficas
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def load_data(self):
        """Cargar datos de los CSVs"""
        self.training_history = None
        self.predictions = None
        self.true_labels = None
        self.confidences = None
        self.test_loss = None
        self.test_accuracy = None
        
        # Cargar historial de entrenamiento
        if os.path.exists('training_history.csv'):
            self.training_history = pd.read_csv('training_history.csv')
            print("✅ Cargado training_history.csv")
        else:
            print("❌ training_history.csv no encontrado")
        
        # Cargar predicciones
        if os.path.exists('predictions.csv'):
            preds_df = pd.read_csv('predictions.csv')
            self.true_labels = preds_df['true_label'].tolist()
            self.predictions = preds_df['predicted_label'].tolist()
            self.confidences = preds_df['confidence'].tolist()
            print("✅ Cargado predictions.csv")
        else:
            print("❌ predictions.csv no encontrado")
        
        # Cargar métricas de test
        if os.path.exists('test_metrics.csv'):
            test_df = pd.read_csv('test_metrics.csv')
            self.test_loss = test_df['test_loss'].iloc[0]
            self.test_accuracy = test_df['test_accuracy'].iloc[0]
            print("✅ Cargado test_metrics.csv")
        else:
            print("❌ test_metrics.csv no encontrado")
        
        return self.training_history is not None or self.predictions is not None
    
    def plot_training_history(self):
        """Graficar historial de entrenamiento"""
        if self.training_history is None:
            print("❌ No hay datos de entrenamiento para graficar")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📈 Fashion-MNIST Transformer - Training History', fontsize=16, fontweight='bold')
        
        epochs = self.training_history['epoch']
        loss = self.training_history['loss']
        accuracy = self.training_history['accuracy']
        lr = self.training_history['learning_rate']
        
        # 1. Loss vs Epoch
        ax1.plot(epochs, loss, 'b-', linewidth=2, marker='o')
        ax1.set_title('📉 Training Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # 2. Accuracy vs Epoch
        ax2.plot(epochs, accuracy, 'g-', linewidth=2, marker='s')
        ax2.set_title('🎯 Training Accuracy', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # 3. Learning Rate vs Epoch
        ax3.semilogy(epochs, lr, 'r-', linewidth=2, marker='^')
        ax3.set_title('📊 Learning Rate Schedule', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate (log scale)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Loss vs Accuracy
        ax4.scatter(loss, accuracy, 
                   c=epochs, cmap='viridis', s=60, alpha=0.7)
        ax4.set_title('📈 Loss vs Accuracy', fontweight='bold')
        ax4.set_xlabel('Loss')
        ax4.set_ylabel('Accuracy (%)')
        ax4.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Epoch')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Gráfica guardada como 'training_history.png'")
    
    def plot_confusion_matrix(self):
        """Graficar matriz de confusión"""
        if not self.predictions or not self.true_labels:
            print("❌ No hay datos de predicciones para la matriz de confusión")
            return
        
        # Calcular matriz de confusión
        cm = confusion_matrix(self.true_labels, self.predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('🎯 Confusion Matrix - Fashion-MNIST Transformer', fontsize=16, fontweight='bold')
        
        # Matriz absoluta
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('📊 Absolute Counts', fontweight='bold')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Matriz normalizada
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax2, cbar_kws={'label': 'Proportion'})
        ax2.set_title('📈 Normalized (by True Label)', fontweight='bold')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Matriz de confusión guardada como 'confusion_matrix.png'")
        
        # Mostrar estadísticas por clase
        self.print_classification_stats(cm)
    
    def print_classification_stats(self, cm):
        """Imprimir estadísticas detalladas por clase"""
        print("\n📊 Estadísticas por clase:")
        print("=" * 80)
        
        # Calcular métricas por clase
        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Crear DataFrame para mejor visualización
        stats_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Support': np.sum(cm, axis=1)
        })
        
        print(stats_df.round(3).to_string(index=False))
        
        # Estadísticas globales
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        macro_avg_precision = np.mean(precision)
        macro_avg_recall = np.mean(recall)
        macro_avg_f1 = np.mean(f1_score)
        
        print(f"\n🎯 Métricas globales:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Macro Avg Precision: {macro_avg_precision:.3f}")
        print(f"   Macro Avg Recall: {macro_avg_recall:.3f}")
        print(f"   Macro Avg F1-Score: {macro_avg_f1:.3f}")
    
    def plot_per_class_accuracy(self):
        """Graficar accuracy por clase"""
        if not self.predictions or not self.true_labels:
            print("❌ No hay datos para accuracy por clase")
            return
        
        # Calcular accuracy por clase
        cm = confusion_matrix(self.true_labels, self.predictions)
        class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
        
        # Crear gráfica
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        bars = plt.bar(range(len(self.class_names)), class_accuracy * 100, 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        plt.title('📊 Accuracy por Clase - Fashion-MNIST Transformer', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Clases', fontweight='bold')
        plt.ylabel('Accuracy (%)', fontweight='bold')
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Agregar valores en las barras
        for i, (bar, acc) in enumerate(zip(bars, class_accuracy)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Línea de accuracy promedio
        avg_accuracy = np.mean(class_accuracy) * 100
        plt.axhline(y=avg_accuracy, color='red', linestyle='--', linewidth=2, 
                   label=f'Promedio: {avg_accuracy:.1f}%')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('per_class_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Accuracy por clase guardado como 'per_class_accuracy.png'")
    
    def plot_roc_curves(self):
        """Graficar curvas ROC multiclase (simulando probabilidades ya que no se tienen)"""
        if not self.predictions or not self.true_labels or not self.confidences:
            print("❌ No hay datos para curvas ROC")
            return
        
        # Binarizar las etiquetas para ROC multiclase
        y_true_bin = label_binarize(self.true_labels, classes=list(range(10)))
        
        # Simular probabilidades basadas en confidences (one-hot con confidence en predicted)
        y_score = np.zeros((len(self.true_labels), 10))
        for i, (pred, conf) in enumerate(zip(self.predictions, self.confidences)):
            y_score[i, pred] = conf
            # Distribuir el resto uniformemente para simular softmax
            remaining = (1 - conf) / 9
            for j in range(10):
                if j != pred:
                    y_score[i, j] = remaining
        
        # Calcular ROC para cada clase
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(10):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Micro-average ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(10):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= 10
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Plotear
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Todas las clases
        plt.subplot(2, 2, 1)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, color in zip(range(10), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('🎯 ROC Curves - All Classes')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Micro y Macro average
        plt.subplot(2, 2, 2)
        plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', 
                linewidth=4, label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})')
        plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', 
                linewidth=4, label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('📈 ROC Curves - Average')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: AUC por clase
        plt.subplot(2, 2, 3)
        class_aucs = [roc_auc[i] for i in range(10)]
        bars = plt.bar(range(10), class_aucs, color=colors, alpha=0.7)
        plt.title('📊 AUC por Clase')
        plt.xlabel('Clases')
        plt.ylabel('AUC')
        plt.xticks(range(10), [name[:8] + '...' if len(name) > 8 else name 
                              for name in self.class_names], rotation=45)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bar, auc_val in zip(bars, class_aucs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{auc_val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Subplot 4: Distribución de AUC
        plt.subplot(2, 2, 4)
        plt.hist(class_aucs, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(class_aucs), color='red', linestyle='--', 
                   label=f'Media: {np.mean(class_aucs):.3f} ')
        plt.title('📈 Distribución de AUC')
        plt.xlabel('AUC Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Curvas ROC guardadas como 'roc_curves.png'")
    
    def plot_learning_curves(self):
        """Graficar curvas de aprendizaje avanzadas"""
        if self.training_history is None:
            print("❌ No hay datos de entrenamiento para curvas de aprendizaje")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📚 Learning Curves Analysis - Fashion-MNIST Transformer', 
        fontsize=16, fontweight='bold')
        
        epochs = self.training_history['epoch']
        loss = self.training_history['loss']
        accuracy = self.training_history['accuracy']
        lr = self.training_history['learning_rate']
        
        # 1. Loss con smoothing
        ax1.plot(epochs, loss, 'b-', alpha=0.3, label='Raw Loss')
        if len(loss) > 3:
            smoothed_loss = savgol_filter(loss, min(len(loss)//3*2 +1, 5), 2)
            ax1.plot(epochs, smoothed_loss, 'b-', linewidth=3, label='Smoothed Loss')
        ax1.set_title('📉 Training Loss (Smoothed)', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy con intervalos de confianza
        ax2.plot(epochs, accuracy, 'g-', linewidth=2, marker='o', label='Accuracy')
        if len(accuracy) > 5:
            window = 3
            rolling_std = self.training_history['accuracy'].rolling(window, center=True).std()
            rolling_mean = self.training_history['accuracy'].rolling(window, center=True).mean()
            ax2.fill_between(epochs, 
                           rolling_mean - rolling_std, 
                           rolling_mean + rolling_std, 
                           alpha=0.2, color='green', label='±1 Std Dev')
        ax2.set_title('🎯 Training Accuracy', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # 3. Learning Rate Schedule
        ax3.plot(epochs, lr, 'r-', linewidth=2, marker='^')
        ax3.set_title('📊 Learning Rate Schedule', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. Training Efficiency (Acc/Loss ratio)
        efficiency = np.array(accuracy) / (np.array(loss) + 1e-8)
        ax4.plot(epochs, efficiency, 'purple', linewidth=2, marker='s')
        ax4.set_title('⚡ Training Efficiency (Acc/Loss)', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Efficiency Ratio')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Curvas de aprendizaje guardadas como 'learning_curves.png'")
    
    def generate_all_plots(self):
        """Generar todas las visualizaciones"""
        print("\n🎨 Generando visualizaciones...")
        print("=" * 50)
        
        if not self.load_data():
            print("❌ No se encontraron archivos CSV. Ejecuta el programa C++ primero.")
            return
        
        try:
            # 1. Historial de entrenamiento
            print("📈 1. Generando historial de entrenamiento...")
            self.plot_training_history()
            
            # 2. Matriz de confusión
            print("🎯 2. Generando matriz de confusión...")
            self.plot_confusion_matrix()
            
            # 3. Accuracy por clase
            print("📊 3. Generando accuracy por clase...")
            self.plot_per_class_accuracy()
            
            # 4. Curvas ROC
            print("📈 4. Generando curvas ROC...")
            self.plot_roc_curves()
            
            # 5. Curvas de aprendizaje
            print("📚 5. Generando curvas de aprendizaje...")
            self.plot_learning_curves()
            
            if self.test_loss is not None:
                print(f"\n📈 Test Loss: {self.test_loss:.4f}")
                print(f"Test Accuracy: {self.test_accuracy:.2f}%")
            
            print("\n✅ Todas las visualizaciones generadas exitosamente!")
            
        except Exception as e:
            print(f"❌ Error generando gráficas: {e}")
            print("💡 Asegúrate de tener instaladas: matplotlib, seaborn, scikit-learn, pandas, scipy")
    
def main():
    visualizer = Visualizer()
    visualizer.generate_all_plots()

if __name__ == "__main__":
    main()