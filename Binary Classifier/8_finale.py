# ecg_classification_gui.py
# Simple GUI for ECG classification on uploaded CSV files

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import json
import joblib
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class ECGPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Classification Tool")
        self.root.geometry("1000x800")
        
        # Initialize variables
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.model_name = None
        self.results_df = None
        
        # Create GUI elements
        self.create_widgets()
        
        # Try to load model on startup
        self.load_model()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="ECG Abnormality Classification Tool", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Model section
        model_frame = ttk.LabelFrame(main_frame, text="Model Information", padding="10")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Model Status:").grid(row=0, column=0, sticky=tk.W)
        self.model_status_label = ttk.Label(model_frame, text="Not loaded", foreground="red")
        self.model_status_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Button(model_frame, text="Load Model", command=self.load_model).grid(row=0, column=2)
        
        self.model_info_text = scrolledtext.ScrolledText(model_frame, height=4, width=60)
        self.model_info_text.grid(row=1, column=0, columnspan=3, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # File upload section
        upload_frame = ttk.LabelFrame(main_frame, text="Upload CSV File", padding="10")
        upload_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        upload_frame.columnconfigure(1, weight=1)
        
        ttk.Label(upload_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W)
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ttk.Entry(upload_frame, textvariable=self.file_path_var, state="readonly")
        self.file_path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 10))
        
        ttk.Button(upload_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        ttk.Button(upload_frame, text="Classify", command=self.classify_data).grid(row=0, column=3, padx=(10, 0))
        
        # Instructions
        instructions = """
Instructions:
1. Upload a CSV file containing ECG features (without 'label' column for unseen data)
2. The CSV should have the same feature columns as the training data
3. Click 'Classify' to predict Normal/Abnormal for each row
4. Results will be displayed below with confidence scores
        """
        self.instructions_label = ttk.Label(upload_frame, text=instructions, justify=tk.LEFT)
        self.instructions_label.grid(row=1, column=0, columnspan=4, pady=(10, 0), sticky=tk.W)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Results summary
        self.summary_text = scrolledtext.ScrolledText(results_frame, height=6, width=80)
        self.summary_text.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Results table
        self.tree = ttk.Treeview(results_frame, columns=('Index', 'Prediction', 'Confidence', 'Prob_Normal', 'Prob_Abnormal'), show='headings')
        self.tree.heading('Index', text='Row')
        self.tree.heading('Prediction', text='Prediction')
        self.tree.heading('Confidence', text='Confidence')
        self.tree.heading('Prob_Normal', text='P(Normal)')
        self.tree.heading('Prob_Abnormal', text='P(Abnormal)')
        
        # Column widths
        self.tree.column('Index', width=60)
        self.tree.column('Prediction', width=100)
        self.tree.column('Confidence', width=100)
        self.tree.column('Prob_Normal', width=100)
        self.tree.column('Prob_Abnormal', width=100)
        
        self.tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for tree
        tree_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=tree_scroll.set)
        
        # Export buttons
        button_frame = ttk.Frame(results_frame)
        button_frame.grid(row=2, column=0, pady=(10, 0))
        
        ttk.Button(button_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Show Statistics", command=self.show_statistics).pack(side=tk.LEFT)
        
        # Configure grid weights for main frame
        main_frame.rowconfigure(3, weight=1)
    
    def load_model(self):
        """Load the trained ECG model"""
        try:
            # Update this path to match your deployment folder
            pkg_dir = Path("ecg_model_deployment_20250821_125946")
            predictor_path = pkg_dir / "ecg_predictor.joblib"
            meta_path = pkg_dir / "ecg_features.json"
            
            if not predictor_path.exists():
                # Try to browse for model file
                predictor_path = filedialog.askopenfilename(
                    title="Select ECG Predictor File",
                    filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
                )
                if not predictor_path:
                    return
                predictor_path = Path(predictor_path)
                meta_path = predictor_path.parent / "ecg_features.json"
            
            # Load predictor object
            predictor_obj = joblib.load(predictor_path)
            
            # Extract components from the predictor object
            if hasattr(predictor_obj, 'model'):
                self.model = predictor_obj.model
                self.scaler = predictor_obj.scaler
                self.selected_features = predictor_obj.selected_features
                self.model_name = predictor_obj.model_name
            else:
                # If it's just a direct model object
                self.model = predictor_obj
                self.scaler = None
                
                # Try to load features from metadata
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    self.selected_features = meta["selected_features"]
                    self.model_name = meta.get("model_name", "Unknown")
                else:
                    messagebox.showerror("Error", "Could not load feature information")
                    return
            
            # Update status
            self.model_status_label.config(text="Loaded successfully", foreground="green")
            
            # Update info text
            info_text = f"""Model loaded successfully!
Path: {predictor_path}
Model type: {self.model_name}
Features required: {len(self.selected_features)}
Scaler: {'Yes' if self.scaler else 'No'}
Key features: {', '.join(self.selected_features[:5])}{'...' if len(self.selected_features) > 5 else ''}
"""
            self.model_info_text.delete(1.0, tk.END)
            self.model_info_text.insert(1.0, info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model_status_label.config(text="Failed to load", foreground="red")
    
    def browse_file(self):
        """Browse for CSV file to classify"""
        file_path = filedialog.askopenfilename(
            title="Select CSV file to classify",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
    
    def predict_batch(self, features_df):
        """
        Batch prediction compatible with any sklearn model
        """
        # Ensure all required features are present
        missing = set(self.selected_features) - set(features_df.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Select and order features correctly
        X = features_df[self.selected_features].copy()
        
        # Handle any remaining NaN values
        if X.isnull().any().any():
            print("Warning: NaN values found in features. Filling with median values.")
            X = X.fillna(X.median())
        
        # Scale features if scaler exists
        if self.scaler is not None:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X.values
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        
        return {
            'predictions': predictions.astype(int),
            'probabilities': probabilities,
            'probability_normal': probabilities[:, 0],
            'probability_abnormal': probabilities[:, 1],
            'confidence': np.max(probabilities, axis=1),
            'model_used': self.model_name,
            'n_samples': len(X),
            'timestamp': datetime.now().isoformat()
        }
    
    def classify_data(self):
        """Classify the uploaded CSV data"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if not self.file_path_var.get():
            messagebox.showerror("Error", "Please select a CSV file")
            return
        
        try:
            # Load CSV
            df = pd.read_csv(self.file_path_var.get())
            
            # Check if required features are present
            missing_features = set(self.selected_features) - set(df.columns)
            if missing_features:
                messagebox.showerror(
                    "Missing Features", 
                    f"The following required features are missing:\n{', '.join(missing_features)}\n\n"
                    f"Required features:\n{', '.join(self.selected_features)}"
                )
                return
            
            # Check if there's a 'label' column and warn user
            if 'label' in df.columns:
                response = messagebox.askyesno(
                    "Label Column Detected",
                    "The CSV contains a 'label' column. This suggests it might be training data.\n"
                    "For truly unseen data, the CSV should NOT have a 'label' column.\n\n"
                    "Do you want to proceed anyway? (The label column will be ignored)"
                )
                if not response:
                    return
            
            # Clean data - remove rows with NaN in required features
            df_clean = df.dropna(subset=self.selected_features)
            if len(df_clean) < len(df):
                messagebox.showwarning(
                    "Data Cleaning", 
                    f"Removed {len(df) - len(df_clean)} rows with missing values.\n"
                    f"Classifying {len(df_clean)} complete rows."
                )
            
            if df_clean.empty:
                messagebox.showerror("Error", "No complete rows found after removing missing values")
                return
            
            # Extract features only (ignore any label column)
            X_features = df_clean[self.selected_features]
            
            # Make predictions using our batch prediction method
            self.update_status("Making predictions...")
            predictions_dict = self.predict_batch(X_features)
            
            # Store results
            self.results_df = pd.DataFrame({
                'row_index': df_clean.index,
                'prediction': predictions_dict['predictions'],
                'prediction_label': ['Abnormal' if p == 1 else 'Normal' for p in predictions_dict['predictions']],
                'confidence': predictions_dict['confidence'],
                'prob_normal': predictions_dict['probability_normal'],
                'prob_abnormal': predictions_dict['probability_abnormal']
            })
            
            # Update GUI
            self.display_results()
            self.update_status(f"Classification completed! Processed {len(self.results_df)} samples.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            self.update_status("Classification failed")
    
    def display_results(self):
        """Display classification results in the tree view"""
        # Clear existing results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add results to tree
        for idx, row in self.results_df.iterrows():
            # Color code by prediction
            tags = ('abnormal',) if row['prediction_label'] == 'Abnormal' else ('normal',)
            
            self.tree.insert('', 'end', values=(
                int(row['row_index']),
                row['prediction_label'],
                f"{row['confidence']:.3f}",
                f"{row['prob_normal']:.3f}",
                f"{row['prob_abnormal']:.3f}"
            ), tags=tags)
        
        # Configure tag colors
        self.tree.tag_configure('abnormal', background='#ffebee')  # Light red
        self.tree.tag_configure('normal', background='#e8f5e8')    # Light green
        
        # Update summary
        self.update_summary()
    
    def update_summary(self):
        """Update the summary text with classification statistics"""
        if self.results_df is None:
            return
        
        n_total = len(self.results_df)
        n_normal = len(self.results_df[self.results_df['prediction'] == 0])
        n_abnormal = len(self.results_df[self.results_df['prediction'] == 1])
        avg_confidence = self.results_df['confidence'].mean()
        high_conf_threshold = 0.8
        n_high_conf = len(self.results_df[self.results_df['confidence'] >= high_conf_threshold])
        
        # Find samples with highest abnormal probability
        top_abnormal = self.results_df.nlargest(3, 'prob_abnormal')
        
        summary_text = f"""CLASSIFICATION SUMMARY
=====================================
Total samples processed: {n_total}
Normal predictions: {n_normal} ({n_normal/n_total:.1%})
Abnormal predictions: {n_abnormal} ({n_abnormal/n_total:.1%})

CONFIDENCE ANALYSIS
Average confidence: {avg_confidence:.3f}
High confidence predictions (>={high_conf_threshold}): {n_high_conf} ({n_high_conf/n_total:.1%})

TOP ABNORMAL CASES (highest probability):
"""
        
        for idx, row in top_abnormal.iterrows():
            summary_text += f"  Row {int(row['row_index'])}: {row['prob_abnormal']:.3f} confidence\n"
        
        summary_text += f"""
RECOMMENDATIONS
• Normal cases: Review for routine follow-up
• Abnormal cases: Consider further cardiac evaluation  
• Low confidence predictions: May need additional testing
• Focus on high-confidence abnormal predictions for immediate review
"""
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary_text)
    
    def export_results(self):
        """Export results to CSV file"""
        if self.results_df is None:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Add timestamp and metadata to results
                export_df = self.results_df.copy()
                export_df['timestamp'] = datetime.now().isoformat()
                export_df['model_used'] = self.model_name
                export_df['source_file'] = self.file_path_var.get()
                
                # Reorder columns for better readability
                cols = ['row_index', 'prediction_label', 'confidence', 'prob_normal', 'prob_abnormal', 
                       'prediction', 'timestamp', 'model_used', 'source_file']
                export_df = export_df[cols]
                
                export_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Results exported to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def show_statistics(self):
        """Show detailed statistics in a new window"""
        if self.results_df is None:
            messagebox.showwarning("Warning", "No results available")
            return
        
        # Create statistics window
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Classification Statistics")
        stats_window.geometry("800x600")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(stats_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistics tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")
        
        stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate detailed statistics
        detailed_stats = self.generate_detailed_statistics()
        stats_text.insert(1.0, detailed_stats)
        
        # Visualization tab
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="Visualizations")
        
        # Create plots
        self.create_result_plots(viz_frame)
    
    def generate_detailed_statistics(self):
        """Generate detailed statistics text"""
        df = self.results_df
        
        # Basic statistics
        n_total = len(df)
        n_normal = len(df[df['prediction'] == 0])
        n_abnormal = len(df[df['prediction'] == 1])
        
        # Confidence statistics
        confidence_stats = df['confidence'].describe()
        
        # Probability statistics
        prob_normal_stats = df['prob_normal'].describe()
        prob_abnormal_stats = df['prob_abnormal'].describe()
        
        # Confidence bands
        high_conf = len(df[df['confidence'] >= 0.8])
        med_conf = len(df[(df['confidence'] >= 0.6) & (df['confidence'] < 0.8)])
        low_conf = len(df[df['confidence'] < 0.6])
        
        # Risk stratification
        high_risk = len(df[(df['prediction'] == 1) & (df['confidence'] >= 0.8)])
        medium_risk = len(df[(df['prediction'] == 1) & (df['confidence'] >= 0.6) & (df['confidence'] < 0.8)])
        uncertain_abnormal = len(df[(df['prediction'] == 1) & (df['confidence'] < 0.6)])
        
        stats_text = f"""DETAILED CLASSIFICATION STATISTICS
================================================

BASIC COUNTS
Total samples: {n_total}
Normal predictions: {n_normal} ({n_normal/n_total:.1%})
Abnormal predictions: {n_abnormal} ({n_abnormal/n_total:.1%})

CONFIDENCE DISTRIBUTION
High confidence (>=0.8): {high_conf} ({high_conf/n_total:.1%})
Medium confidence (0.6-0.8): {med_conf} ({med_conf/n_total:.1%})
Low confidence (<0.6): {low_conf} ({low_conf/n_total:.1%})

RISK STRATIFICATION (Abnormal Cases)
High-risk (Abnormal + High Confidence): {high_risk}
Medium-risk (Abnormal + Medium Confidence): {medium_risk}  
Uncertain abnormal (Low Confidence): {uncertain_abnormal}

CONFIDENCE STATISTICS
Mean: {confidence_stats['mean']:.4f}
Std:  {confidence_stats['std']:.4f}
Min:  {confidence_stats['min']:.4f}
25%:  {confidence_stats['25%']:.4f}
50%:  {confidence_stats['50%']:.4f}
75%:  {confidence_stats['75%']:.4f}
Max:  {confidence_stats['max']:.4f}

PROBABILITY STATISTICS - NORMAL
Mean: {prob_normal_stats['mean']:.4f}
Std:  {prob_normal_stats['std']:.4f}

PROBABILITY STATISTICS - ABNORMAL  
Mean: {prob_abnormal_stats['mean']:.4f}
Std:  {prob_abnormal_stats['std']:.4f}

CLINICAL INTERPRETATION
• High confidence abnormal cases need immediate review
• Medium confidence cases warrant closer monitoring  
• Low confidence predictions may benefit from additional testing
• Consider clinical context alongside these predictions
• This tool is for screening purposes - not a replacement for clinical judgment

FEATURE IMPORTANCE
The model uses {len(self.selected_features)} features including:
{', '.join(self.selected_features[:10])}{'...' if len(self.selected_features) > 10 else ''}
"""
        return stats_text
    
    def create_result_plots(self, parent_frame):
        """Create visualization plots"""
        if self.results_df is None:
            return
        
        # Create matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Prediction distribution
        pred_counts = self.results_df['prediction_label'].value_counts()
        colors = ['lightgreen' if label == 'Normal' else 'lightcoral' for label in pred_counts.index]
        axes[0,0].pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', 
                     startangle=90, colors=colors)
        axes[0,0].set_title('Prediction Distribution')
        
        # 2. Confidence histogram
        axes[0,1].hist(self.results_df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].axvline(0.8, color='red', linestyle='--', alpha=0.7, label='High Confidence Threshold')
        axes[0,1].axvline(0.6, color='orange', linestyle='--', alpha=0.7, label='Medium Confidence Threshold')
        axes[0,1].set_xlabel('Confidence Score')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Confidence Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Confidence by prediction
        normal_conf = self.results_df[self.results_df['prediction'] == 0]['confidence']
        abnormal_conf = self.results_df[self.results_df['prediction'] == 1]['confidence']
        
        if len(normal_conf) > 0:
            axes[1,0].hist(normal_conf, bins=15, alpha=0.7, label='Normal', color='green')
        if len(abnormal_conf) > 0:
            axes[1,0].hist(abnormal_conf, bins=15, alpha=0.7, label='Abnormal', color='red')
        
        axes[1,0].set_xlabel('Confidence Score')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Confidence by Prediction Type')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Probability scatter
        scatter = axes[1,1].scatter(
            self.results_df['prob_normal'], 
            self.results_df['prob_abnormal'],
            c=self.results_df['prediction'], 
            cmap='RdYlGn_r', 
            alpha=0.6,
            s=30
        )
        axes[1,1].set_xlabel('P(Normal)')
        axes[1,1].set_ylabel('P(Abnormal)')
        axes[1,1].set_title('Probability Space')
        axes[1,1].plot([0, 1], [1, 0], 'k--', alpha=0.5, linewidth=2, label='Decision Boundary')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1,1], label='Prediction (0=Normal, 1=Abnormal)')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def update_status(self, message):
        """Update status in the summary text"""
        current_time = datetime.now().strftime("%H:%M:%S")
        status_message = f"[{current_time}] {message}\n"
        
        # Insert at beginning of summary text
        current_text = self.summary_text.get(1.0, tk.END)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, status_message + current_text)
        self.root.update()

# Additional utility function to create sample CSV template
def create_sample_csv_template():
    """Create a sample CSV template for users"""
    # This would need to be customized based on your actual features
    # You can run this separately to generate a template file
    
    sample_features = {
        'heart_rate': [72, 85, 95, 68, 78],
        'rr_interval_mean': [0.83, 0.71, 0.63, 0.88, 0.77],
        'qrs_duration': [0.08, 0.12, 0.15, 0.09, 0.10],
        'qt_interval': [0.38, 0.42, 0.45, 0.36, 0.40],
        'pr_interval': [0.16, 0.18, 0.22, 0.15, 0.17]
        # Add more features as needed based on your model
    }
    
    df = pd.DataFrame(sample_features)
    df.to_csv('sample_ecg_template.csv', index=False)
    print("Sample CSV template created: sample_ecg_template.csv")

def main():
    # Create and run the GUI
    root = tk.Tk()
    app = ECGPredictor(root)
    root.mainloop()

if __name__ == "__main__":
    main()