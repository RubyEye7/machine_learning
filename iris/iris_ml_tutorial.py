"""
Iris Dataset Machine Learning Tutorial
=====================================

This script demonstrates fundamental machine learning concepts using the famous Iris dataset.
We'll cover data exploration, visualization, and multiple ML algorithms.

Author: ML Tutorial
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load the Iris dataset and perform initial exploration."""
    print("="*60)
    print("STEP 1: Loading and Exploring the Iris Dataset")
    print("="*60)
    
    # Load the Iris dataset
    iris = datasets.load_iris()
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nSpecies Distribution:")
    print(df['species_name'].value_counts())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return df, iris

def visualize_data(df):
    """Create visualizations to understand the data better."""
    print("\n" + "="*60)
    print("STEP 2: Data Visualization")
    print("="*60)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Iris Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Pairplot of features
    plt.subplot(2, 2, 1)
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species]
        plt.scatter(species_data['sepal length (cm)'], species_data['sepal width (cm)'], 
                   label=species, alpha=0.7)
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('Sepal Length vs Sepal Width')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Petal analysis
    plt.subplot(2, 2, 2)
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species]
        plt.scatter(species_data['petal length (cm)'], species_data['petal width (cm)'], 
                   label=species, alpha=0.7)
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.title('Petal Length vs Petal Width')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Box plot of sepal length
    plt.subplot(2, 2, 3)
    df.boxplot(column='sepal length (cm)', by='species_name', ax=axes[1,0])
    plt.title('Sepal Length Distribution by Species')
    plt.suptitle('')  # Remove automatic title
    
    # 4. Box plot of petal length
    plt.subplot(2, 2, 4)
    df.boxplot(column='petal length (cm)', by='species_name', ax=axes[1,1])
    plt.title('Petal Length Distribution by Species')
    plt.suptitle('')  # Remove automatic title
    
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[['sepal length (cm)', 'sepal width (cm)', 
                            'petal length (cm)', 'petal width (cm)']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def prepare_data(df, iris):
    """Prepare data for machine learning."""
    print("\n" + "="*60)
    print("STEP 3: Data Preparation")
    print("="*60)
    
    # Features and target
    X = iris.data
    y = iris.target
    
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print("Feature names:", iris.feature_names)
    print("Target names:", iris.target_names)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nData scaling completed!")
    print("Training data mean:", np.mean(X_train_scaled, axis=0).round(3))
    print("Training data std:", np.std(X_train_scaled, axis=0).round(3))
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def train_multiple_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train and evaluate multiple machine learning models."""
    print("\n" + "="*60)
    print("STEP 4: Training Multiple ML Models")
    print("="*60)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB()
    }
    
    # Models that work better with scaled data
    scaled_models = ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Choose appropriate data (scaled or unscaled)
        if name in scaled_models:
            train_X, test_X = X_train_scaled, X_test_scaled
        else:
            train_X, test_X = X_train, X_test
        
        # Train the model
        model.fit(train_X, y_train)
        
        # Make predictions
        y_pred = model.predict(test_X)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, train_X, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results

def evaluate_best_model(results, y_test, iris):
    """Evaluate the best performing model in detail."""
    print("\n" + "="*60)
    print("STEP 5: Detailed Model Evaluation")
    print("="*60)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model_info = results[best_model_name]
    
    print(f"Best Model: {best_model_name}")
    print(f"Test Accuracy: {best_model_info['accuracy']:.4f}")
    
    # Detailed classification report
    print(f"\nClassification Report for {best_model_name}:")
    print(classification_report(y_test, best_model_info['predictions'], 
                              target_names=iris.target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, best_model_info['predictions'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    return best_model_name, best_model_info

def model_comparison_visualization(results):
    """Visualize model performance comparison."""
    print("\n" + "="*60)
    print("STEP 6: Model Performance Comparison")
    print("="*60)
    
    # Extract results for plotting
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    cv_means = [results[name]['cv_mean'] for name in model_names]
    cv_stds = [results[name]['cv_std'] for name in model_names]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Test accuracy comparison
    bars1 = ax1.bar(model_names, accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Model Test Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Cross-validation scores with error bars
    bars2 = ax2.bar(model_names, cv_means, yerr=cv_stds, 
                    color='lightcoral', alpha=0.7, capsize=5)
    ax2.set_title('Cross-Validation Scores (5-fold)')
    ax2.set_ylabel('CV Score')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, mean in zip(bars2, cv_means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nModel Performance Summary:")
    print("-" * 50)
    for name in model_names:
        print(f"{name:20s}: Test={accuracies[model_names.index(name)]:.4f}, "
              f"CV={cv_means[model_names.index(name)]:.4f}")

def hyperparameter_tuning_example(X_train_scaled, y_train):
    """Demonstrate hyperparameter tuning with GridSearchCV."""
    print("\n" + "="*60)
    print("STEP 7: Hyperparameter Tuning Example")
    print("="*60)
    
    # Example with Random Forest
    print("Tuning Random Forest hyperparameters...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def make_predictions_on_new_data(best_model, scaler, iris):
    """Demonstrate how to make predictions on new data."""
    print("\n" + "="*60)
    print("STEP 8: Making Predictions on New Data")
    print("="*60)
    
    # Create some example new data points
    new_samples = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Should be setosa
        [6.2, 2.8, 4.8, 1.8],  # Should be versicolor  
        [7.2, 3.0, 5.8, 1.6]   # Should be virginica
    ])
    
    print("New samples to predict:")
    print("Sample 1: Sepal Length=5.1, Sepal Width=3.5, Petal Length=1.4, Petal Width=0.2")
    print("Sample 2: Sepal Length=6.2, Sepal Width=2.8, Petal Length=4.8, Petal Width=1.8")
    print("Sample 3: Sepal Length=7.2, Sepal Width=3.0, Petal Length=5.8, Petal Width=1.6")
    
    # Scale the new data
    new_samples_scaled = scaler.transform(new_samples)
    
    # Make predictions
    predictions = best_model.predict(new_samples_scaled)
    probabilities = best_model.predict_proba(new_samples_scaled)
    
    print("\nPredictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        species_name = iris.target_names[pred]
        confidence = prob.max()
        print(f"Sample {i+1}: {species_name} (confidence: {confidence:.3f})")
        print(f"  Probabilities: {dict(zip(iris.target_names, prob.round(3)))}")

def main():
    """Main function to run the complete ML tutorial."""
    print("🌸 IRIS DATASET MACHINE LEARNING TUTORIAL 🌸")
    print("This tutorial will teach you the fundamentals of machine learning!")
    
    # Step 1: Load and explore data
    df, iris = load_and_explore_data()
    
    # Step 2: Visualize data
    visualize_data(df)
    
    # Step 3: Prepare data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data(df, iris)
    
    # Step 4: Train multiple models
    results = train_multiple_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    
    # Step 5: Evaluate best model
    best_model_name, best_model_info = evaluate_best_model(results, y_test, iris)
    
    # Step 6: Visualize model comparison
    model_comparison_visualization(results)
    
    # Step 7: Hyperparameter tuning
    tuned_model = hyperparameter_tuning_example(X_train_scaled, y_train)
    
    # Step 8: Make predictions on new data
    make_predictions_on_new_data(best_model_info['model'], scaler, iris)
    
    print("\n" + "="*60)
    print("🎉 TUTORIAL COMPLETE! 🎉")
    print("="*60)
    print("You've successfully learned:")
    print("✅ Data loading and exploration")
    print("✅ Data visualization")
    print("✅ Data preprocessing and scaling")
    print("✅ Training multiple ML models")
    print("✅ Model evaluation and comparison")
    print("✅ Hyperparameter tuning")
    print("✅ Making predictions on new data")
    print("\nNext steps: Try modifying parameters, adding new models, or using different datasets!")

if __name__ == "__main__":
    main() 